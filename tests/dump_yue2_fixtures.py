#!/usr/bin/env python3
"""Dump YuE2 reference fixtures for mlx-serve's env-gated parity tests.

USER-RUN (needs torch + transformers + safetensors + network). Not run in CI.
Loads the m-a-p/YuE2-3B + m-a-p/YuE2-Vae checkpoints through their OWN torch
modeling code (the reference sources in --ref, default /tmp/yue2_ref) and
writes numpy fixtures that the Zig engine's `yue2` fixture test consumes via

  YUE2_TEST_MODEL=<converted pack dir> YUE2_FIXTURES=<model_dir>/fixtures

Files written under <out>/fixtures/:

  yue2_meta.json                     constants + seed/steps/frames + VAE geometry
  yue2_ar_prefill.i32.raw            the prefix token batch
  yue2_ar_decode16.i32.raw           prefix + 16 offset codec ids
  yue2_codec.i32.raw                 the OFFSET-SUBTRACTED codec ids (seed-fixed)
  yue2_ar_logits_prefill.f32         full last-position logits row [vocab] fp32
  yue2_ar_logits_decode16.f32        same after 16 offset codec ids are in context
  yue2_noise.f32                     the seeded CPU FP32 song noise [frames,64]
  yue2_velocity.f32                  first-chunk velocity probe at t=0.5 [frames,64]
  yue2_nar_latent.f32                CachedNAR.solve latents [frames,64] fp32
  yue2_vae_latent.f32                copy of the solved latents (what gets decoded)
  yue2_wav.f32                       decode_tiled(audio)[0].T interleaved [S,2] fp32

Everything is fed to the reference code as-is (reference `song_chunks` draws
the noise, `synthesize/CachedNAR.solve` solves, `nar.py.velocity` is probed,
`YuE2VAE.decode_tiled` decodes): no hand-transcribed math, so the fixture pins
exactly what a faithful port must reproduce (RoPE halves, the NAR per-head
q/k-norm + positional layout, the ODE midpoint, the VAE convT dependency crop).

The codec is OFFSET-SUBTRACTED (song_chunks' contract): chunk ar_tokens are
prefix + (id + CODEC_OFFSET) + [MUSIC_END]. AR probe `decode16` appends the
same offset codec so the offset region is exercised inside the AR context.
"""

import argparse
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def stage_ref(ref_dir: str) -> str:
    """Copy the flat reference package (relative imports) into a temp package."""
    src = Path(ref_dir)
    pkg = tempfile.mkdtemp(prefix="yue2ref-")
    (Path(pkg) / "__init__.py").write_text("")
    for py in src.glob("*.py"):
        shutil.copyfile(py, Path(pkg) / py.name)
    sys.path.insert(0, pkg)
    return pkg


def load_ref():
    protocol = importlib.import_module("protocol")
    modeling = importlib.import_module("modeling_yue2")
    vae_mod = importlib.import_module("modeling_vae")
    nar = importlib.import_module("nar")
    return protocol, modeling, vae_mod, nar


def tokenizer_from(model_dir: Path):
    """Reference YuE2TextTokenizer layout over the checkpoint's qwen.tiktoken."""
    import base64
    import tiktoken

    ranks = {base64.b64decode(t): int(r) for t, r in
             (line.split() for line in (model_dir / "qwen.tiktoken").read_bytes().splitlines() if line)}
    if len(ranks) != 151643:
        raise SystemExit(f"expected checkpoint-native qwen.tiktoken, got {len(ranks)} ranks")
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
    specials += [f"<extra_{i}>" for i in range(200)]
    specials[204:206] = ["<abc>", "</abc>"]
    pat = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    return tiktoken.Encoding("YuE2", pat_str=pat, mergeable_ranks=ranks,
                             special_tokens={s: i + len(ranks) for i, s in enumerate(specials)})


def dump_ar(model, out, prefix, codec, sample_n=16):
    import torch
    decode16 = list(prefix) + [c + 151853 for c in codec[:sample_n]]
    for name, ids in (("prefill", prefix), ("decode16", decode16)):
        np.asarray(ids, dtype=np.int32).tofile(out / f"yue2_ar_{name}.i32.raw")
        with torch.no_grad():
            log = model(input_ids=torch.tensor([ids], dtype=torch.long, device=model.device))
            row = log.logits[0, -1].detach().float().cpu().numpy()
        row.tofile(out / f"yue2_ar_logits_{name}.f32")
    np.asarray(codec, dtype=np.int32).tofile(out / "yue2_codec.i32.raw")
    print(f"[ar] prefill(len={len(prefix)}) + decode16 logits dumped ({row.shape[0]} vocab)")


def dump_nar(model, out, prefix, codec, seed, steps, device):
    _, _, _, nar = load_ref()
    chunks = nar.song_chunks(prefix, codec, seed, 24576)
    assert len(chunks) == 1  # frames << context, one original chunk
    chunk = chunks[0]
    chunk.noise.tofile(out / "yue2_noise.f32")

    engine = nar.CachedNAR(model, chunk, attention="sdpa")
    state = chunk.noise.to(device=device, dtype=next(model.vae2llm.parameters()).dtype)
    raw = 0.0  # logit(t=0.5), the midpoint probe
    velocity = engine.velocity(state, raw).float().cpu().numpy()
    engine.close()
    velocity.tofile(out / "yue2_velocity.f32")
    print(f"[nar] velocity probe @t=0.5 ({velocity.shape})")

    latents = nar.synthesize(model, prefix, codec, seed, steps=steps, context=24576)
    assert latents.shape == (len(codec), 64)
    latents.numpy().tofile(out / "yue2_nar_latent.f32")
    latents.numpy().tofile(out / "yue2_vae_latent.f32")
    print(f"[nar] solved latents {latents.shape} @ {steps} steps (dumped to vae_latent too)")


def dump_vae(vae, out, latents_path, meta):
    import torch
    latent = torch.tensor(np.fromfile(latents_path, dtype=np.float32), dtype=torch.float32)
    latent = latent.reshape(-1, 64).T.unsqueeze(0)  # [1,64,frames] as pipeline.decode
    with torch.inference_mode():
        audio = vae.decode_tiled(latent, core_frames=1024, halo_frames=16,
                                 output_device="cpu")
    meta["required_halo_1024"] = int(vae.required_halo())
    meta["natural_output_length"] = int(vae.natural_output_length(latent.shape[-1]))
    if not torch.isfinite(audio).all():
        raise SystemExit("VAE produced non-finite audio")
    wav = audio[0].float().clamp(-1, 1).T.contiguous().numpy()  # [S,2] interleaved
    wav.tofile(out / "yue2_wav.f32")
    print(f"[vae] decode_tiled -> {wav.shape[0]} frames ({wav.shape[0] / 48000:.1f}s)")


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="model root dir to write fixtures/ under")
    ap.add_argument("--ref", default=os.environ.get("YUE2_REF_DIR", "/tmp/yue2_ref"),
                    help="reference package dir (contains nar.py etc.)")
    ap.add_argument("--model", required=True,
                    help="LOCAL dir holding the YuE2-3B checkpoint + qwen.tiktoken")
    ap.add_argument("--vae", default="m-a-p/YuE2-Vae")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=1280,
                    help="codec frame count (1280 = two 1024-core VAE tiles)")
    ap.add_argument("--steps", type=int, default=8,
                    help="ODE steps (recorded in meta; engine uses the same)")
    ap.add_argument("--device", default=None, help="torch device (default: mps if available)")
    ap.add_argument("--ar-only", action="store_true")
    args = ap.parse_args()

    stage_ref(args.ref)
    protocol, modeling, vae_mod, nar = load_ref()
    CODEC_OFFSET = protocol.CODEC_OFFSET
    CODEC_SIZE = protocol.CODEC_SIZE
    CONTEXT = protocol.CONTEXT
    EOD, ABC_START, ABC_END, MUSIC_START, MUSIC_END = (
        protocol.EOD, protocol.ABC_START, protocol.ABC_END, protocol.MUSIC_START, protocol.MUSIC_END)

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16

    out = Path(args.out) / "fixtures"
    out.mkdir(parents=True, exist_ok=True)

    model_root = Path(args.model)
    if not (model_root / "qwen.tiktoken").exists():
        raise SystemExit(f"--model must be a local dir with qwen.tiktoken: {model_root}")
    model = modeling.YuE2ForCausalLM.from_pretrained(str(model_root), torch_dtype=dtype).to(device)
    model.eval()
    meta = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "constants": {"EOD": EOD, "ABC_START": ABC_START, "ABC_END": ABC_END,
                      "MUSIC_START": MUSIC_START, "MUSIC_END": MUSIC_END,
                      "CODEC_OFFSET": CODEC_OFFSET, "CODEC_SIZE": CODEC_SIZE,
                      "CONTEXT": CONTEXT},
        "seed": args.seed, "steps": args.steps, "frames": args.frames,
        "device": str(getattr(model, "device", device)),
    }

    tok = tokenizer_from(model_root)

    text = "Jazz ballad\n[Tags]\nslow, smoky\n[Lyrics]\nMoon over water\n"
    prefix = [EOD] + list(tok.encode(text)) + [ABC_START, ABC_END, MUSIC_START]

    rng = np.random.default_rng(args.seed)
    codec = [int(v) for v in rng.integers(0, CODEC_SIZE, args.frames, dtype=np.int64)]

    dump_ar(model, out, prefix, codec)
    meta["prefix_len"] = len(prefix)
    if not args.ar_only:
        dump_nar(model, out, prefix, codec, args.seed, args.steps, device)
        vae = vae_mod.YuE2VAE.from_pretrained(args.vae, decoder_only=True)
        vae.eval()
        dump_vae(vae, out, out / "yue2_vae_latent.f32", meta)

    (out / "yue2_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"fixtures under {out}")


if __name__ == "__main__":
    sys.exit(main())