#!/usr/bin/env python3
"""Convert YuE2-3B + YuE2-Vae to the mlx-serve pack layout (data, not weights).

OUTPUT directory (one pack):
  config.json                   model cfg + a "vae" section (parseCfg/parseVaeCfg)
  yue2_generation_config.json   copied if the AR repo ships one (else defaults)
  qwen.tiktoken                 the AR tokenizer (copied verbatim)
  ar.safetensors                AR partition  (model.* AR half + lm_head)   bf16
  nar.safetensors               NAR partition (nar_* halves + llm2vae/vae2llm
                                + time_embedder + latent_pos_embed.pe)      bf16
  vae.safetensors               YuE2-Vae decoder ONLY, weight-norm kept raw
                                (weight_g/weight_v; fused at engine load)    f32
                                — written LAST (the pack completion marker)

The engine (src/yue2.zig Engine.load) splits the two halves of every MoT layer:
the AR half reads ar.safetensors, the NAR half nar.safetensors. bf16 matches the
reference serving dtype (tests/dump_yue2_fixtures.py); the VAE requires f32.

Source keys are the exact YuE2ForCausalLM / YuE2VAE state_dicts:
  model.embed_tokens.weight, model.layers.{i}.{input_layernorm,self_attn,
  post_attention_layernorm,mlp,...} (AR), model.layers.{i}.{nar_input_layernorm,
  nar_self_attn,nar_pre_mlp_layernorm,nar_mlp,...} (NAR), model.norm.weight,
  lm_head.weight, llm2vae.{weight,bias}, vae2llm.{weight,bias},
  time_embedder.mlp.{0,2}.{weight,bias}, latent_pos_embed.pe,
  decoder.layers.*.{weight_g,weight_v,bias,alpha,beta}

Usage:
  python3 scripts/convert_yue2_weights.py --out <pack_dir> \
      [--ar m-a-p/YuE2-3B | <local dir>] [--vae m-a-p/YuE2-Vae | <local dir>]
  python3 scripts/convert_yue2_weights.py --self-test   # mapping logic only
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


# ── AR / NAR / VAE key partitioning (pure; --self-test pins it) ────────────

# NAR-only submodule markers inside a DecoderLayer.
_NAR_MARKERS = (".nar_input_layernorm.", ".nar_self_attn.",
                ".nar_pre_mlp_layernorm.", ".nar_mlp.")

# NAR auxiliary tensors live at the model root (not under model.*).
_NAR_ROOT = (
    "llm2vae.weight", "llm2vae.bias", "vae2llm.weight", "vae2llm.bias",
    "time_embedder.mlp.0.weight", "time_embedder.mlp.0.bias",
    "time_embedder.mlp.2.weight", "time_embedder.mlp.2.bias",
    "latent_pos_embed.pe",
)


def ar_out_key(k):
    """AR partition key, or None to drop."""
    if k == "lm_head.weight":
        return k
    if not k.startswith("model."):
        return None
    if any(m in k for m in _NAR_MARKERS):
        return None
    return k


def nar_out_key(k):
    """NAR partition key, or None to drop."""
    if k in _NAR_ROOT:
        return k
    if k.startswith("model.") and any(m in k for m in _NAR_MARKERS):
        return k
    return None


def vae_out_key(k):
    """VAE decoder key, or None to drop (the encoder is never served)."""
    return k if k.startswith("decoder.") else None


def split_state(state):
    """(ar, nar) dicts from one combined YuE2ForCausalLM state_dict."""
    ar, nar = {}, {}
    for k, v in state.items():
        a = ar_out_key(k)
        if a is not None:
            ar[a] = v
            continue
        n = nar_out_key(k)
        if n is not None:
            nar[n] = v
    return ar, nar


# ── IO (torch/safetensors; the conversion host has both) ────────────────────

def _resolve_dir(ref, allow_patterns):
    p = Path(ref).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(ref, allow_patterns=allow_patterns))


def _read_safetensors_dir(d, only=None):
    """Merge every .safetensors in `d` (sharded or single). `only(key)` filters."""
    import torch
    from safetensors import safe_open
    index = d / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = sorted(p.name for p in d.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no .safetensors in {d}")
    state = {}
    for name in files:
        with safe_open(d / name, framework="pt", device="cpu") as h:
            for key in h.keys():
                if only is not None and not only(key):
                    continue
                if key in state:
                    raise SystemExit(f"duplicate tensor {key}")
                state[key] = h.get_tensor(key)
    return state


def _save_bf16(state, path):
    import torch
    from safetensors.torch import save_file
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in state.items()}
    save_file(out, str(path))
    return len(out)


def _save_f32(state, path):
    from safetensors.torch import save_file
    out = {k: v.to(dtype=v.dtype).float().contiguous() for k, v in state.items()}
    save_file(out, str(path))
    return len(out)


def _vae_section(vae_cfg):
    dec = vae_cfg.get("decoder_config") or {}
    return {
        "channels": int(dec.get("channels", 64)),
        "c_mults": [int(x) for x in dec.get("c_mults", [1, 2, 4, 8, 16, 32])],
        "strides": [int(x) for x in dec.get("strides", [2, 2, 4, 4, 5, 6])],
        "latent_dim": int(dec.get("latent_dim", 64)),
        "out_channels": int(dec.get("out_channels", 2)),
        "use_snake": bool(dec.get("use_snake", True)),
        "final_tanh": bool(dec.get("final_tanh", False)),
        "sample_rate": int(vae_cfg.get("sample_rate", 48000)),
        "decode_core_frames": int(vae_cfg.get("decode_core_frames", 1024)),
        "decode_halo_frames": int(vae_cfg.get("decode_halo_frames", 16)),
    }


def _write_config(out, lm_cfg, vae_cfg):
    cfg = {
        "model_type": "yue2",
        "architectures": ["YuE2ForCausalLM"],
        "hidden_size": int(lm_cfg["hidden_size"]),
        "num_hidden_layers": int(lm_cfg["num_hidden_layers"]),
        "num_attention_heads": int(lm_cfg["num_attention_heads"]),
        "num_key_value_heads": int(lm_cfg["num_key_value_heads"]),
        "head_dim": int(lm_cfg["head_dim"]),
        "intermediate_size": int(lm_cfg["intermediate_size"]),
        "vocab_size": int(lm_cfg["vocab_size"]),
        "rms_norm_eps": float(lm_cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(lm_cfg.get("rope_theta", 1000000.0)),
        "max_position_embeddings": int(lm_cfg["max_position_embeddings"]),
        "latent_type": "vae",
        "latent_dim": int(lm_cfg.get("latent_dim", 64)),
        "max_latent_frames": int(lm_cfg.get("max_latent_frames", 24576)),
        "timestep_shift": float(lm_cfg.get("timestep_shift", 1.0)),
        "vae": _vae_section(vae_cfg),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def convert(ar_ref, vae_ref, out):
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ar_dir = _resolve_dir(ar_ref, ["*.json", "*.safetensors", "*.tiktoken"])
    vae_dir = _resolve_dir(vae_ref, ["*.json", "*.safetensors"])

    lm_cfg = json.loads((ar_dir / "config.json").read_text())
    vae_cfg = json.loads((vae_dir / "config.json").read_text())

    state = _read_safetensors_dir(ar_dir)
    ar, nar = split_state(state)
    if not ar or not nar:
        raise SystemExit(f"partition empty: ar={len(ar)} nar={len(nar)}")

    vae_state = _read_safetensors_dir(vae_dir, only=lambda k: k.startswith("decoder."))
    vae_state = {k: v for k, v in vae_state.items()}
    if not vae_state:
        raise SystemExit("no decoder.* tensors in the VAE checkpoint")

    # qwen.tiktoken: copy from the AR repo (required by the Zig tokenizer).
    tok_src = ar_dir / "qwen.tiktoken"
    if tok_src.exists():
        shutil.copyfile(tok_src, out / "qwen.tiktoken")
    else:
        raise SystemExit(f"qwen.tiktoken not found in {ar_dir}")

    gc = ar_dir / "yue2_generation_config.json"
    if gc.exists():
        shutil.copyfile(gc, out / "yue2_generation_config.json")

    _write_config(out, lm_cfg, vae_cfg)
    na = _save_bf16(ar, out / "ar.safetensors")
    nn = _save_bf16(nar, out / "nar.safetensors")
    # VAE LAST: its presence is the pack-completion marker.
    nv = _save_f32(vae_state, out / "vae.safetensors")
    print(f"[convert_yue2] wrote {out}: ar={na} nar={nn} vae={nv}")


# ── self-test (pure mapping; no torch needed) ───────────────────────────────

def _self_test():
    assert ar_out_key("model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert ar_out_key("model.norm.weight") == "model.norm.weight"
    assert ar_out_key("lm_head.weight") == "lm_head.weight"
    assert ar_out_key("model.layers.0.input_layernorm.weight") is not None
    assert ar_out_key("model.layers.0.self_attn.q_proj.weight") is not None
    assert ar_out_key("model.layers.0.post_attention_layernorm.weight") is not None
    assert ar_out_key("model.layers.0.mlp.down_proj.weight") is not None
    # NAR halves never leak into the AR map.
    assert ar_out_key("model.layers.0.nar_input_layernorm.weight") is None
    assert ar_out_key("model.layers.0.nar_self_attn.k_proj.weight") is None
    assert ar_out_key("llm2vae.weight") is None
    assert ar_out_key("decoder.layers.0.weight_g") is None

    assert nar_out_key("model.layers.0.nar_input_layernorm.weight") is not None
    assert nar_out_key("model.layers.0.nar_self_attn.q_norm.weight") is not None
    assert nar_out_key("model.layers.0.nar_mlp.up_proj.weight") is not None
    assert nar_out_key("llm2vae.weight") == "llm2vae.weight"
    assert nar_out_key("vae2llm.bias") == "vae2llm.bias"
    assert nar_out_key("time_embedder.mlp.0.weight") is not None
    assert nar_out_key("time_embedder.mlp.2.bias") is not None
    assert nar_out_key("latent_pos_embed.pe") == "latent_pos_embed.pe"
    assert nar_out_key("model.layers.0.self_attn.q_proj.weight") is None
    assert nar_out_key("model.embed_tokens.weight") is None

    assert vae_out_key("decoder.layers.0.weight_g") is not None
    assert vae_out_key("decoder.layers.7.layers.0.alpha") is not None
    assert vae_out_key("encoder.layers.0.weight_v") is None

    ar, nar = split_state({
        "model.embed_tokens.weight": 1,
        "model.layers.0.self_attn.q_proj.weight": 2,
        "model.layers.0.nar_self_attn.q_proj.weight": 3,
        "lm_head.weight": 4,
        "llm2vae.weight": 5,
    })
    assert set(ar) == {"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "lm_head.weight"}
    assert set(nar) == {"model.layers.0.nar_self_attn.q_proj.weight", "llm2vae.weight"}

    sec = _vae_section({"decoder_config": {"c_mults": [1, 2], "strides": [2, 2]}})
    assert sec["c_mults"] == [1, 2] and sec["latent_dim"] == 64
    print("[convert_yue2] self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--ar", default="m-a-p/YuE2-3B")
    ap.add_argument("--vae", default="m-a-p/YuE2-Vae")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.out:
        ap.error("--out is required (or use --self-test)")
    convert(args.ar, args.vae, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
