//! YuE2 (M-A-P) — AR–NAR Mixture-of-Transformers song generation on the
//! shared music route (`/v1/audio/music-generations`, `.yue2` AudioBackend).
//!
//! Port of the reference pipeline (multimodal-art-projection/YuE, Apache-2.0):
//! `protocol.py` (prompt + token contract), `sampling.py` (AR loop +
//! distribution), `nar.py` (`CachedNAR` flow matching — the memory-bounded
//! version, not `modeling_yue2.nar_velocity`), `modeling_yue2.py` (backbone)
//! and `modeling_vae.py` (YuE2-Vae Oobleck decoder, stable-audio-tools
//! lineage; SnakeBeta (c) NVIDIA, MIT).
//!
//! Weights arrive PRE-CONVERTED by scripts/convert_yue2_weights.py (one pack):
//!   config.json (model_type "yue2"), yue2_generation_config.json,
//!   qwen.tiktoken, ar.safetensors (AR partition), nar.safetensors (NAR
//!   partition), vae.safetensors (YuE2-Vae decoder, weight-norm SHIPPED
//!   verbatim and fused at ENGINE load, f32) — written LAST (the completion
//!   marker).
//!
//! Parity: env-gated YUE2_* oracles fed by tests/dump_yue2_fixtures.py.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const tok_mod = @import("tokenizer.zig");
const wav_mod = @import("wav.zig");
const sse = @import("gen_sse.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;

// ── protocol constants (protocol.py "yue2-native-v1"; checkpoint-native) ────

pub const EOD: u32 = 151643;
pub const ABC_START: u32 = 151847;
pub const ABC_END: u32 = 151848;
pub const MUSIC_START: u32 = 151851;
pub const MUSIC_END: u32 = 151852;
pub const CODEC_OFFSET: u32 = 151853;
pub const CODEC_SIZE: u32 = 32768;
pub const LATENT_START: u32 = 184621;
pub const LATENT_END: u32 = 184622;
pub const LATENT_PAD: u32 = 184623;
/// 25 latent frames per second (48 kHz / 1920 downsampling).
pub const FRAMES_PER_SECOND: u32 = 25;
pub const CONTEXT: u32 = 24576;
/// Semantic phase cap in the reference generation config = 360 s.
pub const MAX_SECONDS: u32 = 360;
pub const MIN_SECONDS: u32 = 1;

pub const Cot = enum {
    off,
    melody,
    full,

    pub fn parse(raw: []const u8) ?Cot {
        if (std.mem.eql(u8, raw, "off")) return .off;
        if (std.mem.eql(u8, raw, "melody")) return .melody;
        if (std.mem.eql(u8, raw, "full")) return .full;
        return null;
    }
};

/// protocol.py INSTRUCTIONS, byte-exact.
pub fn instructionFor(cot: Cot) []const u8 {
    return switch (cot) {
        .off => "Generate music with codec tokens from the given conditions.",
        .melody => "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
        .full => "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
    };
}

// ── sampling (protocol.py Sampling + yue2_generation_config.json) ──────────

pub const Sampling = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 0.95,
    top_k: u32 = 100,
    repetition_penalty: f32 = 1.2,
    penalty_window: u32 = 50,
    min_tokens: u32 = 200,
    max_tokens: u32 = 9000,
};

pub const ABC_SAMPLING = Sampling{
    .temperature = 0.7,
    .top_p = 0.9,
    .top_k = 30,
    .repetition_penalty = 1.005,
    .penalty_window = 100,
    .min_tokens = 32,
    .max_tokens = 4096,
};

pub const SEMANTIC_SAMPLING = Sampling{};
pub const DEFAULT_ODE_STEPS: u32 = 32;

pub const GenConfig = struct {
    abc: Sampling = ABC_SAMPLING,
    semantic: Sampling = SEMANTIC_SAMPLING,
    ode_steps: u32 = DEFAULT_ODE_STEPS,
};

/// `resolve_sampling` semantics: a dict OVERRIDES the defaults field-by-field;
/// absent/null keeps the default.
pub fn parseGenConfig(content: []const u8, defaults: GenConfig) !GenConfig {
    var out = defaults;
    var parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, content, .{}) catch return error.InvalidGenConfig;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidGenConfig;
    if (objField(parsed.value.object, "abc")) |v| {
        if (v != .object) return error.InvalidGenConfig;
        out.abc = try samplingFromObj(v.object, out.abc);
    }
    if (objField(parsed.value.object, "semantic")) |v| {
        if (v != .object) return error.InvalidGenConfig;
        out.semantic = try samplingFromObj(v.object, out.semantic);
    }
    if (objField(parsed.value.object, "ode_steps")) |v| {
        if (v != .integer or v.integer < 1) return error.InvalidGenConfig;
        out.ode_steps = @intCast(v.integer);
    }
    return out;
}

fn objField(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn samplingFromObj(obj: std.json.ObjectMap, d: Sampling) !Sampling {
    var out = d;
    if (objField(obj, "temperature")) |v| out.temperature = try jsonF32(v);
    if (objField(obj, "top_p")) |v| out.top_p = try jsonF32(v);
    if (objField(obj, "top_k")) |v| {
        if (v != .integer or v.integer < 1) return error.InvalidGenConfig;
        out.top_k = @intCast(v.integer);
    }
    if (objField(obj, "repetition_penalty")) |v| out.repetition_penalty = try jsonF32(v);
    if (objField(obj, "penalty_window")) |v| {
        if (v != .integer) return error.InvalidGenConfig;
        out.penalty_window = @intCast(v.integer);
    }
    if (objField(obj, "min_tokens")) |v| {
        if (v != .integer) return error.InvalidGenConfig;
        out.min_tokens = @intCast(v.integer);
    }
    if (objField(obj, "max_tokens")) |v| {
        if (v != .integer or v.integer < 1) return error.InvalidGenConfig;
        out.max_tokens = @intCast(v.integer);
    }
    if (out.temperature < 0 or out.temperature > 5 or out.top_p <= 0 or out.top_p > 1) return error.InvalidGenConfig;
    if (out.repetition_penalty <= 0 or out.penalty_window < 1 or out.penalty_window > 100) return error.InvalidGenConfig;
    if (out.min_tokens > out.max_tokens) return error.InvalidGenConfig;
    return out;
}

fn jsonU32(v: std.json.Value) !u32 {
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.InvalidConfig;
    return @intCast(v.integer);
}

fn jsonF32(v: std.json.Value) !f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => return error.InvalidGenConfig,
    };
    if (!std.math.isFinite(f)) return error.InvalidGenConfig;
    return f;
}

// ── request + prompt assembly (protocol.py SongRequest/token_prefixes) ────

pub const MusicRequest = struct {
    style: []const u8,
    lyrics: []const u8 = "",
    cot: Cot = .full,
    seed: u64 = 831001,
    /// User-supplied score (cot must be melody/full; overrides AR planning).
    abc: ?[]const u8 = null,
    cfg_scale: ?f32 = null,
    /// ODE steps (midpoint solver).
    steps: u32 = DEFAULT_ODE_STEPS,
    /// Semantic-phase cap in seconds (×25 frames), clamped [1, 360].
    max_seconds: u32 = MAX_SECONDS,
};

/// protocol.py SongRequest.guidance: cfg_scale override, else 1.01 for
/// cot=off (historical), 1.0 for symbolic modes.
pub fn guidanceOf(req: *const MusicRequest) f32 {
    if (req.cfg_scale) |g| return g;
    return if (req.cot == .off) 1.01 else 1.0;
}

/// The conditioning text: `<instruction>\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n`.
pub fn promptText(allocator: std.mem.Allocator, req: *const MusicRequest) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}\n[Tags]\n{s}\n[Lyrics]\n{s}\n", .{ instructionFor(req.cot), req.style, req.lyrics });
}

pub fn validateRequest(req: *const MusicRequest) !void {
    if (req.seed >= (1 << 63)) return error.InvalidSeed;
    if (req.abc) |abc| {
        if (req.cot == .off) return error.AbcWithCotOff;
        if (std.mem.trim(u8, abc, " \t\r\n").len == 0) return error.EmptyAbc;
    }
    if (req.cfg_scale) |g| {
        if (!std.math.isFinite(g) or g < 0 or g > 20) return error.InvalidCfgScale;
    }
}

/// token_prefixes: `base` is [EOD] ++ encode(prompt_text); `abc_ids` null =
/// the AR PLANNING prefix (stops after ABC_START), non-null = the full
/// score-conditioned prefix.
pub fn tokenPrefixes(allocator: std.mem.Allocator, base: []const u32, cot: Cot, abc_ids: ?[]const u32) ![]u32 {
    if (cot == .off) {
        const out = try allocator.alloc(u32, base.len + 3);
        @memcpy(out[0..base.len], base);
        out[base.len] = ABC_START;
        out[base.len + 1] = ABC_END;
        out[base.len + 2] = MUSIC_START;
        return out;
    }
    if (abc_ids) |ids| {
        var out = try allocator.alloc(u32, base.len + ids.len + 3);
        @memcpy(out[0..base.len], base);
        out[base.len] = ABC_START;
        @memcpy(out[base.len + 1 ..][0..ids.len], ids);
        out[base.len + 1 + ids.len] = ABC_END;
        out[base.len + 2 + ids.len] = MUSIC_START;
        return out;
    }
    const out = try allocator.alloc(u32, base.len + 1);
    @memcpy(out[0..base.len], base);
    out[base.len] = ABC_START;
    return out;
}

/// negative_prefix: the CFG unconditional branch. cot=off keeps only the
/// instruction; symbolic modes retain the EXACT positive-branch ABC IDs.
pub fn negativePrefix(allocator: std.mem.Allocator, instr_ids: []const u32, cot: Cot, abc_ids: []const u32) ![]u32 {
    if (cot == .off) {
        const out = try allocator.alloc(u32, instr_ids.len + 1);
        @memcpy(out[0..instr_ids.len], instr_ids);
        out[instr_ids.len] = MUSIC_START;
        return out;
    }
    var out = try allocator.alloc(u32, instr_ids.len + abc_ids.len + 3);
    @memcpy(out[0..instr_ids.len], instr_ids);
    out[instr_ids.len] = ABC_START;
    @memcpy(out[instr_ids.len + 1 ..][0..abc_ids.len], abc_ids);
    out[instr_ids.len + 1 + abc_ids.len] = ABC_END;
    out[instr_ids.len + 2 + abc_ids.len] = MUSIC_START;
    return out;
}

/// chunk_ranges: codec tokens per original acoustic chunk. `min((context -
/// prefix_tokens - 3) // 2, CONTEXT)` — one chunk's full sequence is
/// prefix + chunk + MUSIC_END + (T+2) NAR rows, so 2·size + 3 fits the
/// context. 0 = no room (caller refuses by name).
pub fn chunkSizeFor(prefix_tokens: usize, context: usize) usize {
    if (context <= prefix_tokens + 3) return 0;
    return @min((context - prefix_tokens - 3) / 2, CONTEXT);
}

// ── checkpoint config (config.json; YuE2-3B single-member family) ──────────

pub const Cfg = struct {
    layers: u32 = 28,
    hidden: u32 = 2048,
    heads: u32 = 16,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    inter: u32 = 6144,
    vocab: u32 = 184704,
    eps: f32 = 1e-6,
    rope_theta: f32 = 1_000_000.0,
    max_pos: u32 = 24576,
    latent_dim: u32 = 64,
    max_latent_frames: u32 = 24576,
    timestep_shift: f32 = 1.0,
};

const REAL_CONFIG_JSON =
    \\{"model_type":"yue2","architectures":["YuE2ForCausalLM"],"hidden_size":2048,"num_hidden_layers":28,
    \\ "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"intermediate_size":6144,
    \\ "vocab_size":184704,"rms_norm_eps":1e-06,"rope_theta":1000000.0,"max_position_embeddings":24576,
    \\ "latent_type":"vae","latent_dim":64,"max_latent_frames":24576,"timestep_shift":1.0}
;

pub fn parseCfg(content: []const u8) !Cfg {
    var parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, content, .{}) catch return error.InvalidConfig;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidConfig;
    const obj = parsed.value.object;
    var out = Cfg{};
    const lt = objField(obj, "latent_type") orelse return error.InvalidConfig;
    if (lt != .string or !std.mem.eql(u8, lt.string, "vae")) return error.UnsupportedLatentType;
    out.layers = try jsonU32Field(obj, "num_hidden_layers");
    out.hidden = try jsonU32Field(obj, "hidden_size");
    out.heads = try jsonU32Field(obj, "num_attention_heads");
    out.kv_heads = try jsonU32Field(obj, "num_key_value_heads");
    out.head_dim = try jsonU32Field(obj, "head_dim");
    out.inter = try jsonU32Field(obj, "intermediate_size");
    out.vocab = try jsonU32Field(obj, "vocab_size");
    out.max_pos = try jsonU32Field(obj, "max_position_embeddings");
    out.latent_dim = try jsonU32Field(obj, "latent_dim");
    out.max_latent_frames = try jsonU32Field(obj, "max_latent_frames");
    if (objField(obj, "rms_norm_eps")) |v| out.eps = try jsonF32(v);
    if (objField(obj, "rope_theta")) |v| out.rope_theta = try jsonF32(v);
    if (objField(obj, "timestep_shift")) |v| out.timestep_shift = try jsonF32(v);
    if (out.heads == 0 or out.kv_heads == 0 or out.heads % out.kv_heads != 0) return error.InvalidConfig;
    if (out.hidden != out.heads * out.head_dim) return error.InvalidConfig;
    if (out.latent_dim != 64 or out.vocab < CODEC_OFFSET + CODEC_SIZE) return error.InvalidConfig;
    if (out.rope_theta <= 0 or out.eps <= 0 or out.max_pos < 1024) return error.InvalidConfig;
    return out;
}

fn jsonU32Field(obj: std.json.ObjectMap, key: []const u8) !u32 {
    const v = objField(obj, key) orelse return error.InvalidConfig;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.InvalidConfig;
    return @intCast(v.integer);
}

// ════════════════════════════════════════════════════════════════════════
// MLX op helpers (the music3 set, [yue2]-tagged)
// ════════════════════════════════════════════════════════════════════════

fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}

fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}

fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}

fn addA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, x, y, s));
    return o;
}

fn subA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, x, y, s));
    return o;
}

fn mulA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, x, y, s));
    return o;
}

fn divA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&o, x, y, s));
    return o;
}

/// Scalar op in x's OWN dtype (a bare f32 scalar would promote bf16 operands
/// to f32 — the MageFlow scalarLike rule).
fn scalarLike(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(c);
    return astype(c, mlx.mlx_array_dtype(x), s);
}

fn mulScalar(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = try scalarLike(x, v, s);
    defer _ = mlx.mlx_array_free(c);
    return mulA(x, c, s);
}

fn sliceA(x: mlx.mlx_array, start: []const c_int, stop: []const c_int, s: S) !mlx.mlx_array {
    var strides: [8]c_int = .{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, start.ptr, start.len, stop.ptr, stop.len, strides[0..start.len].ptr, start.len, s));
    return o;
}

fn sliceUpdateA(dst: mlx.mlx_array, src: mlx.mlx_array, start: []const c_int, stop: []const c_int, s: S) !mlx.mlx_array {
    var strides: [8]c_int = .{ 1, 1, 1, 1, 1, 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice_update(&o, dst, src, start.ptr, start.len, stop.ptr, stop.len, strides[0..start.len].ptr, start.len, s));
    return o;
}

fn concat2(x: mlx.mlx_array, y: mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, x);
    _ = mlx.mlx_vector_array_append_value(vec, y);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}

fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, s));
    return mulA(x, sig, s);
}

fn expA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_exp(&o, x, s));
    return o;
}

fn sinA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sin(&o, x, s));
    return o;
}

fn tanhA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_tanh(&o, x, s));
    return o;
}

fn rmsNorm(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}

/// [B,T,H*hd] → [B,H,T,hd]
fn splitHeads(x: mlx.mlx_array, heads: c_int, hd: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const x4 = try reshape(x, &[_]c_int{ sh[0], sh[1], heads, hd }, s);
    defer _ = mlx.mlx_array_free(x4);
    return transpose(x4, &[_]c_int{ 0, 2, 1, 3 }, s);
}

/// [B,H,T,hd] → [B,T,H*hd]
fn mergeHeads(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const t = try transpose(x, &[_]c_int{ 0, 2, 1, 3 }, s);
    defer _ = mlx.mlx_array_free(t);
    return reshape(t, &[_]c_int{ sh[0], sh[2], sh[1] * sh[3] }, s);
}

/// MoT per-head RoPE with the checkpoint's HALF-split rotary
/// (modeling_yue2.RotaryEmbedding + _apply_rotary) = MLX `traditional`: the
/// reference applies cos/sin to the first/second HALF of each head's vector,
/// not interleaved pairs. offset = the AR cache length (= position of the
/// first NEW token; `cache_position` covers the whole new batch).
fn ropeAt(x: mlx.mlx_array, dims: c_int, theta: f32, offset: c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&o, x, dims, true, mlx.mlx_optional_float.some(theta), 1.0, offset, .{ .ctx = null }, s));
    return o;
}

fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mode: [*:0]const u8, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    const null_a = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, null_a, null_a, false, s));
    return o;
}

fn zerosA(shape: []const c_int, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, s));
    return o;
}

fn takeRows(table: mlx.mlx_array, ids: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&o, table, ids, 0, s));
    return o;
}

fn i32Arr(vals: []const i32) mlx.mlx_array {
    const sh = [_]c_int{@intCast(vals.len)};
    return mlx.mlx_array_new_data(vals.ptr, &sh, 1, .int32);
}

fn u32Arr(vals: []const u32) mlx.mlx_array {
    const sh = [_]c_int{@intCast(vals.len)};
    return mlx.mlx_array_new_data(vals.ptr, &sh, 1, .uint32);
}

/// Materialize a slice/transpose that outlives its parent (the sliceContig
/// rule): contiguous + eval breaks the graph edge to the parent buffer.
fn materialize(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    _ = mlx.mlx_array_eval(o);
    return o;
}

fn evalA(x: mlx.mlx_array) void {
    _ = mlx.mlx_array_eval(x);
}

fn readScalarI32(arr: mlx.mlx_array) !i32 {
    evalA(arr);
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, arr));
    return v;
}

fn getW(w: *const Weights, key: []const u8) !mlx.mlx_array {
    return w.get(key) orelse {
        log.err("[yue2] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingWeight;
    };
}

fn putWeight(w: *Weights, key: []const u8, arr: mlx.mlx_array) !void {
    const owned = try w.allocator.dupe(u8, key);
    try w.map.put(owned, arr);
}

fn removeWeight(w: *Weights, key: []const u8) void {
    if (w.map.fetchRemove(key)) |kv| {
        _ = mlx.mlx_array_free(kv.value);
        w.allocator.free(kv.key);
    }
}

// ════════════════════════════════════════════════════════════════════════
// Quantized linears (the music3 QLin: handles + geometry resolved at load)
// ════════════════════════════════════════════════════════════════════════

const QLin = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array = .{ .ctx = null },
    bi: mlx.mlx_array = .{ .ctx = null },
    bias: mlx.mlx_array = .{ .ctx = null },
    bits: u32 = 0,
    gs: u32 = 0,

    fn forward(self: *const QLin, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var o = mlx.mlx_array_new();
        if (self.sc.ctx != null) {
            try mlx.check(mlx.mlx_quantized_matmul(&o, x, self.w, self.sc, self.bi, true, mlx.mlx_optional_int.some(@intCast(self.gs)), mlx.mlx_optional_int.some(@intCast(self.bits)), "affine", s));
        } else {
            var wt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wt);
            const axes = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&wt, self.w, &axes, 2, s));
            try mlx.check(mlx.mlx_matmul(&o, x, wt, s));
        }
        if (self.bias.ctx != null) {
            const r = try addA(o, self.bias, s);
            _ = mlx.mlx_array_free(o);
            o = r;
        }
        return o;
    }
};

/// Resolve `prefix`.{weight,scales,biases,bias} once (the music3 contract:
/// quant geometry needs `in_features`).
fn resolveQLin(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32) !QLin {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const sk = try std.fmt.allocPrint(a, "{s}.scales", .{prefix});
    defer a.free(sk);
    const bk = try std.fmt.allocPrint(a, "{s}.biases", .{prefix});
    defer a.free(bk);
    const ak = try std.fmt.allocPrint(a, "{s}.bias", .{prefix});
    defer a.free(ak);

    var q = QLin{ .w = try getW(w, wk) };
    if (w.get(sk)) |scales| {
        q.sc = scales;
        q.bi = try getW(w, bk);
        const qp = transformer_mod.affineParamsFromGeometry(q.w, scales, in_features) orelse {
            log.err("[yue2] unsolvable quant geometry for {s}\n", .{prefix});
            return error.BadQuantGeometry;
        };
        q.bits = qp.bits;
        q.gs = qp.group_size;
    }
    if (w.get(ak)) |bias| q.bias = bias;
    return q;
}

/// One MoT layer's AR half (modeling_yue2.DecoderLayer, AR path).
const ArLayerW = struct {
    in_ln: mlx.mlx_array,
    pa_ln: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    q: QLin,
    k: QLin,
    v: QLin,
    o: QLin,
    gate: QLin,
    up: QLin,
    down: QLin,
};

/// One MoT layer's NAR half (nar_self_attn + nar_mlp behind their own norms).
const NarLayerW = struct {
    in_ln: mlx.mlx_array,
    pre_ln: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    q: QLin,
    k: QLin,
    v: QLin,
    o: QLin,
    gate: QLin,
    up: QLin,
    down: QLin,
};

fn buildArLayerW(w: *const Weights, a: std.mem.Allocator, cfg: Cfg, i: usize) !ArLayerW {
    var pbuf: [64]u8 = undefined;
    const pfx = try std.fmt.bufPrint(&pbuf, "model.layers.{d}", .{i});
    var lbuf: [80]u8 = undefined;
    const in_key = try std.fmt.bufPrint(&lbuf, "{s}.input_layernorm.weight", .{pfx});
    var lbuf2: [80]u8 = undefined;
    const pa_key = try std.fmt.bufPrint(&lbuf2, "{s}.post_attention_layernorm.weight", .{pfx});
    var nb1: [80]u8 = undefined;
    const qn_key = try std.fmt.bufPrint(&nb1, "{s}.self_attn.q_norm.weight", .{pfx});
    var nb2: [80]u8 = undefined;
    const kn_key = try std.fmt.bufPrint(&nb2, "{s}.self_attn.k_norm.weight", .{pfx});
    return .{
        .in_ln = try getW(w, in_key),
        .pa_ln = try getW(w, pa_key),
        .q_norm = try getW(w, qn_key),
        .k_norm = try getW(w, kn_key),
        .q = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.self_attn.q_proj", .{pfx}), cfg.hidden),
        .k = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.self_attn.k_proj", .{pfx}), cfg.hidden),
        .v = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.self_attn.v_proj", .{pfx}), cfg.hidden),
        .o = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.self_attn.o_proj", .{pfx}), cfg.hidden),
        .gate = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.mlp.gate_proj", .{pfx}), cfg.hidden),
        .up = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.mlp.up_proj", .{pfx}), cfg.hidden),
        .down = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.mlp.down_proj", .{pfx}), cfg.inter),
    };
}

fn buildNarLayerW(w: *const Weights, a: std.mem.Allocator, cfg: Cfg, i: usize) !NarLayerW {
    var pbuf: [64]u8 = undefined;
    const pfx = try std.fmt.bufPrint(&pbuf, "model.layers.{d}", .{i});
    var lbuf: [96]u8 = undefined;
    var lbuf2: [96]u8 = undefined;
    var nb1: [96]u8 = undefined;
    var nb2: [96]u8 = undefined;
    return .{
        .in_ln = try getW(w, try std.fmt.bufPrint(&lbuf, "{s}.nar_input_layernorm.weight", .{pfx})),
        .pre_ln = try getW(w, try std.fmt.bufPrint(&lbuf2, "{s}.nar_pre_mlp_layernorm.weight", .{pfx})),
        .q_norm = try getW(w, try std.fmt.bufPrint(&nb1, "{s}.nar_self_attn.q_norm.weight", .{pfx})),
        .k_norm = try getW(w, try std.fmt.bufPrint(&nb2, "{s}.nar_self_attn.k_norm.weight", .{pfx})),
        .q = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.nar_self_attn.q_proj", .{pfx}), cfg.hidden),
        .k = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.nar_self_attn.k_proj", .{pfx}), cfg.hidden),
        .v = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.nar_self_attn.v_proj", .{pfx}), cfg.hidden),
        .o = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.nar_self_attn.o_proj", .{pfx}), cfg.hidden),
        .gate = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.nar_mlp.gate_proj", .{pfx}), cfg.hidden),
        .up = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.nar_mlp.up_proj", .{pfx}), cfg.hidden),
        .down = try resolveQLin(w, a, try std.fmt.bufPrint(&lbuf, "{s}.nar_mlp.down_proj", .{pfx}), cfg.inter),
    };
}

// ════════════════════════════════════════════════════════════════════════
// VAE decoder (YuE2-Vae Oobleck; weight-norm fused + axes swapped at load)
// ════════════════════════════════════════════════════════════════════════

/// The pack's `config.json` "vae" section (the YuE2-Vae decoder_config).
pub const VaeCfg = struct {
    channels: u32 = 64,
    c_mults: [8]u32 = .{ 1, 2, 4, 8, 16, 32, 0, 0 },
    strides: [8]u32 = .{ 2, 2, 4, 4, 5, 6, 0, 0 },
    depth: u32 = 6,
    latent_dim: u32 = 64,
    out_channels: u32 = 2,
    use_snake: bool = true,
    final_tanh: bool = false,
    sample_rate: u32 = 48000,
    core_frames: u32 = 1024,
    halo_frames: u32 = 16,
};

fn parseVaeCfg(content: []const u8, cfg: Cfg) !VaeCfg {
    var parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, content, .{}) catch return error.InvalidConfig;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidConfig;
    const root = parsed.value.object;
    const vaev = objField(root, "vae") orelse return error.MissingVaeConfig;
    if (vaev != .object) return error.InvalidConfig;
    const obj = vaev.object;
    var out = VaeCfg{};
    out.channels = try jsonU32Field(obj, "channels");
    out.latent_dim = try jsonU32Field(obj, "latent_dim");
    out.out_channels = try jsonU32Field(obj, "out_channels");
    out.sample_rate = try jsonU32Field(obj, "sample_rate");
    if (objField(obj, "use_snake")) |v| {
        if (v != .bool) return error.InvalidConfig;
        out.use_snake = v.bool;
    }
    if (objField(obj, "final_tanh")) |v| {
        if (v != .bool) return error.InvalidConfig;
        out.final_tanh = v.bool;
    }
    if (objField(obj, "decode_core_frames")) |v| out.core_frames = try jsonU32(v);
    if (objField(obj, "decode_halo_frames")) |v| out.halo_frames = try jsonU32(v);
    const mults = objField(obj, "c_mults") orelse return error.InvalidConfig;
    const sts = objField(obj, "strides") orelse return error.InvalidConfig;
    if (mults != .array or sts != .array) return error.InvalidConfig;
    if (mults.array.items.len != sts.array.items.len or mults.array.items.len == 0 or mults.array.items.len > 8) return error.InvalidConfig;
    out.depth = @intCast(mults.array.items.len);
    for (mults.array.items, 0..) |m, i| {
        if (m != .integer) return error.InvalidConfig;
        out.c_mults[i] = @intCast(m.integer);
    }
    for (sts.array.items, 0..) |t, i| {
        if (t != .integer or t.integer < 1) return error.InvalidConfig;
        out.strides[i] = @intCast(t.integer);
    }
    if (out.latent_dim != cfg.latent_dim) return error.InvalidConfig;
    var prod: u64 = 1;
    for (out.strides[0..out.depth]) |t| prod *= t;
    if (prod == 0 or out.sample_rate % prod != 0) return error.InvalidConfig;
    return out;
}

const VaeConv = struct {
    w: mlx.mlx_array,
    bias: ?mlx.mlx_array,
    k: u32,
    pad: u32,
    dil: u32,
};

const VaeConvT = struct {
    w: mlx.mlx_array,
    bias: ?mlx.mlx_array,
    stride: u32,
    pad: u32,
};

/// SnakeBeta params [C] (alpha/beta in log-scale storage; exp at forward).
const VaeSnake = struct {
    alpha: mlx.mlx_array,
    beta: mlx.mlx_array,
};

const VaeResUnit = struct {
    s1: VaeSnake,
    c1: VaeConv, // k7, dilation d
    s2: VaeSnake,
    c2: VaeConv, // k1
};

const VaeBlock = struct {
    /// SnakeBeta BEFORE the upsample (on the block's in_channels).
    snake: VaeSnake,
    convt: VaeConvT,
    rus: [3]VaeResUnit,
};

const VaeDecoder = struct {
    conv_in: VaeConv, // latent_dim → c_mults[-1]*channels, k7 p3
    blocks: []VaeBlock, // depth blocks, deepest (i=depth) stride first
    final_snake: VaeSnake,
    final_conv: VaeConv, // → out_channels, k7 p3, NO bias

    fn deinit(self: *VaeDecoder, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }
};

/// One `decoder.layers.{b}.layers.{sub}` path in a stack buffer.
fn vaeKey(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
}

/// Fuse one weight-norm'd conv: w = g·v/‖v‖ (norm over the trailing (in,K)
/// axes), then PT → MLX axes: Conv1d [out,in,K] → [out,K,in], ConvTranspose1d
/// [in,out,K] → [out,K,in] (the music3 fuseVocoder math).
fn fuseVaeConv(w: *Weights, base: []const u8, is_tconv: bool, s: S) !struct { w: mlx.mlx_array, bias: ?mlx.mlx_array } {
    var gbuf: [96]u8 = undefined;
    var vbuf: [96]u8 = undefined;
    var bbuf: [96]u8 = undefined;
    var wbuf: [96]u8 = undefined;
    const gk = vaeKey(&gbuf, "{s}.weight_g", .{base});
    const vk = vaeKey(&vbuf, "{s}.weight_v", .{base});
    const g = try getW(w, gk);
    const v = try getW(w, vk);

    const sq = try mulA(v, v, s);
    defer _ = mlx.mlx_array_free(sq);
    var s2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s2);
    try mlx.check(mlx.mlx_sum_axis(&s2, sq, 2, true, s));
    var s1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s1);
    try mlx.check(mlx.mlx_sum_axis(&s1, s2, 1, true, s));
    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, s1, s));
    const gv = try mulA(g, v, s);
    defer _ = mlx.mlx_array_free(gv);
    const fused = try divA(gv, norm, s);
    defer _ = mlx.mlx_array_free(fused);

    const perm: []const c_int = if (is_tconv) &[_]c_int{ 1, 2, 0 } else &[_]c_int{ 0, 2, 1 };
    const t = try transpose(fused, perm, s);
    defer _ = mlx.mlx_array_free(t);
    // The map owns the fused weight and drops the raw pair (borrowed handles).
    const fused_w = try materialize(t, s);
    try putWeight(w, vaeKey(&wbuf, "{s}.weight", .{base}), fused_w);
    removeWeight(w, gk);
    removeWeight(w, vk);

    var bias: ?mlx.mlx_array = null;
    if (w.get(vaeKey(&bbuf, "{s}.bias", .{base}))) |b| bias = b;
    return .{ .w = fused_w, .bias = bias };
}

fn vaeSnakeAt(w: *Weights, base: []const u8) !VaeSnake {
    var abuf: [96]u8 = undefined;
    var bbuf: [96]u8 = undefined;
    return .{
        .alpha = try getW(w, vaeKey(&abuf, "{s}.alpha", .{base})),
        .beta = try getW(w, vaeKey(&bbuf, "{s}.beta", .{base})),
    };
}

fn vaeConvAt(w: *Weights, base: []const u8, is_tconv: bool, k: u32, pad: u32, dil: u32, s: S) !VaeConv {
    const fused = try fuseVaeConv(w, base, is_tconv, s);
    return .{ .w = fused.w, .bias = fused.bias, .k = k, .pad = pad, .dil = dil };
}

fn vaeConvTAt(w: *Weights, base: []const u8, s: S, stride: u32) !VaeConvT {
    const fused = try fuseVaeConv(w, base, true, s);
    const pad = (stride + 1) / 2; // torch: math.ceil(stride/2)
    return .{ .w = fused.w, .bias = fused.bias, .stride = stride, .pad = pad };
}

/// Resolve the decoder from the nested checkpoint layout
/// (OobleckDecoder.layers, source-exact):
///   layers.0            conv_in (latent_dim → c_mults[-1]*channels, k7 p3)
///   layers.{b}=1..depth DecoderBlock: layers.0 snake (in_channels),
///                       layers.1 convT (k=2*stride, p=ceil(stride/2)),
///                       layers.{2,3,4} ResidualUnit(d=1,3,9):
///                         layers.{0,2} snake, layers.{1} conv k7 p=3d,
///                         layers.{3} conv k1
///   layers.{depth+1}    final snake
///   layers.{depth+2}    final conv (→ out_channels, k7 p3, NO bias)
/// `c_mults` is the RAW config list; the decoder prepends 1 (block b uses
/// index i=depth-b+1). Every handle BORROWS from the map's fused entries.
fn resolveVaeDecoder(w: *Weights, allocator: std.mem.Allocator, vc: VaeCfg, s: S) !VaeDecoder {
    const conv_in = try vaeConvAt(w, "decoder.layers.0", false, 7, 3, 1, s);
    const blocks = try allocator.alloc(VaeBlock, vc.depth);
    errdefer allocator.free(blocks);
    var bbuf: [48]u8 = undefined;
    var rbuf: [64]u8 = undefined;
    for (0..vc.depth) |b| {
        const i = vc.depth - b; // depth..1 → c_mults index into the prepended list
        const stride = vc.strides[i - 1];
        const blk = vaeKey(&bbuf, "decoder.layers.{d}", .{b + 1});
        const snake = try vaeSnakeAt(w, vaeKey(&rbuf, "{s}.layers.0", .{blk}));
        const convt = try vaeConvTAt(w, vaeKey(&rbuf, "{s}.layers.1", .{blk}), s, stride);
        var rus: [3]VaeResUnit = undefined;
        for ([_]u32{ 1, 3, 9 }, 0..) |dil, r| {
            const ru = vaeKey(&rbuf, "{s}.layers.{d}", .{ blk, r + 2 });
            const s1 = try vaeSnakeAt(w, vaeKey(&rbuf, "{s}.layers.0", .{ru}));
            const c1 = try vaeConvAt(w, vaeKey(&rbuf, "{s}.layers.1", .{ru}), false, 7, 3 * dil, dil, s);
            const s2 = try vaeSnakeAt(w, vaeKey(&rbuf, "{s}.layers.2", .{ru}));
            const c2 = try vaeConvAt(w, vaeKey(&rbuf, "{s}.layers.3", .{ru}), false, 1, 0, 1, s);
            rus[r] = .{ .s1 = s1, .c1 = c1, .s2 = s2, .c2 = c2 };
        }
        blocks[b] = .{ .snake = snake, .convt = convt, .rus = rus };
    }
    const final_snake = try vaeSnakeAt(w, vaeKey(&rbuf, "decoder.layers.{d}", .{vc.depth + 1}));
    const final_conv = try vaeConvAt(w, vaeKey(&rbuf, "decoder.layers.{d}", .{vc.depth + 2}), false, 7, 3, 1, s);
    return .{ .conv_in = conv_in, .blocks = blocks, .final_snake = final_snake, .final_conv = final_conv };
}

// ════════════════════════════════════════════════════════════════════════
// VAE decode (YuE2-Vae Oobleck decoder; f32, exact-boundary tiled)
// ════════════════════════════════════════════════════════════════════════

/// SnakeBeta on channels-last [B,L,C]: x + sin(x·e^α)² / (e^β + 1e-9). The
/// stored α/β are log-scale; the exp is at FORWARD (modeling_vae α_logscale).
fn vaeSnakeF(x: mlx.mlx_array, sn: VaeSnake, s: S) !mlx.mlx_array {
    const al = try expA(sn.alpha, s);
    defer _ = mlx.mlx_array_free(al);
    const be = try expA(sn.beta, s);
    defer _ = mlx.mlx_array_free(be);
    const ax = try mulA(x, al, s);
    defer _ = mlx.mlx_array_free(ax);
    const sn_ = try sinA(ax, s);
    defer _ = mlx.mlx_array_free(sn_);
    const sq = try mulA(sn_, sn_, s);
    defer _ = mlx.mlx_array_free(sq);
    const eps = try scalarLike(be, 1e-9, s);
    defer _ = mlx.mlx_array_free(eps);
    const den = try addA(be, eps, s);
    defer _ = mlx.mlx_array_free(den);
    const frac = try divA(sq, den, s);
    defer _ = mlx.mlx_array_free(frac);
    return addA(x, frac, s);
}

fn vaeConvF(x: mlx.mlx_array, c: VaeConv, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv1d(&o, x, c.w, 1, @intCast(c.pad), @intCast(c.dil), 1, s));
    if (c.bias) |b| {
        const r = try addA(o, b, s);
        _ = mlx.mlx_array_free(o);
        o = r;
    }
    return o;
}

fn vaeConvTF(x: mlx.mlx_array, c: VaeConvT, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv_transpose1d(&o, x, c.w, @intCast(c.stride), @intCast(c.pad), 1, 0, 1, s));
    if (c.bias) |b| {
        const r = try addA(o, b, s);
        _ = mlx.mlx_array_free(o);
        o = r;
    }
    return o;
}

fn vaeResUnitF(x: mlx.mlx_array, ru: VaeResUnit, s: S) !mlx.mlx_array {
    const a1 = try vaeSnakeF(x, ru.s1, s);
    defer _ = mlx.mlx_array_free(a1);
    const c1 = try vaeConvF(a1, ru.c1, s);
    defer _ = mlx.mlx_array_free(c1);
    const a2 = try vaeSnakeF(c1, ru.s2, s);
    defer _ = mlx.mlx_array_free(a2);
    const c2 = try vaeConvF(a2, ru.c2, s);
    defer _ = mlx.mlx_array_free(c2);
    return addA(x, c2, s);
}

/// One full decoder pass: latents [1,T,64] f32 → [1,L,2] f32 (channels-last).
fn vaeForwardOne(e: *const Engine, latents: mlx.mlx_array, s: S) !mlx.mlx_array {
    var x = try astype(latents, .float32, s);
    x = blk: {
        const n = try vaeConvF(x, e.vae.conv_in, s);
        _ = mlx.mlx_array_free(x);
        break :blk n;
    };
    for (e.vae.blocks) |blk| {
        const n = try vaeSnakeF(x, blk.snake, s);
        _ = mlx.mlx_array_free(x);
        x = n;
        const c = try vaeConvTF(x, blk.convt, s);
        _ = mlx.mlx_array_free(x);
        x = c;
        for (blk.rus) |ru| {
            const r = try vaeResUnitF(x, ru, s);
            _ = mlx.mlx_array_free(x);
            x = r;
        }
    }
    x = blk: {
        const n = try vaeSnakeF(x, e.vae.final_snake, s);
        _ = mlx.mlx_array_free(x);
        break :blk n;
    };
    x = blk: {
        const n = try vaeConvF(x, e.vae.final_conv, s);
        _ = mlx.mlx_array_free(x);
        break :blk n;
    };
    // final_tanh=false in the released decoder; the reference applies it in
    // Decoder layers, so it is intentionally absent here.
    return x;
}

/// Geometric output length (modeling_vae._output_length): conv_in k7 p3 and
/// the final k7 p3 are length-preserving; each block's convT multiplies by
/// its stride (odd strides shed 1). 1920·T − 64 for the released geometry.
pub fn vaeOutputLength(vc: VaeCfg, frames: u32) u32 {
    var len: i64 = frames;
    var b: usize = 0;
    while (b < vc.depth) : (b += 1) {
        const stride: i64 = vc.strides[vc.depth - b - 1];
        const pad = @divTrunc(stride + 1, 2);
        len = (len - 1) * stride - 2 * pad + (2 * stride - 1) + 1;
    }
    return @intCast(len);
}

fn vaeRatio(vc: VaeCfg) u32 {
    var r: u32 = 1;
    for (vc.strides[0..vc.depth]) |st| r *= st;
    return r;
}

fn depConv(low: i64, high: i64, stride: i64, pad: i64, dil: i64, k: i64) [2]i64 {
    return .{ low * stride - pad, high * stride - pad + dil * (k - 1) };
}

fn depConvT(low: i64, high: i64, stride: i64, pad: i64, dil: i64, k: i64) [2]i64 {
    const x0 = low + pad - dil * (k - 1);
    return .{ -@divFloor(-x0, stride), @divFloor(high + pad, stride) };
}

/// modeling_vae.required_halo for the released decoder: the inclusive input
/// support of an output frame interval. A smallest halo that misses this
/// silently truncates a tile's receptive field.
pub fn vaeRequiredHalo(vc: VaeCfg) u32 {
    const core = vc.core_frames;
    const ratio = vaeRatio(vc);
    var lo: i64 = 0;
    var hi: i64 = @as(i64, core) * @as(i64, ratio) - 1;
    // Reversed traversal: final_conv, (snakes no-op), blocks depth-1..0, conv_in.
    const a = depConv(lo, hi, 1, 3, 1, 7);
    lo = a[0];
    hi = a[1];
    var b: usize = vc.depth;
    while (b > 0) {
        b -= 1;
        // blocks[] stores the deepest (stride strides[depth-1]) first; the
        // output-side traversal is blocks[depth-1]..blocks[0], so the stride
        // sequence is strides[0], strides[1], ..
        const stride: i64 = vc.strides[vc.depth - 1 - b];
        const pad = @divTrunc(stride + 1, 2);
        // Block reversed: ru9, ru3, ru1 (each k7 then k1 with min/max), convT.
        var r: usize = 3;
        while (r > 0) {
            r -= 1;
            const d: i64 = switch (r) {
                0 => 1,
                1 => 3,
                else => 9,
            };
            // ResidualUnit reversed: conv1 (k1) then conv7 (k7, p=3d, dil d).
            const c1 = depConv(lo, hi, 1, 0, 1, 1);
            const c7 = depConv(c1[0], c1[1], 1, 3 * d, d, 7);
            lo = @min(c7[0], lo);
            hi = @max(c7[1], hi);
        }
        const ct = depConvT(lo, hi, stride, pad, 1, 2 * stride);
        lo = ct[0];
        hi = ct[1];
    }
    const ci = depConv(lo, hi, 1, 3, 1, 7);
    lo = ci[0];
    hi = ci[1];
    const from_low: i64 = -lo;
    const from_high: i64 = hi - @as(i64, core) + 1;
    return @intCast(@max(0, @max(from_low, from_high)));
}

const TileCrop = struct { crop_start: i64, out_start: i64, out_len: i64 };

/// Per-tile crop geometry: the tile covers latent frames [left,right); we copy
/// its core [start,end) to output offset start·ratio. `-64` is absorbed on the
/// last tile because out_end clamps to the natural length.
fn vaeTileCrop(start: usize, end: usize, left: usize, ratio: u32, total: u32) TileCrop {
    const r: i64 = ratio;
    const out_start: i64 = @as(i64, @intCast(start)) * r;
    const out_end = @min(@as(i64, @intCast(end)) * r, @as(i64, total));
    return .{
        .crop_start = @as(i64, @intCast(start - left)) * r,
        .out_start = out_start,
        .out_len = out_end - out_start,
    };
}

/// Exact-boundary tiled decode. `latents` is row-major [frames,64] f32; the
/// return is interleaved stereo [S,2] f32 (S = vaeOutputLength). Cores are
/// copied with no crossfade, matching modeling_vae.decode_tiled.
pub fn vaeDecodeTiled(e: *const Engine, allocator: std.mem.Allocator, latents: []const f32, frames: usize, s: S) ![]f32 {
    if (frames == 0) return error.NoAudioFrames;
    if (latents.len != frames * 64) return error.BadLatentShape;
    const vc = e.vcfg;
    if (vc.halo_frames < vaeRequiredHalo(vc)) return error.InsufficientHalo;
    if (vc.core_frames == 0) return error.InvalidConfig;
    const ratio = vaeRatio(vc);
    const total: usize = vaeOutputLength(vc, @intCast(frames));
    if (total == 0) return error.NoAudioFrames;
    const out = try allocator.alloc(f32, total * 2);
    errdefer allocator.free(out);
    var start: usize = 0;
    while (start < frames) : (start += vc.core_frames) {
        const end = @min(frames, start + vc.core_frames);
        const left = if (start > vc.halo_frames) start - vc.halo_frames else 0;
        const right = @min(frames, end + vc.halo_frames);
        const tile_frames = right - left;
        const sh = [_]c_int{ 1, @intCast(tile_frames), 64 };
        const in = mlx.mlx_array_new_data(latents[left * 64 .. right * 64].ptr, &sh, 3, .float32);
        defer _ = mlx.mlx_array_free(in);
        const tile = try vaeForwardOne(e, in, s);
        defer _ = mlx.mlx_array_free(tile);
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, tile, false, s));
        evalA(c);
        const l: i64 = mlx.getShape(c)[1];
        const d = mlx.mlx_array_data_float32(c) orelse return error.NoData;
        const cp = vaeTileCrop(start, end, left, ratio, @intCast(total));
        if (cp.crop_start + cp.out_len > l) return error.VaeTileShort;
        var j: i64 = 0;
        while (j < cp.out_len) : (j += 1) {
            const si: usize = @intCast(cp.crop_start + j);
            const di: usize = @intCast(cp.out_start + j);
            out[di * 2] = d[si * 2];
            out[di * 2 + 1] = d[si * 2 + 1];
        }
        _ = mlx.mlx_clear_cache();
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Engine
// ════════════════════════════════════════════════════════════════════════

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: S,
    cfg: Cfg,
    gen: GenConfig,
    vcfg: VaeCfg,
    ar_w: Weights,
    nar_w: Weights,
    vae_w: Weights,
    tok: tok_mod.Tokenizer,
    /// Borrowed from ar_w (freed only via the maps).
    ar_lw: []ArLayerW,
    nar_lw: []NarLayerW,
    lm_head: QLin,
    embed: mlx.mlx_array,
    final_norm_w: mlx.mlx_array,
    llm2vae: QLin,
    vae2llm: QLin,
    time1: QLin,
    time2: QLin,
    pos_table: mlx.mlx_array,
    vae: VaeDecoder,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.io = io;
        self.s = mlx.mlx_default_gpu_stream_new();

        const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
        defer allocator.free(cfg_path);
        const cfg_bytes = try readFile(io, allocator, cfg_path);
        defer allocator.free(cfg_bytes);
        self.cfg = try parseCfg(cfg_bytes);
        self.vcfg = try parseVaeCfg(cfg_bytes, self.cfg);

        var gen = GenConfig{};
        if (std.fmt.allocPrint(allocator, "{s}/yue2_generation_config.json", .{model_dir})) |gc_path| {
            defer allocator.free(gc_path);
            if (readFile(io, allocator, gc_path)) |gc| {
                defer allocator.free(gc);
                gen = parseGenConfig(gc, gen) catch blk: {
                    log.warn("[yue2] unreadable yue2_generation_config.json; using defaults\n", .{});
                    break :blk GenConfig{};
                };
            } else |_| {}
        } else |_| {}
        self.gen = gen;

        self.ar_w = try loadFileWeights(allocator, model_dir, "ar.safetensors");
        errdefer self.ar_w.deinit();
        self.nar_w = try loadFileWeights(allocator, model_dir, "nar.safetensors");
        errdefer self.nar_w.deinit();
        self.vae_w = try loadFileWeights(allocator, model_dir, "vae.safetensors");
        errdefer self.vae_w.deinit();

        self.embed = try getW(&self.ar_w, "model.embed_tokens.weight");
        self.final_norm_w = try getW(&self.ar_w, "model.norm.weight");
        self.lm_head = try resolveQLin(&self.ar_w, allocator, "lm_head", self.cfg.hidden);
        self.llm2vae = try resolveQLin(&self.nar_w, allocator, "llm2vae", self.cfg.hidden);
        self.vae2llm = try resolveQLin(&self.nar_w, allocator, "vae2llm", self.cfg.latent_dim);
        self.time1 = try resolveQLin(&self.nar_w, allocator, "time_embedder.mlp.0", 256);
        self.time2 = try resolveQLin(&self.nar_w, allocator, "time_embedder.mlp.2", self.cfg.hidden);
        self.pos_table = try getW(&self.nar_w, "latent_pos_embed.pe");

        self.ar_lw = try allocator.alloc(ArLayerW, self.cfg.layers);
        errdefer allocator.free(self.ar_lw);
        self.nar_lw = try allocator.alloc(NarLayerW, self.cfg.layers);
        errdefer allocator.free(self.nar_lw);
        for (0..self.cfg.layers) |i| {
            self.ar_lw[i] = try buildArLayerW(&self.ar_w, allocator, self.cfg, i);
            self.nar_lw[i] = try buildNarLayerW(&self.nar_w, allocator, self.cfg, i);
        }

        self.vae = try resolveVaeDecoder(&self.vae_w, allocator, self.vcfg, self.s);

        self.tok = try tok_mod.loadTokenizerTiktoken(io, allocator, model_dir);
        log.info("[yue2] engine ready (ar {d} + nar {d} + vae {d} tensors, {d} layers)\n", .{
            self.ar_w.count(), self.nar_w.count(), self.vae_w.count(), self.cfg.layers,
        });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.ar_w.deinit();
        self.nar_w.deinit();
        self.vae_w.deinit();
        self.allocator.free(self.ar_lw);
        self.allocator.free(self.nar_lw);
        self.vae.deinit(self.allocator);
        self.tok.deinit();
        self.allocator.destroy(self);
    }
};

// ════════════════════════════════════════════════════════════════════════
// AR generation (sampling.py _generate → generate_tokens)
// ════════════════════════════════════════════════════════════════════════

pub const Phase = enum { abc, semantic };

/// Deterministic per-draw keys: seed advances by a golden-ratio stride per
/// draw (the music3 Sampler contract). torch's multinomial stream is not
/// reproducible, so SAMPLED output is un-pinned; greedy runs the host argmax
/// below and is the fixture-parity path.
const Sampler = struct {
    seed: u64,
    ctr: u64 = 0,

    fn nextKey(self: *Sampler) mlx.mlx_array {
        var k = mlx.mlx_array_new();
        _ = mlx.mlx_random_key(&k, self.seed +% self.ctr *% 0x9E3779B97F4A7C15);
        self.ctr += 1;
        return k;
    }
};

/// Exact counts of the last `window` history ids (sampling.py window_penalty).
/// One decrement per outgoing id instead of a 184704-wide recount per step.
const WindowRing = struct {
    ids: []i32,
    freq: []u32,
    start: usize = 0,
    len: usize = 0,

    fn init(allocator: std.mem.Allocator, window: u32, vocab: u32) !WindowRing {
        if (window == 0) return error.InvalidWindow;
        const freq = try allocator.alloc(u32, vocab);
        errdefer allocator.free(freq);
        @memset(freq, 0);
        return .{ .ids = try allocator.alloc(i32, window), .freq = freq };
    }

    fn deinit(self: *WindowRing, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.freq);
    }

    fn push(self: *WindowRing, tok: i32) void {
        self.freq[@intCast(tok)] += 1;
        if (self.len == self.ids.len) {
            const old = self.ids[self.start];
            self.freq[@intCast(old)] -= 1;
            self.ids[self.start] = tok;
            self.start = (self.start + 1) % self.ids.len;
        } else {
            self.ids[(self.start + self.len) % self.ids.len] = tok;
            self.len += 1;
        }
    }
};

/// One AR branch's KV cache, [1, kv_heads, cap, head_dim] bf16. CFG runs the
/// branches as SEPARATE forwards + caches because the positive and negative
/// prefixes differ in length — the reference's eager two-cache path (the cuda
/// graph batches only equal-length branches).
const LmKv = struct {
    ks: []mlx.mlx_array,
    vs: []mlx.mlx_array,
    len: c_int = 0,

    fn init(allocator: std.mem.Allocator, cfg: Cfg, cap: u32, s: S) !LmKv {
        const ks = try allocator.alloc(mlx.mlx_array, cfg.layers);
        errdefer allocator.free(ks);
        const vs = try allocator.alloc(mlx.mlx_array, cfg.layers);
        errdefer allocator.free(vs);
        const shape = [_]c_int{ 1, @intCast(cfg.kv_heads), @intCast(cap), @intCast(cfg.head_dim) };
        for (ks, vs) |*k, *v| {
            k.* = try zerosA(&shape, .bfloat16, s);
            v.* = try zerosA(&shape, .bfloat16, s);
        }
        return .{ .ks = ks, .vs = vs };
    }

    fn deinit(self: *LmKv, allocator: std.mem.Allocator) void {
        for (self.ks) |k| _ = mlx.mlx_array_free(k);
        for (self.vs) |v| _ = mlx.mlx_array_free(v);
        allocator.free(self.ks);
        allocator.free(self.vs);
    }

    /// Write `new` [1,KV,T,hd] at position `len` and swap the buffer (the
    /// free-after-update pattern: a captured view must be taken AFTER this).
    fn append(self: *LmKv, li: usize, is_k: bool, new: mlx.mlx_array, s: S) !void {
        const buf = if (is_k) &self.ks[li] else &self.vs[li];
        const sh = mlx.getShape(buf.*);
        const t = mlx.getShape(new)[2];
        const start = [_]c_int{ 0, 0, self.len, 0 };
        const stop = [_]c_int{ sh[0], sh[1], self.len + t, sh[3] };
        const updated = try sliceUpdateA(buf.*, new, &start, &stop, s);
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }

    fn view(self: *const LmKv, li: usize, is_k: bool, upto: c_int, s: S) !mlx.mlx_array {
        const buf = if (is_k) self.ks[li] else self.vs[li];
        const sh = mlx.getShape(buf);
        return sliceA(buf, &[_]c_int{ 0, 0, 0, 0 }, &[_]c_int{ sh[0], sh[1], upto, sh[3] }, s);
    }
};

/// Per-layer AR k/v views retained from one causal prefill (nar.py
/// `_prefill`; `visible_length` = ar_length when nar_cond_end = 0). The
/// views are lazy slices of the LmKv buffers — both must outlive the NAR ODE.
const CachedAr = struct {
    k: []mlx.mlx_array,
    v: []mlx.mlx_array,

    fn init(allocator: std.mem.Allocator, layers: usize) !CachedAr {
        return .{ .k = try allocator.alloc(mlx.mlx_array, layers), .v = try allocator.alloc(mlx.mlx_array, layers) };
    }

    fn deinit(self: *CachedAr, allocator: std.mem.Allocator) void {
        for (self.k) |k| _ = mlx.mlx_array_free(k);
        for (self.v) |v| _ = mlx.mlx_array_free(v);
        allocator.free(self.k);
        allocator.free(self.v);
    }
};

/// One trunk forward over [1,T,hidden] embeds. Appends per-layer k/v to `kv`
/// and returns the post-last-layer residual [1,T,hidden] (NO final norm), the
/// modeling_yue2.DecoderLayer chain acting on `input_layernorm(x)`. `capture`
/// (NAR path) additionally keeps each layer's readable k/v view.
fn trunkForward(e: *const Engine, embeds: mlx.mlx_array, kv: *LmKv, capture: ?*CachedAr, s: S) !mlx.mlx_array {
    const cfg = e.cfg;
    const t_len: c_int = mlx.getShape(embeds)[1];
    const nh: c_int = @intCast(cfg.heads);
    const nkv: c_int = @intCast(cfg.kv_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const prefill = t_len > 1;

    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&h, embeds));
    errdefer _ = mlx.mlx_array_free(h);
    for (e.ar_lw, 0..) |*lw, li| {
        const x = try rmsNorm(h, lw.in_ln, cfg.eps, s);
        defer _ = mlx.mlx_array_free(x);

        const q = try lw.q.forward(x, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try lw.k.forward(x, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try lw.v.forward(x, s);
        defer _ = mlx.mlx_array_free(v);
        const qh = try splitHeads(q, nh, hd, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try splitHeads(k, nkv, hd, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try splitHeads(v, nkv, hd, s);
        defer _ = mlx.mlx_array_free(vh);

        const qn = try rmsNorm(qh, lw.q_norm, cfg.eps, s);
        defer _ = mlx.mlx_array_free(qn);
        const kn = try rmsNorm(kh, lw.k_norm, cfg.eps, s);
        defer _ = mlx.mlx_array_free(kn);

        const qr = try ropeAt(qn, hd, cfg.rope_theta, kv.len, s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try ropeAt(kn, hd, cfg.rope_theta, kv.len, s);
        defer _ = mlx.mlx_array_free(kr);

        try kv.append(li, true, kr, s);
        try kv.append(li, false, vh, s);
        const kview = try kv.view(li, true, kv.len + t_len, s);
        defer _ = mlx.mlx_array_free(kview);
        const vview = try kv.view(li, false, kv.len + t_len, s);
        defer _ = mlx.mlx_array_free(vview);

        const attn = try sdpa(qr, kview, vview, scale, if (prefill) "causal" else "", s);
        defer _ = mlx.mlx_array_free(attn);
        const merged = try mergeHeads(attn, s);
        defer _ = mlx.mlx_array_free(merged);
        const o = try lw.o.forward(merged, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(h, o, s);
        _ = mlx.mlx_array_free(h);
        h = h1;

        const xm = try rmsNorm(h, lw.pa_ln, cfg.eps, s);
        defer _ = mlx.mlx_array_free(xm);
        const gate = try lw.gate.forward(xm, s);
        defer _ = mlx.mlx_array_free(gate);
        const up = try lw.up.forward(xm, s);
        defer _ = mlx.mlx_array_free(up);
        const gact = try silu(gate, s);
        defer _ = mlx.mlx_array_free(gact);
        const gu = try mulA(gact, up, s);
        defer _ = mlx.mlx_array_free(gu);
        const down = try lw.down.forward(gu, s);
        defer _ = mlx.mlx_array_free(down);
        const h2 = try addA(h, down, s);
        _ = mlx.mlx_array_free(h);
        h = h2;

        if (capture) |cap| {
            cap.k[li] = try kv.view(li, true, kv.len + t_len, s);
            cap.v[li] = try kv.view(li, false, kv.len + t_len, s);
        }
        if (prefill) evalA(h);
    }
    kv.len += t_len;
    return h;
}

/// Last position's post-norm hidden [1,hidden] (model.norm.weight from ar_w).
fn finalHidden(e: *const Engine, h: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(h); // [1,T,hidden]
    const last = try sliceA(h, &[_]c_int{ 0, sh[1] - 1, 0 }, &[_]c_int{ 1, sh[1], sh[2] }, s);
    defer _ = mlx.mlx_array_free(last);
    const flat = try reshape(last, &[_]c_int{ 1, sh[2] }, s);
    defer _ = mlx.mlx_array_free(flat);
    return rmsNorm(flat, e.final_norm_w, e.cfg.eps, s);
}

/// lm_head over the last hidden → [1,vocab] bf16 logits.
fn lmHeadLogits(e: *const Engine, last_hidden: mlx.mlx_array, s: S) !mlx.mlx_array {
    return e.lm_head.forward(last_hidden, s);
}

/// Whole-sequence forward: ids → embedded → trunk → last hidden → logits.
fn trunkLogits(e: *const Engine, a: std.mem.Allocator, ids: []const u32, kv: *LmKv, capture: ?*CachedAr, s: S) !mlx.mlx_array {
    const ids32 = try a.alloc(i32, ids.len);
    for (ids, ids32) |u, *d| d.* = @intCast(u);
    const shape = [_]c_int{ 1, @intCast(ids.len) };
    const id_arr = mlx.mlx_array_new_data(ids32.ptr, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(id_arr);
    const embeds = try takeRows(e.embed, id_arr, s);
    defer _ = mlx.mlx_array_free(embeds);
    const h = try trunkForward(e, embeds, kv, capture, s);
    defer _ = mlx.mlx_array_free(h);
    const last = try finalHidden(e, h, s);
    defer _ = mlx.mlx_array_free(last);
    return lmHeadLogits(e, last, s);
}

/// Read a bf16 [1,V] logits row to an f32 buffer (eval + contiguous, CPU).
fn readLogitsRow(arr: mlx.mlx_array, out: []f32, s: S) !void {
    _ = s;
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    const cpu = mlx.mlx_default_cpu_stream_new();
    const f = try astype(arr, .float32, cpu);
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, cpu));
    evalA(c);
    const d = mlx.mlx_array_data_float32(c) orelse return error.NoData;
    @memcpy(out, d[0..out.len]);
}

/// torch argmax: FIRST index of the maximum value.
fn argmaxF32(vals: []const f32) i32 {
    var best: i32 = 0;
    var bestv = -std.math.inf(f32);
    for (vals, 0..) |v, i| {
        if (v > bestv) {
            bestv = v;
            best = @intCast(i);
        }
    }
    return best;
}

/// Lomuto quickselect for the `k0`-th smallest (0-based), in place, over a
/// PRIVATE copy. Deterministic; O(V) average.
fn kthSmallest(copy: []f32, k0: usize) f32 {
    var lo: usize = 0;
    var hi: usize = copy.len;
    while (lo + 1 < hi) {
        const pv = copy[lo];
        var store: usize = lo;
        var i = lo + 1;
        while (i < hi) : (i += 1) {
            if (copy[i] < pv) {
                store += 1;
                const t = copy[i];
                copy[i] = copy[store];
                copy[store] = t;
            }
        }
        const t = copy[lo];
        copy[lo] = copy[store];
        copy[store] = t;
        if (k0 < store) {
            hi = store;
        } else if (k0 > store) {
            lo = store + 1;
        } else {
            return copy[store];
        }
    }
    return copy[lo];
}

/// sampling.py distribution(), host-f32. symbolic runs the reference's FP32
/// arithmetic exactly (the fixture parity path); legacy_off (cot=off) ran in
/// BF16 upstream — the f32 divergence is accepted because off mode is not
/// byte-pinned. `freq` = WindowRing counts (empty → penalty skipped).
/// `scratch` [V] and `cand` [V] are the caller's reused buffers.
/// Returns in `scores` (in place); the caller still owns the buffer.
fn distributionHost(scores: []f32, scratch: []f32, cand: []usize, sampling: Sampling, step: u32, phase: Phase, legacy_off: bool, freq: []const u32) void {
    const end: u32 = if (phase == .abc) ABC_END else MUSIC_END;
    for (scores, 0..) |*v, i| {
        const allowed = if (phase == .abc)
            (i < EOD) or (i == end)
        else
            (i >= CODEC_OFFSET and i < CODEC_OFFSET + CODEC_SIZE) or (i == end);
        if (!allowed) v.* = -std.math.inf(f32);
    }
    if (step < sampling.min_tokens) scores[end] = -std.math.inf(f32);
    if (sampling.repetition_penalty != 1.0 and freq.len == scores.len) {
        for (scores, freq) |*v, f| {
            if (f == 0) continue;
            const alpha = std.math.pow(f32, sampling.repetition_penalty, @floatFromInt(f));
            v.* = if (v.* < 0) v.* * alpha else v.* / alpha;
        }
    }
    if (sampling.temperature == 0) return;
    if (sampling.temperature != 1) {
        for (scores) |*v| v.* /= sampling.temperature;
    }
    const vocab = scores.len;
    const k = @min(sampling.top_k, @as(u32, @intCast(vocab)));
    if (k < vocab) {
        @memcpy(scratch[0..vocab], scores);
        const thresh = kthSmallest(scratch[0..vocab], vocab - k);
        for (scores) |*v| {
            if (v.* < thresh) v.* = -std.math.inf(f32);
        }
    }
    if (sampling.top_p < 1.0) {
        var n: usize = 0;
        for (scores, 0..) |v, i| {
            if (!std.math.isInf(v)) {
                cand[n] = @intCast(i);
                n += 1;
            }
        }
        if (n > 1) {
            var j: usize = 1;
            while (j < n) : (j += 1) {
                const key = scores[cand[j]];
                var p = j;
                while (p > 0 and scores[cand[p - 1]] < key) : (p -= 1) {
                    const tmp = cand[p];
                    cand[p] = cand[p - 1];
                    cand[p - 1] = tmp;
                }
            }
            var mx: f32 = scores[cand[0]];
            for (cand[1..n]) |ci| mx = @max(mx, scores[ci]);
            var psum: f32 = 0;
            for (cand[0..n]) |ci| psum += @exp(scores[ci] - mx);
            var cum: f32 = 0;
            const keep: usize = if (legacy_off) 3 else 1;
            for (cand[0..n], 0..) |ci, q| {
                const prob = @exp(scores[ci] - mx) / psum;
                if (q >= keep and cum > sampling.top_p) scores[ci] = -std.math.inf(f32);
                cum += prob;
            }
        }
    }
}

/// sampling.py generate_tokens (AR). `phase == .semantic` returns RAW token
/// ids — the caller subtracts CODEC_OFFSET (the pipeline's SemanticResult).
/// ABC planning NEVER branches (the reference's plan() runs cfg_scale 1.0);
/// the semantic branch passes cfg_scale = guidanceOf(...) with `negative`.
pub const ArOutcome = struct {
    /// Emitted ids, NOT including the end token. Owned by the caller.
    ids: []i32,
    truncated: bool,

    pub fn deinit(self: *ArOutcome, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
    }
};

pub fn generateTokens(
    e: *const Engine,
    allocator: std.mem.Allocator,
    prefix: []const u32,
    negative: ?[]const u32,
    sampling: Sampling,
    phase: Phase,
    cfg_scale: f32,
    legacy_off: bool,
    seed: u64,
) !ArOutcome {
    const cfg = e.cfg;
    const s = e.s;
    if (prefix.len + sampling.max_tokens > CONTEXT) return error.Yue2ContextOverflow;
    if (cfg_scale != 1.0) {
        const neg = negative orelse return error.Yue2NeedsNegative;
        if (neg.len + sampling.max_tokens > CONTEXT) return error.Yue2ContextOverflow;
    }

    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();

    const cap = prefix.len + sampling.max_tokens;
    var pos_kv = try LmKv.init(allocator, cfg, @intCast(cap), s);
    defer pos_kv.deinit(allocator);
    var pos_logits = try trunkLogits(e, a, prefix, &pos_kv, null, s);
    defer _ = mlx.mlx_array_free(pos_logits);

    var neg_kv: ?LmKv = null;
    var neg_logits: ?mlx.mlx_array = null;
    if (cfg_scale != 1.0) {
        const neg = negative.?;
        const ncap = neg.len + sampling.max_tokens;
        var kv = try LmKv.init(allocator, cfg, @intCast(ncap), s);
        const lg = trunkLogits(e, a, neg, &kv, null, s) catch |er| {
            kv.deinit(allocator);
            return er;
        };
        neg_kv = kv;
        neg_logits = lg;
    }
    defer {
        if (neg_kv) |*kv| kv.deinit(allocator);
    }
    defer {
        if (neg_logits) |lg| _ = mlx.mlx_array_free(lg);
    }

    const vocab: usize = cfg.vocab;
    var ring: ?WindowRing = null;
    defer if (ring) |*r| r.deinit(a);
    if (sampling.repetition_penalty != 1.0) ring = try WindowRing.init(a, sampling.penalty_window, @intCast(vocab));

    const logits_row = try a.alloc(f32, vocab);
    const scores = try a.alloc(f32, vocab);
    const scratch = try a.alloc(f32, vocab);
    const cand = try a.alloc(usize, vocab);

    var history = try allocator.alloc(i32, sampling.max_tokens);
    errdefer allocator.free(history);
    var hlen: usize = 0;

    var smp = Sampler{ .seed = seed };
    var step: u32 = 0;
    while (step < sampling.max_tokens) : (step += 1) {
        // Reference combine order, in the logits' bf16: uncond + cfg*(cond-uncond).
        var combined = pos_logits;
        if (cfg_scale != 1.0) {
            const diff = try subA(pos_logits, neg_logits.?, s);
            defer _ = mlx.mlx_array_free(diff);
            const scale = try scalarLike(diff, cfg_scale, s);
            defer _ = mlx.mlx_array_free(scale);
            const scaled = try mulA(diff, scale, s);
            defer _ = mlx.mlx_array_free(scaled);
            const added = try addA(neg_logits.?, scaled, s);
            defer _ = mlx.mlx_array_free(added);
            combined = added;
        }
        try readLogitsRow(combined, logits_row, s);
        @memcpy(scores, logits_row);
        distributionHost(scores, scratch, cand, sampling, step, phase, legacy_off, if (ring) |*r| r.freq else &.{});

        var token: i32 = undefined;
        if (sampling.temperature == 0) {
            token = argmaxF32(scores);
        } else {
            const sh_v = [_]c_int{@intCast(vocab)};
            const scores_arr = mlx.mlx_array_new_data(scores.ptr, &sh_v, 1, .float32);
            defer _ = mlx.mlx_array_free(scores_arr);
            const key = smp.nextKey();
            defer _ = mlx.mlx_array_free(key);
            var pick = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pick);
            try mlx.check(mlx.mlx_random_categorical(&pick, scores_arr, -1, key, s));
            const pi = try astype(pick, .int32, s);
            defer _ = mlx.mlx_array_free(pi);
            token = try readScalarI32(pi);
        }

        const end: i32 = @intCast(if (phase == .abc) ABC_END else MUSIC_END);
        if (token == end) break;
        history[hlen] = token;
        hlen += 1;
        if (ring) |*r| r.push(token);

        if (step + 1 < sampling.max_tokens) {
            var id: [1]i32 = .{ token };
            const id_shape = [_]c_int{ 1, 1 };
            const id_arr = mlx.mlx_array_new_data(id[0..].ptr, &id_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(id_arr);
            const embeds = try takeRows(e.embed, id_arr, s);
            defer _ = mlx.mlx_array_free(embeds);
            const pos_h = try trunkForward(e, embeds, &pos_kv, null, s);
            defer _ = mlx.mlx_array_free(pos_h);
            const pos_last = try finalHidden(e, pos_h, s);
            defer _ = mlx.mlx_array_free(pos_last);
            const pos_next = try lmHeadLogits(e, pos_last, s);
            _ = mlx.mlx_array_free(pos_logits);
            pos_logits = pos_next;
            if (neg_kv) |*kv| {
                const neg_embeds = try takeRows(e.embed, id_arr, s);
                defer _ = mlx.mlx_array_free(neg_embeds);
                const neg_h = try trunkForward(e, neg_embeds, kv, null, s);
                defer _ = mlx.mlx_array_free(neg_h);
                const neg_last = try finalHidden(e, neg_h, s);
                defer _ = mlx.mlx_array_free(neg_last);
                const neg_next = try lmHeadLogits(e, neg_last, s);
                _ = mlx.mlx_array_free(neg_logits.?);
                neg_logits = neg_next;
            }
        }
    }
    const truncated = step == sampling.max_tokens;
    return .{ .ids = history[0..hlen], .truncated = truncated };
}

// ════════════════════════════════════════════════════════════════════════
// NAR acoustic flow matching (nar.py CachedNAR + song_chunks)
// ════════════════════════════════════════════════════════════════════════

fn chunkRangeSize(frames: usize, prefix_tokens: usize, context: usize) ?usize {
    if (frames < 1 or prefix_tokens + 3 >= context) return null;
    const size = @min((context - prefix_tokens - 3) / 2, CONTEXT);
    return if (size < 1) null else size;
}

/// model._shift_t_value with the release's timestep_shift 1.0 → sigmoid.
fn timeShift(raw: f64) f32 {
    return @floatCast(1.0 / (1.0 + std.math.exp(-raw)));
}

/// sampling's logit, clamped to ±20 as the reference does before the shift.
fn logitClamped(t: f64) f64 {
    const l = @log(t / (1.0 - t));
    return @max(-20.0, @min(20.0, l));
}

/// TimestepEmbedder(256→2048→2048) on the sigmoid-shifted raw timestep,
/// → [2048] bf16 (the reference casts emb to the weight dtype then MLPs).
fn timeEmbed(e: *const Engine, raw: f64, s: S) !mlx.mlx_array {
    const t = timeShift(raw);
    var emb: [256]f32 = undefined;
    const half: usize = 128;
    var freqs_buf: [128]f32 = undefined;
    for (&freqs_buf, 0..) |*f, k| {
        f.* = std.math.exp(-@log(10000.0) * @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(half)));
    }
    for (0..half) |k| {
        const arg = t * freqs_buf[k];
        emb[k] = @cos(arg);
        emb[half + k] = @sin(arg);
    }
    const sh = [_]c_int{256};
    const ear = mlx.mlx_array_new_data(&emb, &sh, 1, .float32);
    defer _ = mlx.mlx_array_free(ear);
    const eb = try astype(ear, .bfloat16, s);
    defer _ = mlx.mlx_array_free(eb);
    const h1 = try e.time1.forward(eb, s);
    defer _ = mlx.mlx_array_free(h1);
    const ac = try silu(h1, s);
    defer _ = mlx.mlx_array_free(ac);
    return e.time2.forward(ac, s);
}

fn concatAxis(x: mlx.mlx_array, y: mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    _ = mlx.mlx_vector_array_append_value(vec, x);
    _ = mlx.mlx_vector_array_append_value(vec, y);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}

/// One chunk's NAR session: the causal AR prefill (whose per-layer k/v the
/// ODE re-attends), the precomputed position embedding, and the chunk shape.
/// `cache` (LmKv) must outlive `cap` — the views reference its buffers.
const NarSession = struct {
    T: usize,
    ar_length: usize,
    cache: LmKv,
    cap: CachedAr,
    pos_emb: mlx.mlx_array,

    fn deinit(self: *NarSession, allocator: std.mem.Allocator) void {
        _ = mlx.mlx_array_free(self.pos_emb);
        self.cap.deinit(allocator);
        self.cache.deinit(allocator);
    }
};

/// nar.py CachedNAR._prefill: causal AR forward whose output is discarded;
/// only the per-layer k/v matter (they keep the first `visible` = ar_length
/// positions for the NAR's cross-attention). `T` = the chunk's codec frame
/// count; `ids` = prefix + codec + [MUSIC_END] (ar_length = ids.len).
fn narPrefill(e: *const Engine, a: std.mem.Allocator, ids: []const u32, T: usize, s: S) !NarSession {
    const ar_length = ids.len;
    var cache = try LmKv.init(e.allocator, e.cfg, @intCast(ar_length), s);
    errdefer cache.deinit(e.allocator);
    var cap = try CachedAr.init(a, e.cfg.layers);
    errdefer cap.deinit(a);
    const ids32 = try a.alloc(i32, ar_length);
    for (ids, ids32) |u, *d| d.* = @intCast(u);
    const shape = [_]c_int{ 1, @intCast(ar_length) };
    const id_arr = mlx.mlx_array_new_data(ids32.ptr, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(id_arr);
    const embeds = try takeRows(e.embed, id_arr, s);
    defer _ = mlx.mlx_array_free(embeds);
    const h = try trunkForward(e, embeds, &cache, &cap, s);
    _ = mlx.mlx_array_free(h);

    const nar_len = T + 2;
    const pos_ids = try a.alloc(i32, nar_len);
    defer a.free(pos_ids);
    const max_frame: i32 = @intCast(e.cfg.max_latent_frames - 1);
    for (pos_ids, 0..) |*p, i| p.* = @min(@as(i32, @intCast(i)), max_frame);
    const pos_arr = i32Arr(pos_ids);
    defer _ = mlx.mlx_array_free(pos_arr);
    const pos_emb = try takeRows(e.pos_table, pos_arr, s);
    return .{ .T = T, .ar_length = ar_length, .cache = cache, .cap = cap, .pos_emb = pos_emb };
}

/// nar.py velocity(): one ODE midpoint evaluation. `work` is [1,T+2,64] bf16
/// (rows 1..T+1 are the current state; row 0 and T+1 are the F.pad zeros).
/// Returns the velocity in [1,T,64] bf16.
fn narVelocity(e: *const Engine, ses: *const NarSession, work: mlx.mlx_array, raw: f64, s: S) !mlx.mlx_array {
    const nar_len = ses.T + 2;
    const nh: c_int = @intCast(e.cfg.heads);
    const nkv: c_int = @intCast(e.cfg.kv_heads);
    const hd: c_int = @intCast(e.cfg.head_dim);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(e.cfg.head_dim)));

    var x = try e.vae2llm.forward(work, s);
    errdefer _ = mlx.mlx_array_free(x);
    const te = try timeEmbed(e, raw, s);
    defer _ = mlx.mlx_array_free(te);
    x = try addA(x, te, s);
    x = try addA(x, ses.pos_emb, s);

    for (e.nar_lw, 0..) |*lw, li| {
        const nx = try rmsNorm(x, lw.in_ln, e.cfg.eps, s);
        defer _ = mlx.mlx_array_free(nx);
        const q = try lw.q.forward(nx, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try lw.k.forward(nx, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try lw.v.forward(nx, s);
        defer _ = mlx.mlx_array_free(v);
        const qh = try splitHeads(q, nh, hd, s);
        defer _ = mlx.mlx_array_free(qh);
        const kh = try splitHeads(k, nkv, hd, s);
        defer _ = mlx.mlx_array_free(kh);
        const vh = try splitHeads(v, nkv, hd, s);
        defer _ = mlx.mlx_array_free(vh);
        const qn = try rmsNorm(qh, lw.q_norm, e.cfg.eps, s);
        defer _ = mlx.mlx_array_free(qn);
        const kn = try rmsNorm(kh, lw.k_norm, e.cfg.eps, s);
        defer _ = mlx.mlx_array_free(kn);
        const qr = try ropeAt(qn, hd, e.cfg.rope_theta, @intCast(ses.ar_length), s);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try ropeAt(kn, hd, e.cfg.rope_theta, @intCast(ses.ar_length), s);
        defer _ = mlx.mlx_array_free(kr);

        const kcat = try concatAxis(ses.cap.k[li], kr, 2, s);
        defer _ = mlx.mlx_array_free(kcat);
        const vcat = try concatAxis(ses.cap.v[li], vh, 2, s);
        defer _ = mlx.mlx_array_free(vcat);
        const attn = try sdpa(qr, kcat, vcat, scale, "", s);
        defer _ = mlx.mlx_array_free(attn);
        const merged = try mergeHeads(attn, s);
        defer _ = mlx.mlx_array_free(merged);
        const o = try lw.o.forward(merged, s);
        defer _ = mlx.mlx_array_free(o);
        const h1 = try addA(x, o, s);
        _ = mlx.mlx_array_free(x);
        x = h1;
        const xm = try rmsNorm(x, lw.pre_ln, e.cfg.eps, s);
        defer _ = mlx.mlx_array_free(xm);
        const gate = try lw.gate.forward(xm, s);
        defer _ = mlx.mlx_array_free(gate);
        const up = try lw.up.forward(xm, s);
        defer _ = mlx.mlx_array_free(up);
        const gact = try silu(gate, s);
        defer _ = mlx.mlx_array_free(gact);
        const gu = try mulA(gact, up, s);
        defer _ = mlx.mlx_array_free(gu);
        const down = try lw.down.forward(gu, s);
        defer _ = mlx.mlx_array_free(down);
        const h2 = try addA(x, down, s);
        _ = mlx.mlx_array_free(x);
        x = h2;
        evalA(x);
    }

    const xn = try rmsNorm(x, e.final_norm_w, e.cfg.eps, s);
    defer _ = mlx.mlx_array_free(xn);
    const out = try e.llm2vae.forward(xn, s);
    defer _ = mlx.mlx_array_free(out);
    const sh = mlx.getShape(out); // [1, nar_len, 64]
    return sliceA(out, &[_]c_int{ 0, 1, 0 }, &[_]c_int{ sh[0], @as(c_int, @intCast(nar_len - 1)), sh[2] }, s);
}

fn sliceMiddle(work: mlx.mlx_array, t_end: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(work);
    return sliceA(work, &[_]c_int{ 0, 1, 0 }, &[_]c_int{ sh[0], t_end, sh[2] }, s);
}

/// One chunk solve (nar.py CachedNAR.solve): midpoint ODE, state held in bf16
/// rows 1..T+1 of `work` exactly as the reference F.pad + dtype casts do.
fn narSolveChunk(e: *const Engine, allocator: std.mem.Allocator, ses: *NarSession, steps: u32, state0: []const f32, s: S) ![]f32 {
    const T: c_int = @intCast(ses.T);
    var work = try zerosA(&[_]c_int{ 1, T + 2, @as(c_int, @intCast(e.cfg.latent_dim)) }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(work);
    const state = try f32ToBf16(state0, &[_]c_int{ 1, T, @as(c_int, @intCast(e.cfg.latent_dim)) }, s);
    defer _ = mlx.mlx_array_free(state);
    const work0 = work;
    work = try sliceUpdateA(work, state, &[_]c_int{ 0, 1, 0 }, &[_]c_int{ 1, T + 1, @as(c_int, @intCast(e.cfg.latent_dim)) }, s);
    _ = mlx.mlx_array_free(work0);

    const dt = 1.0 / @as(f64, @floatFromInt(steps));
    var step: usize = 0;
    while (step < steps) : (step += 1) {
        const t = 1.0 - @as(f64, @floatFromInt(step)) * dt;
        const first = try narVelocity(e, ses, work, logitClamped(t), s);
        defer _ = mlx.mlx_array_free(first);
        const mid_over_two = try scalarLike(first, @floatCast(dt / 2.0), s);
        defer _ = mlx.mlx_array_free(mid_over_two);
        const decay_hi = try mulA(first, mid_over_two, s);
        defer _ = mlx.mlx_array_free(decay_hi);
        const cur = try sliceMiddle(work, T + 1, s);
        defer _ = mlx.mlx_array_free(cur);
        const mid = try subA(cur, decay_hi, s);
        defer _ = mlx.mlx_array_free(mid);
        const work1 = try sliceUpdateA(work, mid, &[_]c_int{ 0, 1, 0 }, &[_]c_int{ 1, T + 1, @as(c_int, @intCast(e.cfg.latent_dim)) }, s);
        _ = mlx.mlx_array_free(work);
        work = work1;

        const second = try narVelocity(e, ses, work, logitClamped(t - dt / 2.0), s);
        defer _ = mlx.mlx_array_free(second);
        const dt_scaled = try scalarLike(second, @floatCast(dt), s);
        defer _ = mlx.mlx_array_free(dt_scaled);
        const scaled2 = try mulA(second, dt_scaled, s);
        defer _ = mlx.mlx_array_free(scaled2);
        const cur2 = try sliceMiddle(work, T + 1, s);
        defer _ = mlx.mlx_array_free(cur2);
        const next = try subA(cur2, scaled2, s);
        defer _ = mlx.mlx_array_free(next);
        const work2 = try sliceUpdateA(work, next, &[_]c_int{ 0, 1, 0 }, &[_]c_int{ 1, T + 1, @as(c_int, @intCast(e.cfg.latent_dim)) }, s);
        _ = mlx.mlx_array_free(work);
        work = work2;
    }

    const fin = try sliceMiddle(work, T + 1, s);
    defer _ = mlx.mlx_array_free(fin);
    var dor = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dor);
    const cpu = mlx.mlx_default_cpu_stream_new();
    const f = try astype(fin, .float32, cpu);
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_contiguous(&dor, f, false, cpu));
    evalA(dor);
    const d = mlx.mlx_array_data_float32(dor) orelse return error.NoData;
    const out = try allocator.alloc(f32, @intCast(ses.T * e.cfg.latent_dim));
    @memcpy(out, d[0..out.len]);
    return out;
}

fn f32ToBf16(data: []const f32, shape: []const c_int, s: S) !mlx.mlx_array {
    const a = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(a);
    return astype(a, .bfloat16, s);
}

/// nar.py synthesize: draw the song noise ONCE (or take the fixture's), chunk
/// the codec, and solve each chunk serially against its own AR prefill.
/// `codec` holds OFFSET-SUBTRACTED semantic ids in [0, CODEC_SIZE), exactly
/// song_chunks' contract; chunk ids are prefix + (id + CODEC_OFFSET) +
/// [MUSIC_END]. The caller (gen.zig pipeline) strips CODEC_OFFSET like
/// pipeline.py SemanticResult.tokens.
pub const NarOutcome = struct {
    /// [T_total, 64] fp32 latents, one row per codec frame. Owned by caller.
    latents: []f32,
    pub fn deinit(self: *NarOutcome, allocator: std.mem.Allocator) void {
        allocator.free(self.latents);
    }
};

pub fn narSynthesize(
    e: *const Engine,
    allocator: std.mem.Allocator,
    prefix: []const u32,
    codec: []const u32,
    seed: u64,
    steps: u32,
    noise_override: ?[]const f32,
    s: S,
) !NarOutcome {
    if (prefix.len == 0) return error.Yue2EmptyPrefix;
    if (codec.len == 0) return error.Yue2EmptyCodec;
    if (steps == 0) return error.InvalidOdeSteps;
    for (codec) |id| if (id >= CODEC_SIZE) return error.Yue2CodecOutOfRange;
    if (e.cfg.latent_dim != 64) return error.UnsupportedLatentType;

    var arena_inst = std.heap.ArenaAllocator.init(allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();

    const noise: []f32 = if (noise_override) |n|
        @constCast(n)
    else blk: {
        const ns = try a.alloc(f32, codec.len * 64);
        const sh = [_]c_int{ @intCast(codec.len), 64 };
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        _ = mlx.mlx_random_key(&key, seed);
        var arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_random_normal(&arr, &sh, 2, .float32, 0.0, 1.0, key, s));
        var cpu_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cpu_arr);
        const cpu = mlx.mlx_default_cpu_stream_new();
        const fc = try astype(arr, .float32, cpu);
        defer _ = mlx.mlx_array_free(fc);
        try mlx.check(mlx.mlx_contiguous(&cpu_arr, fc, false, cpu));
        evalA(cpu_arr);
        const d = mlx.mlx_array_data_float32(cpu_arr) orelse return error.NoData;
        @memcpy(ns, d[0..ns.len]);
        break :blk ns;
    };

    const size = chunkRangeSize(codec.len, prefix.len, CONTEXT) orelse return error.Yue2ChunkTooSmall;
    var latents = try allocator.alloc(f32, codec.len * 64);
    errdefer allocator.free(latents);
    var chunk_start: usize = 0;
    while (chunk_start < codec.len) {
        const chunk_end = @min(chunk_start + size, codec.len);
        const T = chunk_end - chunk_start;

        const ids = try a.alloc(u32, prefix.len + T + 1);
        @memcpy(ids[0..prefix.len], prefix);
        for (ids[prefix.len .. prefix.len + T], codec[chunk_start..chunk_end]) |*d, c| {
            d.* = c + CODEC_OFFSET;
        }
        ids[prefix.len + T] = MUSIC_END;

        var ses = try narPrefill(e, a, ids, T, s);
        defer ses.deinit(a);

        const solved = try narSolveChunk(e, allocator, &ses, steps, noise[chunk_start * 64 .. chunk_end * 64], s);
        defer allocator.free(solved);
        @memcpy(latents[chunk_start * 64 .. chunk_end * 64], solved);
        chunk_start = chunk_end;
    }
    return .{ .latents = latents };
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(64 * 1024 * 1024));
}

/// Load ONE safetensors file into a Weights map (CPU stream; iterator +1
/// transferred into the map — the model.zig pattern).
fn loadFileWeights(allocator: std.mem.Allocator, model_dir: []const u8, file: []const u8) !Weights {
    var w = Weights.init(allocator);
    errdefer w.deinit();
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
    defer allocator.free(path);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, cpu_s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (rc != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const owned_key = try allocator.dupe(u8, std.mem.span(key.?));
        errdefer allocator.free(owned_key);
        try w.map.put(owned_key, value);
    }
    log.info("[yue2] loaded {d} tensors from {s}\n", .{ w.count(), file });
    return w;
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic protocol/config layer (no weights, no GPU).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "yue2 instructions are byte-exact (protocol.py INSTRUCTIONS)" {
    try testing.expectEqualStrings("Generate music with codec tokens from the given conditions.", instructionFor(.off));
    try testing.expectEqualStrings("Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.", instructionFor(.melody));
    try testing.expectEqualStrings("Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.", instructionFor(.full));
}

test "yue2 promptText: instruction + [Tags] + [Lyrics], trailing newline" {
    const a = testing.allocator;
    const req = MusicRequest{ .style = "English rock", .lyrics = "[Verse]\nhello" };
    const text = try promptText(a, &req);
    defer a.free(text);
    try testing.expectEqualStrings("Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.\n[Tags]\nEnglish rock\n[Lyrics]\n[Verse]\nhello\n", text);
}

test "yue2 tokenPrefixes: planning, scored, off and negative branches" {
    const a = testing.allocator;
    const base = [_]u32{ EOD, 5, 6 };
    // Planning prefix (abc_ids null): stops after ABC_START.
    const plan = try tokenPrefixes(a, &base, .full, null);
    defer a.free(plan);
    try testing.expectEqualSlices(u32, &.{ EOD, 5, 6, ABC_START }, plan);
    // Scored prefix: full [EOD, text, ABC_START, abc, ABC_END, MUSIC_START].
    const abc = [_]u32{ 7, 8 };
    const scored = try tokenPrefixes(a, &base, .melody, &abc);
    defer a.free(scored);
    try testing.expectEqualSlices(u32, &.{ EOD, 5, 6, ABC_START, 7, 8, ABC_END, MUSIC_START }, scored);
    // cot=off skips the ABC block entirely.
    const off = try tokenPrefixes(a, &base, .off, null);
    defer a.free(off);
    try testing.expectEqualSlices(u32, &.{ EOD, 5, 6, ABC_START, ABC_END, MUSIC_START }, off);
    // Negative branch: instruction-only for off, exact ABC for symbolic.
    const instr = [_]u32{ EOD, 5 };
    const neg_off = try negativePrefix(a, &instr, .off, &.{});
    defer a.free(neg_off);
    try testing.expectEqualSlices(u32, &.{ EOD, 5, MUSIC_START }, neg_off);
    const neg_sym = try negativePrefix(a, &instr, .full, &abc);
    defer a.free(neg_sym);
    try testing.expectEqualSlices(u32, &.{ EOD, 5, ABC_START, 7, 8, ABC_END, MUSIC_START }, neg_sym);
}

test "yue2 guidance defaults: 1.0 symbolic, 1.01 off, explicit override" {
    var req = MusicRequest{ .style = "s", .lyrics = "" };
    try testing.expectEqual(@as(f32, 1.0), guidanceOf(&req));
    req.cot = .off;
    try testing.expectEqual(@as(f32, 1.01), guidanceOf(&req));
    req.cfg_scale = 2.5;
    try testing.expectEqual(@as(f32, 2.5), guidanceOf(&req));
}

test "yue2 request validation: abc rules and cfg bounds" {
    const req = MusicRequest{ .style = "s", .lyrics = "", .abc = "X:1", .cot = .off };
    try testing.expectError(error.AbcWithCotOff, validateRequest(&req));
    var bad = MusicRequest{ .style = "s", .lyrics = "", .abc = "  ", .cot = .full };
    try testing.expectError(error.EmptyAbc, validateRequest(&bad));
    bad = .{ .style = "s", .lyrics = "", .cfg_scale = 21.0 };
    try testing.expectError(error.InvalidCfgScale, validateRequest(&bad));
    bad = .{ .style = "s", .lyrics = "", .seed = 1 << 63 };
    try testing.expectError(error.InvalidSeed, validateRequest(&bad));
    const ok = MusicRequest{ .style = "s", .lyrics = "", .abc = "X:1", .cot = .full, .cfg_scale = 20.0 };
    try validateRequest(&ok);
}

test "yue2 chunk math: (context - prefix - 3) / 2, python floor semantics" {
    // prefix 200: (24576-203)/2 = 12186.
    try testing.expectEqual(@as(usize, 12186), chunkSizeFor(200, CONTEXT));
    // Exact fit: 2*size+3 == context.
    try testing.expectEqual(@as(usize, 1000), chunkSizeFor(24576 - 2003, CONTEXT));
    // One token over: floor drops a whole frame pair.
    try testing.expectEqual(@as(usize, 999), chunkSizeFor(24576 - 2002, CONTEXT));
    // No room at all -> 0 (caller refuses by name).
    try testing.expectEqual(@as(usize, 0), chunkSizeFor(24573, CONTEXT));
    try testing.expectEqual(@as(usize, 0), chunkSizeFor(30000, CONTEXT));
}

test "yue2 sampling defaults match the released yue2_generation_config.json" {
    const abc = ABC_SAMPLING;
    try testing.expectEqual(@as(f32, 0.7), abc.temperature);
    try testing.expectEqual(@as(f32, 0.9), abc.top_p);
    try testing.expectEqual(@as(u32, 30), abc.top_k);
    try testing.expectEqual(@as(f32, 1.005), abc.repetition_penalty);
    try testing.expectEqual(@as(u32, 100), abc.penalty_window);
    try testing.expectEqual(@as(u32, 32), abc.min_tokens);
    try testing.expectEqual(@as(u32, 4096), abc.max_tokens);
    const sem = SEMANTIC_SAMPLING;
    try testing.expectEqual(@as(f32, 1.0), sem.temperature);
    try testing.expectEqual(@as(f32, 0.95), sem.top_p);
    try testing.expectEqual(@as(u32, 100), sem.top_k);
    try testing.expectEqual(@as(f32, 1.2), sem.repetition_penalty);
    try testing.expectEqual(@as(u32, 50), sem.penalty_window);
    try testing.expectEqual(@as(u32, 200), sem.min_tokens);
    try testing.expectEqual(@as(u32, 9000), sem.max_tokens);
}

test "yue2 parseGenConfig: field-by-field overrides, null keeps default" {
    const a = testing.allocator;
    const content =
        \\{"abc":{"temperature":0.5},"semantic":null,"ode_steps":16}
    ;
    const gc = try parseGenConfig(content, GenConfig{});
    _ = a;
    try testing.expectEqual(@as(f32, 0.5), gc.abc.temperature);
    // Unspecified fields keep defaults.
    try testing.expectEqual(@as(u32, 30), gc.abc.top_k);
    try testing.expectEqual(SEMANTIC_SAMPLING, gc.semantic);
    try testing.expectEqual(@as(u32, 16), gc.ode_steps);
    // An invalid sampling field is a named error.
    try testing.expectError(error.InvalidGenConfig, parseGenConfig("{\"abc\":{\"top_p\":1.5}}", GenConfig{}));
    try testing.expectError(error.InvalidGenConfig, parseGenConfig("not json", GenConfig{}));
}

test "yue2 parseCfg: the released YuE2-3B config parses field-exact" {    const cfg = try parseCfg(REAL_CONFIG_JSON);
    try testing.expectEqual(@as(u32, 28), cfg.layers);
    try testing.expectEqual(@as(u32, 2048), cfg.hidden);
    try testing.expectEqual(@as(u32, 16), cfg.heads);
    try testing.expectEqual(@as(u32, 8), cfg.kv_heads);
    try testing.expectEqual(@as(u32, 128), cfg.head_dim);
    try testing.expectEqual(@as(u32, 6144), cfg.inter);
    try testing.expectEqual(@as(u32, 184704), cfg.vocab);
    try testing.expectEqual(@as(u32, 24576), cfg.max_pos);
    try testing.expectEqual(@as(u32, 64), cfg.latent_dim);
    try testing.expectEqual(@as(f32, 1.0), cfg.timestep_shift);
    try testing.expectApproxEqAbs(@as(f32, 1e-6), cfg.eps, 1e-12);
    // latent_type is the only supported family.
    try testing.expectError(error.UnsupportedLatentType, parseCfg("{\"model_type\":\"yue2\",\"latent_type\":\"rvq\",\"hidden_size\":2048}"));
    try testing.expectError(error.InvalidConfig, parseCfg("{\"latent_type\":\"vae\"}"));
    try testing.expectError(error.InvalidConfig, parseCfg("not json"));
}

test "yue2 vae config parses the embedded decoder section" {
    const content =
        \\{"latent_type":"vae","latent_dim":64,"vae":{"channels":64,"latent_dim":64,
        \\ "out_channels":2,"sample_rate":48000,"use_snake":true,"final_tanh":false,
        \\ "c_mults":[1,2,4,8,16,32],"strides":[2,2,4,4,5,6]}}
    ;
    const vc = try parseVaeCfg(content, Cfg{});
    try testing.expectEqual(@as(u32, 64), vc.channels);
    try testing.expectEqual(@as(u32, 2), vc.out_channels);
    try testing.expectEqual(@as(u32, 48000), vc.sample_rate);
    try testing.expectEqual(@as(u32, 6), vc.depth);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 4, 8, 16, 32 }, vc.c_mults[0..6]);
    try testing.expectEqualSlices(u32, &.{ 2, 2, 4, 4, 5, 6 }, vc.strides[0..6]);
    // A missing section, a stride that does not divide the sample rate, and a
    // latent width that disagrees with the trunk are all named errors.
    try testing.expectError(error.MissingVaeConfig, parseVaeCfg("{\"latent_dim\":64}", Cfg{}));
    try testing.expectError(error.InvalidConfig, parseVaeCfg(
        "{\"vae\":{\"channels\":64,\"latent_dim\":64,\"out_channels\":2,\"sample_rate\":48000,\"c_mults\":[1,2],\"strides\":[3,3]}}",
        Cfg{},
    ));
    try testing.expectError(error.InvalidConfig, parseVaeCfg(
        "{\"vae\":{\"channels\":64,\"latent_dim\":32,\"out_channels\":2,\"sample_rate\":48000,\"c_mults\":[1,2],\"strides\":[2,2]}}",
        Cfg{},
    ));
}

test "yue2 vae geometry: output length, ratio, required halo" {
    const vc = VaeCfg{};
    try testing.expectEqual(@as(u32, 1920), vaeRatio(vc));
    // 1920·T − 64 (odd stride 5/6 shed exactly 64 at the deepest level).
    try testing.expectEqual(@as(u32, 1856), vaeOutputLength(vc, 1));
    try testing.expectEqual(@as(u32, 3776), vaeOutputLength(vc, 2));
    try testing.expectEqual(@as(u32, 15296), vaeOutputLength(vc, 8));
    try testing.expectEqual(@as(u32, 2457536), vaeOutputLength(vc, 1280));
    // The shipped decoder needs 12 frames of halo for a 1024-frame core, so
    // config decode_halo_frames=16 is valid. A wrong block traversal order
    // reads 37 here (the residual dilations must be divided by the shallow
    // strides, not the deep ones).
    try testing.expectEqual(@as(u32, 12), vaeRequiredHalo(vc));
    try testing.expect(vc.halo_frames >= vaeRequiredHalo(vc));
}

test "yue2 vae tile crops: last tile absorbs the -64, earlier tiles fit" {
    const vc = VaeCfg{};
    const ratio = vaeRatio(vc);
    const frames: usize = 2000; // two tiles (1024 + 976)
    const total = vaeOutputLength(vc, @intCast(frames));
    var start: usize = 0;
    while (start < frames) : (start += vc.core_frames) {
        const end = @min(frames, start + vc.core_frames);
        const left = if (start > vc.halo_frames) start - vc.halo_frames else 0;
        const right = @min(frames, end + vc.halo_frames);
        const tile_len = vaeOutputLength(vc, @intCast(right - left));
        const cp = vaeTileCrop(start, end, left, ratio, total);
        // The crop must be covered by the tile, and land in the output.
        try testing.expect(cp.crop_start >= 0);
        try testing.expect(cp.crop_start + cp.out_len <= tile_len);
        try testing.expect(cp.out_start + cp.out_len <= total);
    }
    // The final tile consumes its whole natural length (crop + out_len == tile).
    const last_start = start - vc.core_frames;
    const last_cp = vaeTileCrop(last_start, frames, last_start - vc.halo_frames, ratio, total);
    try testing.expectEqual(total, @as(u32, @intCast(last_cp.out_start + last_cp.out_len)));
}

test "yue2 missing weight is a named error" {
    var w = Weights.init(testing.allocator);
    defer w.deinit();
    try testing.expectError(error.MissingWeight, getW(&w, "model.embed_tokens.weight"));
    // Reference these so the loader/resolver bodies are semantically checked
    // even when no checkpoint is present (Zig analyzes only what is reachable).
    _ = &Engine.load;
    _ = &Engine.deinit;
    _ = &loadFileWeights;
    _ = &resolveVaeDecoder;
    _ = &buildArLayerW;
    _ = &buildNarLayerW;
    _ = &resolveQLin;
    _ = &fuseVaeConv;
    _ = &generateTokens;
    _ = &narSynthesize;
    _ = &timeEmbed;
    _ = &narPrefill;
    _ = &narVelocity;
    _ = &narSolveChunk;
    _ = &vaeDecodeTiled;
    _ = &vaeForwardOne;
    _ = &vaeRequiredHalo;
    _ = &vaeOutputLength;
}

test "yue2 chunkRangeSize mirrors protocol.chunk_ranges" {
    try testing.expectEqual(@as(?usize, 12236), chunkRangeSize(5000, 100, 24576));
    // The whole song fits in one chunk when prefix leaves room.
    try testing.expectEqual(@as(?usize, 43), chunkRangeSize(30, 10, 100));
    // A prefix that swallows the context and an empty codec are both invalid.
    try testing.expectEqual(@as(?usize, null), chunkRangeSize(50, 97, 100));
    try testing.expectEqual(@as(?usize, null), chunkRangeSize(0, 10, 24576));
}

test "yue2 timeShift is the sigmoid of the clamped logit (shift=1.0)" {
    try testing.expectApproxEqAbs(@as(f32, 0.5), timeShift(0.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), timeShift(20.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), timeShift(-20.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), timeShift(logitClamped(0.5)), 1e-6);
}

test "yue2 logitClamped saturates outside ±20 like the reference clamp" {
    try testing.expectEqual(@as(f64, 20.0), logitClamped(1.0));
    try testing.expectEqual(@as(f64, -20.0), logitClamped(0.0));
    try testing.expectApproxEqAbs(@as(f64, 0.0), logitClamped(0.5), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, -@log(9.0)), logitClamped(0.1), 1e-9);
}

// Gated on YUE2_TEST_MODEL (a converted pack dir): the loader resolves every
// handle and the shapes match the checkpoint contract.
test "yue2 load: shapes match the checkpoint contract" {
    const model_dir = std.mem.span(std.c.getenv("YUE2_TEST_MODEL") orelse return error.SkipZigTest);
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var e = try Engine.load(io, testing.allocator, model_dir);
    defer e.deinit();

    try testing.expectEqualSlices(c_int, &.{ 184704, 2048 }, mlx.getShape(e.embed)[0..2]);
    try testing.expectEqualSlices(c_int, &.{ 184704, 2048 }, mlx.getShape(e.lm_head.w)[0..2]);
    try testing.expectEqualSlices(c_int, &.{ 24576, 2048 }, mlx.getShape(e.pos_table)[0..2]);
    try testing.expectEqualSlices(c_int, &.{ 64, 2048 }, mlx.getShape(e.llm2vae.w)[0..2]);
    try testing.expectEqualSlices(c_int, &.{ 2048, 64 }, mlx.getShape(e.vae2llm.w)[0..2]);
    try testing.expectEqualSlices(c_int, &.{ 2048, 256 }, mlx.getShape(e.time1.w)[0..2]);
    // A few VAE shapes: conv_in fused to MLX [out,K,in], first block convT.
    try testing.expectEqualSlices(c_int, &.{ 2048, 7, 64 }, mlx.getShape(e.vae.conv_in.w)[0..3]);
    try testing.expectEqualSlices(c_int, &.{ 1024, 12, 2048 }, mlx.getShape(e.vae.blocks[0].convt.w)[0..3]);
    try testing.expect(e.vae.conv_in.bias != null);
    // The final conv ships without bias.
    try testing.expectEqual(@as(?mlx.mlx_array, null), e.vae.final_conv.bias);
    // Every fused conv dropped the raw v/g tensors.
    try testing.expect(e.vae_w.get("decoder.layers.0.weight_v") == null);
    try testing.expect(e.vae_w.get("decoder.layers.0.weight") != null);
}

test "yue2 tiktoken layout: protocol token ids agree with the tokenizer's specials" {
    // EOD is the first special; <abc>/</abc> are pinned at special index
    // 204/205 (tokenization_yue2.py). If either side moves, generation
    // prompts stop matching the checkpoint vocabulary.
    try testing.expectEqual(@as(u32, @intCast(tok_mod.TIKTOKEN_QWEN_RANKS)), EOD);
    try testing.expectEqual(@as(u32, @intCast(tok_mod.TIKTOKEN_QWEN_RANKS + 204)), ABC_START);
    try testing.expectEqual(@as(u32, @intCast(tok_mod.TIKTOKEN_QWEN_RANKS + 205)), ABC_END);
}

test "yue2 argmaxF32 returns the FIRST index of the maximum (torch semantics)" {
    try testing.expectEqual(@as(i32, 1), argmaxF32(&.{ -2, 3, 3, 1 }));
    try testing.expectEqual(@as(i32, 3), argmaxF32(&.{ 0, 0, 0, 7 }));
    try testing.expectEqual(@as(i32, 0), argmaxF32(&.{ -std.math.inf(f32), -std.math.inf(f32) }));
}

test "yue2 kthSmallest is the value at the 0-based rank" {
    var buf = [_]f32{ 7, 1, 9, 4, 3 };
    try testing.expectEqual(@as(f32, 4), kthSmallest(&buf, 2));
    var buf2 = [_]f32{ 7, 1, 9, 4, 3 };
    try testing.expectEqual(@as(f32, 9), kthSmallest(&buf2, 4));
    var buf3 = [_]f32{ 5 };
    try testing.expectEqual(@as(f32, 5), kthSmallest(&buf3, 0));
}

test "yue2 WindowRing counts exactly the last window ids" {
    var ring = try WindowRing.init(testing.allocator, 3, 8);
    defer ring.deinit(testing.allocator);
    ring.push(1);
    ring.push(2);
    ring.push(1);
    try testing.expectEqual(@as(u32, 2), ring.freq[1]);
    try testing.expectEqual(@as(u32, 1), ring.freq[2]);
    ring.push(3);
    // The oldest id (1) left the window: [2,1,3].
    try testing.expectEqual(@as(u32, 1), ring.freq[1]);
    try testing.expectEqual(@as(u32, 1), ring.freq[2]);
    try testing.expectEqual(@as(u32, 1), ring.freq[3]);
    // A window of zero ids is a contract violation (the ring would divide by zero).
    try testing.expectError(error.InvalidWindow, WindowRing.init(testing.allocator, 0, 8));
}

test "yue2 distribution: semantic greedy picks the first allowed codec token, masks the end before min_tokens" {
    const v: usize = 184704;
    const scores = try testing.allocator.alloc(f32, v);
    defer testing.allocator.free(scores);
    const scratch = try testing.allocator.alloc(f32, v);
    defer testing.allocator.free(scratch);
    const cand = try testing.allocator.alloc(usize, v);
    defer testing.allocator.free(cand);
    for (scores) |*s| s.* = -1.0;
    distributionHost(scores, scratch, cand, SEMANTIC_SAMPLING, 0, .semantic, false, &.{});
    // Allowed = codec range + MUSIC_END; step 0 < min_tokens so MUSIC_END is
    // masked; every allowed entry ties at -1.0 → greedy argmax = CODEC_OFFSET.
    try testing.expectEqual(@as(i32, @intCast(CODEC_OFFSET)), argmaxF32(scores));
    try testing.expectEqual(@as(f32, -std.math.inf(f32)), scores[30]);
    try testing.expectEqual(@as(f32, -std.math.inf(f32)), scores[MUSIC_END]);
    // Past min_tokens (200 default) with a higher end token, greedy picks the end.
    for (scores) |*s| s.* = -1.0;
    scores[EOD + 7] = 2.0;
    scores[MUSIC_END] = 5.0;
    distributionHost(scores, scratch, cand, SEMANTIC_SAMPLING, 200, .semantic, false, &.{});
    try testing.expectEqual(@as(i32, @intCast(MUSIC_END)), argmaxF32(scores));
}

test "yue2 distribution: abc phase allowed set is [0,EOD) plus ABC_END" {
    const v: usize = ABC_END + 1;
    const scores = try testing.allocator.alloc(f32, v);
    defer testing.allocator.free(scores);
    const scratch = try testing.allocator.alloc(f32, v);
    defer testing.allocator.free(scratch);
    const cand = try testing.allocator.alloc(usize, v);
    defer testing.allocator.free(cand);
    for (scores) |*s| s.* = -1.0;
    scores[7] = 2.0;
    scores[@intCast(ABC_END)] = 9.0;
    distributionHost(scores, scratch, cand, ABC_SAMPLING, 32, .abc, true, &.{});
    // min_tokens 32 satisfied; ABC_END wins.
    try testing.expectEqual(@as(i32, @intCast(ABC_END)), argmaxF32(scores));
    // Ids >= EOD outside the end token are refused even when hot.
    for (scores) |*s| s.* = -1.0;
    scores[7] = 2.0;
    scores[@intCast(ABC_END)] = 9.0;
    scores[EOD + 3] = 50.0;
    distributionHost(scores, scratch, cand, ABC_SAMPLING, 32, .abc, true, &.{});
    try testing.expectEqual(@as(f32, -std.math.inf(f32)), scores[EOD + 3]);
    try testing.expectEqual(@as(i32, @intCast(ABC_END)), argmaxF32(scores));
}


test "yue2 distribution: window penalty divides positives, multiplies negatives" {
    const v: usize = 184704;
    const scores = try testing.allocator.alloc(f32, v);
    defer testing.allocator.free(scores);
    const scratch = try testing.allocator.alloc(f32, v);
    defer testing.allocator.free(scratch);
    const cand = try testing.allocator.alloc(usize, v);
    defer testing.allocator.free(cand);
    const freq = try testing.allocator.alloc(u32, v);
    defer testing.allocator.free(freq);
    @memset(freq, 0);
    freq[CODEC_OFFSET] = 1;
    freq[CODEC_OFFSET + 1] = 2;
    for (scores) |*s| s.* = 1.0;
    scores[CODEC_OFFSET] = 0.5;
    scores[CODEC_OFFSET + 1] = -0.5;
    const sampling = Sampling{ .temperature = 0, .top_p = 1.0, .top_k = 0, .repetition_penalty = 2.0, .penalty_window = 50, .min_tokens = 0, .max_tokens = 9000 };
    distributionHost(scores, scratch, cand, sampling, 0, .semantic, false, freq);
    try testing.expectEqual(@as(f32, 0.25), scores[CODEC_OFFSET]);
    try testing.expectEqual(@as(f32, -2.0), scores[CODEC_OFFSET + 1]);
}

test "yue2 distribution: top-k keeps values >= threshold (ties retained), top-p cuts past the head" {
    const v: usize = 184704;
    const scores = try testing.allocator.alloc(f32, v);
    defer testing.allocator.free(scores);
    const scratch = try testing.allocator.alloc(f32, v);
    defer testing.allocator.free(scratch);
    const cand = try testing.allocator.alloc(usize, v);
    defer testing.allocator.free(cand);
    for (scores) |*s| s.* = -5.0;
    const A = CODEC_OFFSET;
    const B = CODEC_OFFSET + 47;
    const C = CODEC_OFFSET + 147;
    const D = CODEC_OFFSET + 483;
    scores[A] = 5.0;
    scores[B] = 4.0;
    scores[C] = 3.0;
    scores[D] = 2.0;
    // top_k 3 → threshold 3; D falls below. symbolic keeps only the head past top_p 0.5.
    const sampling = Sampling{ .temperature = 1.0, .top_p = 0.5, .top_k = 3, .repetition_penalty = 1.0, .penalty_window = 50, .min_tokens = 0, .max_tokens = 9000 };
    distributionHost(scores, scratch, cand, sampling, 0, .semantic, false, &.{});
    try testing.expectEqual(@as(f32, 5.0), scores[A]);
    try testing.expectEqual(@as(f32, -std.math.inf(f32)), scores[B]);
    try testing.expectEqual(@as(f32, -std.math.inf(f32)), scores[C]);
    try testing.expectEqual(@as(f32, -std.math.inf(f32)), scores[D]);
    // legacy_off keeps the first THREE past the head (the off-mode ladder).
    for (scores) |*s| s.* = -5.0;
    scores[A] = 5.0;
    scores[B] = 4.0;
    scores[C] = 3.0;
    scores[D] = 2.0;
    distributionHost(scores, scratch, cand, sampling, 0, .semantic, true, &.{});
    try testing.expectEqual(@as(f32, 5.0), scores[A]);
    try testing.expectEqual(@as(f32, 4.0), scores[B]);
    try testing.expectEqual(@as(f32, 3.0), scores[C]);
    try testing.expectEqual(@as(f32, -std.math.inf(f32)), scores[D]);
    // Equal-to-threshold C stays (>= semantics), so with top_k 2 (threshold 4)
    // the tie at 4 is retained by both.
    for (scores) |*s| s.* = -5.0;
    scores[A] = 5.0;
    scores[B] = 4.0;
    scores[C] = 4.0;
    scores[D] = 2.0;
    const topk2 = Sampling{ .temperature = 1.0, .top_p = 1.0, .top_k = 2, .repetition_penalty = 1.0, .penalty_window = 50, .min_tokens = 0, .max_tokens = 9000 };
    distributionHost(scores, scratch, cand, topk2, 0, .semantic, false, &.{});
    try testing.expectEqual(@as(f32, 5.0), scores[A]);
    try testing.expectEqual(@as(f32, 4.0), scores[B]);
    try testing.expectEqual(@as(f32, 4.0), scores[C]);
    try testing.expectEqual(@as(f32, -std.math.inf(f32)), scores[D]);
}

// ════════════════════════════════════════════════════════════════════════
// Env-gated fixtures: YUE2_TEST_MODEL (a scripts/convert_yue2_weights.py
// pack) + YUE2_FIXTURES (tests/dump_yue2_fixtures.py output). Fed on the
// 128 GB Mac; skip (with a counted skip) when either env is absent.
// ════════════════════════════════════════════════════════════════════════

fn fixturesDir() ![]const u8 {
    return std.mem.span(std.c.getenv("YUE2_FIXTURES") orelse return error.SkipZigTest);
}

fn testEngine(io: std.Io, a: std.mem.Allocator) !*Engine {
    const model_dir = std.mem.span(std.c.getenv("YUE2_TEST_MODEL") orelse return error.SkipZigTest);
    return Engine.load(io, a, model_dir);
}

fn readRawF32(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]f32 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const bytes = try readFile(io, a, path);
    defer a.free(bytes);
    const n = bytes.len / 4;
    try testing.expect(bytes.len % 4 == 0);
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

fn readRawU32(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u32 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const bytes = try readFile(io, a, path);
    defer a.free(bytes);
    const n = bytes.len / 4;
    try testing.expect(bytes.len % 4 == 0);
    const out = try a.alloc(u32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

fn readMetaU32(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) !u32 {
    const path = try std.fmt.allocPrint(a, "{s}/yue2_meta.json", .{ dir });
    defer a.free(path);
    const bytes = try readFile(io, a, path);
    defer a.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    const raw = parsed.value.object.get(name) orelse return error.MissingFixtureField;
    return std.math.cast(u32, raw.integer) orelse return error.MissingFixtureField;
}

fn cosineF64f(x: []const f32, y: []const f32) f64 {
    var dot: f64 = 0;
    var nx: f64 = 0;
    var ny: f64 = 0;
    for (x, y) |a, b| {
        dot += @as(f64, a) * b;
        nx += @as(f64, a) * a;
        ny += @as(f64, b) * b;
    }
    return dot / (@sqrt(nx) * @sqrt(ny) + 1e-30);
}

fn rmsRatioF64(x: []const f32, y: []const f32) f64 {
    var sx: f64 = 0;
    var sy: f64 = 0;
    for (x) |a| sx += @as(f64, a) * a;
    for (y) |b| sy += @as(f64, b) * b;
    return @sqrt(sx / @as(f64, @floatFromInt(x.len))) / (@sqrt(sy / @as(f64, @floatFromInt(y.len))) + 1e-30);
}

/// cos AND rms_ratio over host f32 pairs — a cosine alone cannot see a scale
/// error, so a wrong factor anywhere in the chain fails its rms bar.
fn assertHostParity(got: []const f32, ref: []const f32, label: []const u8, min_cos: f64, rms_tol: f64) !void {
    try testing.expectEqual(ref.len, got.len);
    const cos = cosineF64f(got, ref);
    const rr = rmsRatioF64(got, ref);
    std.debug.print("[yue2-{s}] cos={d:.6} rms_ratio={d:.4} n={d}\n", .{ label, cos, rr, got.len });
    try testing.expect(cos > min_cos);
    try testing.expect(rr > 1.0 - rms_tol and rr < 1.0 + rms_tol);
}

fn assertArrayParity(arr: mlx.mlx_array, ref: []const f32, label: []const u8, min_cos: f64, rms_tol: f64, s: S) !void {
    const f = try astype(arr, .float32, s);
    defer _ = mlx.mlx_array_free(f);
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    evalA(c);
    const n: usize = @intCast(mlx.mlx_array_size(c));
    try testing.expectEqual(ref.len, n);
    const d = mlx.mlx_array_data_float32(c) orelse return error.NoData;
    try assertHostParity(d[0..n], ref, label, min_cos, rms_tol);
}

test "yue2 fixture: AR prefill + decode16 logits match the reference (YUE2_TEST_MODEL + YUE2_FIXTURES)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena_inst = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    var e = try testEngine(io, a);
    defer e.deinit();
    const fix = try fixturesDir();
    for ([_][]const u8{ "prefill", "decode16" }) |name| {
        const ids = try readRawU32(io, a, fix, try std.fmt.allocPrint(a, "yue2_ar_{s}.i32.raw", .{name}));
        const ref_logits = try readRawF32(io, a, fix, try std.fmt.allocPrint(a, "yue2_ar_logits_{s}.f32", .{name}));
        try testing.expectEqual(e.cfg.vocab, ref_logits.len);
        var kv = try LmKv.init(a, e.cfg, @intCast(ids.len), e.s);
        defer kv.deinit(a);
        const lg = try trunkLogits(e, a, ids, &kv, null, e.s);
        defer _ = mlx.mlx_array_free(lg);
        const row = try a.alloc(f32, e.cfg.vocab);
        try readLogitsRow(lg, row, e.s);
        try assertHostParity(row, ref_logits, name, 0.99, 0.06);
        try testing.expectEqual(argmaxF32(ref_logits), argmaxF32(row));
    }
}

test "yue2 fixture: NAR velocity at t=0.5 matches the reference (YUE2_TEST_MODEL + YUE2_FIXTURES)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena_inst = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    const fix = try fixturesDir();

    const noise = try readRawF32(io, a, fix, "yue2_noise.f32");
    try testing.expect(noise.len % 64 == 0);
    const frames = noise.len / 64;
    const prefix = try readRawU32(io, a, fix, "yue2_ar_prefill.i32.raw");
    const codec = try readRawU32(io, a, fix, "yue2_codec.i32.raw");
    try testing.expectEqual(frames, codec.len);

    const ids = try a.alloc(u32, prefix.len + frames + 1);
    @memcpy(ids[0..prefix.len], prefix);
    for (ids[prefix.len .. prefix.len + frames], codec) |*d, c| d.* = c + CODEC_OFFSET;
    ids[prefix.len + frames] = MUSIC_END;

    var e = try testEngine(io, a);
    defer e.deinit();
    const s = e.s;

    var ses = try narPrefill(e, a, ids, frames, s);
    defer ses.deinit(a);

    const sh_work = [_]c_int{ 1, @intCast(frames + 2), 64 };
    var work = try zerosA(&sh_work, .bfloat16, s);
    defer _ = mlx.mlx_array_free(work);
    const state = try f32ToBf16(noise, &[_]c_int{ 1, @intCast(frames), 64 }, s);
    defer _ = mlx.mlx_array_free(state);
    const work0 = work;
    work = try sliceUpdateA(work, state, &[_]c_int{ 0, 1, 0 }, &[_]c_int{ 1, @intCast(frames + 1), 64 }, s);
    _ = mlx.mlx_array_free(work0);

    const vel = try narVelocity(e, &ses, work, 0.0, s);
    defer _ = mlx.mlx_array_free(vel);
    const ref_vel = try readRawF32(io, a, fix, "yue2_velocity.f32");
    try assertArrayParity(vel, ref_vel, "velocity t=0.5", 0.98, 0.1, s);
}

test "yue2 fixture: narSynthesize latents match the reference at the fixture's steps (YUE2_TEST_MODEL + YUE2_FIXTURES)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena_inst = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_inst.deinit();
    const a = arena_inst.allocator();
    const fix = try fixturesDir();
    const steps = try readMetaU32(io, a, fix, "steps");
    const seed = try readMetaU32(io, a, fix, "seed");

    const noise = try readRawF32(io, a, fix, "yue2_noise.f32");
    const ref_latents = try readRawF32(io, a, fix, "yue2_nar_latent.f32");
    try testing.expectEqual(noise.len, ref_latents.len);
    const prefix = try readRawU32(io, a, fix, "yue2_ar_prefill.i32.raw");
    const codec = try readRawU32(io, a, fix, "yue2_codec.i32.raw");

    var e = try testEngine(io, a);
    defer e.deinit();
    var out = try narSynthesize(e, a, prefix, codec, seed, steps, noise, e.s);
    defer out.deinit(a);
    try assertHostParity(out.latents, ref_latents, "nar latents", 0.97, 0.12);
}

pub fn parseCot(s: ?[]const u8) ?Cot {
    if (s == null) return null;
    const str = s.?;
    if (std.ascii.eqlIgnoreCase(str, "off")) return .off;
    if (std.ascii.eqlIgnoreCase(str, "melody")) return .melody;
    if (std.ascii.eqlIgnoreCase(str, "full")) return .full;
    return null;
}

pub fn generate(self: *Engine, allocator: std.mem.Allocator, req: MusicRequest, progress: ?sse.Progress) ![]u8 {
    _ = self; _ = allocator; _ = req; _ = progress;
    return error.NotImplemented;
}
