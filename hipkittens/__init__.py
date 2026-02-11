"""
HipKittens — High-performance HIP C++ GPU kernels for AMD CDNA3/CDNA4.

Usage:
    import hipkittens as hk

    # Normalization
    hk.rmsnorm_fwd(x, w, out, eps, N)
    hk.fused_add_rmsnorm_fwd(x, res, w, out, res_out, eps, N)
    hk.layernorm_fwd(x, w, b, out, eps, N)

    # Activations
    hk.silu_fwd(x, out)
    hk.gelu_fwd(x, out)
    hk.silu_and_mul_fwd(xg, out)

    # GEMM
    hk.gemm.a16w16.dispatch(A, B, C)
    hk.gemm.a8w8.dispatch(A, B, C, scale_a, scale_b)

    # Access any kernel module directly
    hk.modules['rmsnorm_tk'].rmsnorm_fwd(...)
"""

import importlib.util
import os
import sys
import types

_KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

# Registry of all loaded modules
modules = {}

# ─── Module loading ───

def _load_so(name, so_path):
    """Load a compiled .so kernel module by path."""
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _find_and_load_all():
    """Discover and load all *_tk.so kernel modules."""
    suffix = ".cpython-312-x86_64-linux-gnu.so"
    # Also try generic .so for other Python versions
    skip_prefixes = ("ds_read_test", "hk_", "mfma_", "gemm_debug", "test_")

    for root, _dirs, files in os.walk(_KERNELS_DIR):
        for f in sorted(files):
            if not f.endswith(suffix):
                continue
            mod_name = f.replace(suffix, "")
            if any(mod_name.startswith(p) for p in skip_prefixes):
                continue
            # Skip duplicate gemm_a16w16 variants (cdna3, hk versions)
            if mod_name in ("gemm_a16w16_cdna3_tk", "gemm_hk_tk", "mfma_test_tk"):
                continue
            so_path = os.path.join(root, f)
            try:
                modules[mod_name] = _load_so(mod_name, so_path)
            except Exception:
                pass  # Skip modules that fail to load

_find_and_load_all()

# ─── Top-level convenience functions (from tested kernels) ───

# Normalization
def _get(mod_name, func_name):
    m = modules.get(mod_name)
    if m is None:
        raise RuntimeError(f"Kernel module '{mod_name}' not loaded. Did you compile it?")
    fn = getattr(m, func_name, None)
    if fn is None:
        raise RuntimeError(f"Function '{func_name}' not found in '{mod_name}'")
    return fn

# --- Normalization ---
def rmsnorm_fwd(x, w, out, eps, N):
    return _get("rmsnorm_tk", "rmsnorm_fwd")(x, w, out, eps, N)

def fused_add_rmsnorm_fwd(x, res, w, out, res_out, eps, N):
    return _get("rmsnorm_tk", "fused_add_rmsnorm_fwd")(x, res, w, out, res_out, eps, N)

def layernorm_fwd(x, w, b, out, eps, N):
    return _get("layernorm_tk", "layernorm_fwd")(x, w, b, out, eps, N)

def fused_add_layernorm_fwd(x, res, w, b, out, res_out, eps, N):
    return _get("layernorm_tk", "fused_add_layernorm_fwd")(x, res, w, b, out, res_out, eps, N)

def rmsnorm_pad(x, w, out, eps, N):
    return _get("fused_add_rmsnorm_pad_tk", "rmsnorm_pad")(x, w, out, eps, N)

def fused_add_rmsnorm_pad(x, res, w, out, res_out, eps, N):
    return _get("fused_add_rmsnorm_pad_tk", "fused_add_rmsnorm_pad")(x, res, w, out, res_out, eps, N)

# --- Activations ---
def silu_fwd(x, out):
    return _get("activation_tk", "silu_fwd")(x, out)

def gelu_fwd(x, out):
    return _get("activation_tk", "gelu_fwd")(x, out)

def relu_fwd(x, out):
    return _get("activation_tk", "relu_fwd")(x, out)

def tanh_fwd(x, out):
    return _get("activation_tk", "tanh_fwd")(x, out)

def gelu_tanh_fwd(x, out):
    return _get("activation_tk", "gelu_tanh_fwd")(x, out)

def silu_and_mul_fwd(xg, out):
    return _get("activation_tk", "silu_and_mul_fwd")(xg, out)

def gelu_and_mul_fwd(xg, out):
    return _get("activation_tk", "gelu_and_mul_fwd")(xg, out)

# --- Softmax ---
def softmax_fwd(x, out):
    return _get("softmax_tk", "softmax_fwd")(x, out)

# --- RoPE ---
def rope_fwd(x, out, cos_half, sin_half):
    return _get("rope_tk", "rope_fwd")(x, out, cos_half, sin_half)

# --- Top-K ---
def topk_fwd(x, out_vals, out_idx, K):
    return _get("topk_tk", "topk_fwd")(x, out_vals, out_idx, K)

# --- Causal Conv1D ---
def causal_conv1d_fwd(*args, **kwargs):
    return _get("causal_conv1d_tk", "causal_conv1d_fwd")(*args, **kwargs)

def causal_conv1d_bias_silu_fwd(*args, **kwargs):
    return _get("causal_conv1d_tk", "causal_conv1d_bias_silu_fwd")(*args, **kwargs)

# --- Fused QKV Split + QK RoPE ---
def fused_qkv_split_qk_rope_fwd(*args, **kwargs):
    return _get("fused_qkv_split_qk_rope_tk", "fused_qkv_split_qk_rope_fwd")(*args, **kwargs)

# --- Quantization ---
def per_token_quant_fwd(*args, **kwargs):
    return _get("quant_tk", "per_token_quant_fwd")(*args, **kwargs)

def fused_rmsnorm_fp8_quant_fwd(*args, **kwargs):
    return _get("fused_fp8_quant_tk", "fused_rmsnorm_fp8_quant_fwd")(*args, **kwargs)

def fused_rmsnorm_mxfp4_quant_fwd(*args, **kwargs):
    return _get("fused_mxfp4_quant_tk", "fused_rmsnorm_mxfp4_quant_fwd")(*args, **kwargs)

# --- L2 Norm ---
def l2norm_fwd(*args, **kwargs):
    return _get("l2norm_tk", "l2norm_fwd")(*args, **kwargs)

def l2norm_bwd(*args, **kwargs):
    return _get("l2norm_tk", "l2norm_bwd")(*args, **kwargs)

# ─── Namespace sub-modules for GEMM / MoE / etc. ───

class _SubModule:
    """Lazy namespace that wraps a kernel module."""
    def __init__(self, mod_name):
        self._mod_name = mod_name
    def __getattr__(self, name):
        return getattr(modules[self._mod_name], name)
    def __repr__(self):
        return f"<hipkittens.{self._mod_name}>"

class _GemmNamespace:
    """hk.gemm.a16w16.dispatch(A, B, C) etc."""
    a16w16          = _SubModule("gemm_a16w16_tk")
    a16w16_atomic   = _SubModule("gemm_a16w16_atomic_tk")
    a16w16_gated    = _SubModule("gemm_a16w16_gated_tk")
    a8w8            = _SubModule("gemm_a8w8_tk")
    a8w8_blockscale = _SubModule("gemm_a8w8_blockscale_tk")
    a8w8_per_token  = _SubModule("gemm_a8w8_per_token_scale_tk")
    a16w8_blockscale= _SubModule("gemm_a16w8_blockscale_tk")
    a16wfp4         = _SubModule("gemm_a16wfp4_tk")
    a8wfp4          = _SubModule("gemm_a8wfp4_tk")
    afp4wfp4        = _SubModule("gemm_afp4wfp4_tk")

class _BatchedGemmNamespace:
    """hk.batched_gemm.bf16.dispatch(...) etc."""
    bf16     = _SubModule("batched_gemm_bf16_tk")
    a8w8     = _SubModule("batched_gemm_a8w8_tk")
    a16wfp4  = _SubModule("batched_gemm_a16wfp4_tk")
    afp4wfp4 = _SubModule("batched_gemm_afp4wfp4_tk")

class _FusedGemmNamespace:
    """hk.fused_gemm.a8w8_blockscale_a16w16.dispatch(...) etc."""
    a8w8_blockscale_a16w16 = _SubModule("fused_gemm_a8w8_blockscale_a16w16_tk")
    a8w8_blockscale_mul_add= _SubModule("fused_gemm_a8w8_blockscale_mul_add_tk")
    afp4wfp4_a16w16        = _SubModule("fused_gemm_afp4wfp4_a16w16_tk")
    afp4wfp4_mul_add       = _SubModule("fused_gemm_afp4wfp4_mul_add_tk")

class _MoeGemmNamespace:
    """hk.moe.gemm(...), hk.moe.silu_fused(...) etc."""
    gemm              = _SubModule("moe_op_tk")
    e2e               = _SubModule("moe_op_e2e_tk")
    gelu              = _SubModule("moe_op_gelu_tk")
    silu_fused        = _SubModule("moe_op_silu_fused_tk")
    a4w4              = _SubModule("moe_op_gemm_a4w4_tk")
    a8w4              = _SubModule("moe_op_gemm_a8w4_tk")
    a8w8              = _SubModule("moe_op_gemm_a8w8_tk")
    a8w8_blockscale   = _SubModule("moe_op_gemm_a8w8_blockscale_tk")
    mxfp4             = _SubModule("moe_op_mxfp4_tk")
    mxfp4_silu_fused  = _SubModule("moe_op_mxfp4_silu_fused_tk")

class _MoeRoutingNamespace:
    """hk.moe_routing.align_block_size(...) etc."""
    align_block_size          = _SubModule("moe_align_block_size_tk")
    bitmatrix                 = _SubModule("moe_bitmatrix_tk")
    expt_data                 = _SubModule("moe_expt_data_tk")
    topk                      = _SubModule("moe_topk_tk")
    sigmoid_top1_fused        = _SubModule("moe_routing_sigmoid_top1_fused_tk")
    quant                     = _SubModule("quant_moe_tk")

class _FeedforwardNamespace:
    """hk.feedforward.gated.dispatch_4096(...) etc."""
    gated            = _SubModule("ff_fused_gated_tk")
    ungated          = _SubModule("ff_fused_ungated_tk")
    kv_cache         = _SubModule("fused_kv_cache_tk")
    mul_add          = _SubModule("fused_mul_add_tk")
    qk_concat        = _SubModule("fused_qk_concat_tk")

class _GdrDecodeNamespace:
    """hk.gdr_decode.recurrent.fused_recurrent_fwd(...) etc."""
    recurrent                  = _SubModule("fused_recurrent_tk")
    sigmoid_gating_recurrent   = _SubModule("fused_sigmoid_gating_recurrent_tk")
    causal_conv1d_split_qkv    = _SubModule("causal_conv1d_split_qkv_tk")
    qkvzba_split               = _SubModule("fused_qkvzba_split_tk")
    utils                      = _SubModule("gdr_utils_tk")

class _GdrPrefillNamespace:
    """hk.gdr_prefill.chunk.chunk_pipeline(...) etc."""
    chunk                    = _SubModule("chunk_tk")
    chunk_delta_h            = _SubModule("chunk_delta_h_tk")
    chunk_o                  = _SubModule("chunk_o_tk")
    cumsum                   = _SubModule("cumsum_tk")
    fused_cumsum_kkt         = _SubModule("fused_cumsum_kkt_tk")
    fused_gdn_gating         = _SubModule("fused_gdn_gating_prefill_tk")
    index                    = _SubModule("index_tk")
    l2norm                   = _SubModule("l2norm_tk")
    solve_tril               = _SubModule("solve_tril_tk")
    wy_representation        = _SubModule("wy_representation_tk")
    causal_conv1d_fwd_split  = _SubModule("causal_conv1d_fwd_split_qkv_tk")

class _AttentionNamespace:
    """hk.attention.mla_decode(...) etc."""
    mla_decode   = _SubModule("mla_decode_rope_tk")
    sparse_mla   = _SubModule("unified_attn_sparse_mla_tk")

gemm         = _GemmNamespace()
batched_gemm = _BatchedGemmNamespace()
fused_gemm   = _FusedGemmNamespace()
moe          = _MoeGemmNamespace()
moe_routing  = _MoeRoutingNamespace()
feedforward  = _FeedforwardNamespace()
gdr_decode   = _GdrDecodeNamespace()
gdr_prefill  = _GdrPrefillNamespace()
attention    = _AttentionNamespace()
