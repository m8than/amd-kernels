#!/usr/bin/env python3
"""
Parity tests for all GEMM HipKittens kernels.
Each test runs in a subprocess to avoid HIP error cascading.
Tests gemm-basic (10), gemm-batched (4), and gemm-fused (4) kernels.
"""

import subprocess
import sys
import os
import json
import textwrap

PYTHON = "/root/aiter-hipkittens/amd-kernels/.venv/bin/python"
KERNELS_DIR = "/root/aiter-hipkittens/amd-kernels/kernels"

# ============================================================================
# Individual test scripts (run in subprocess)
# ============================================================================

TEST_GEMM_A16W16 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_a16w16_tk",
    "{KDIR}/gemm-basic/gemm_a16w16/gemm_a16w16_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=256, BLOCK_N=256, BLOCK_K=64
M, N, K = 256, 256, 256
torch.manual_seed(42)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
C_ref = (A.float() @ B.float().T).bfloat16()

mod.dispatch(A, B, C)
torch.cuda.synchronize()

md = (C.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.02}}))
""")

TEST_GEMM_A16W16_ATOMIC = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_a16w16_atomic_tk",
    "{KDIR}/gemm-basic/gemm_a16w16_atomic/gemm_a16w16_atomic_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64; C is float32 (atomic accumulate)
M, N, K = 128, 128, 128
torch.manual_seed(42)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
C_ref = A.float() @ B.float().T

mod.dispatch(A, B, C)
torch.cuda.synchronize()

md = (C - C_ref).abs().max().item()
re = md / (C_ref.abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_GEMM_A16W16_GATED = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_a16w16_gated_tk",
    "{KDIR}/gemm-basic/gemm_a16w16_gated/gemm_a16w16_gated_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128 (total B rows), BLOCK_K=64
# B is [N, K] where N = 2*half_n. Output is [M, N/2].
M, N, K = 128, 256, 128  # N=256, half=128
torch.manual_seed(42)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
half_n = N // 2
C = torch.zeros(M, half_n, dtype=torch.bfloat16, device="cuda")

B0 = B[:half_n, :]
B1 = B[half_n:, :]
acc0 = A.float() @ B0.float().T
acc1 = A.float() @ B1.float().T
gate = acc1 * torch.sigmoid(acc1)  # silu
C_ref = (acc0 * gate).bfloat16()

mod.dispatch(A, B, C)
torch.cuda.synchronize()

md = (C.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_GEMM_A8W8 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_a8w8_tk",
    "{KDIR}/gemm-basic/gemm_a8w8/gemm_a8w8_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
# dispatch(A_bf16, B_bf16, a_scale, b_scale, C)
# a_scale: fp32[M], b_scale: fp32[N]
M, N, K = 128, 128, 128
torch.manual_seed(42)
A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
B_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
a_scale = torch.rand(M, dtype=torch.float32, device="cuda") * 0.1 + 0.01
b_scale = torch.rand(N, dtype=torch.float32, device="cuda") * 0.1 + 0.01
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

C_ref = ((A_int8.float() @ B_int8.float().T) * a_scale[:, None] * b_scale[None, :]).bfloat16()

A_bf16 = A_int8.to(torch.bfloat16)
B_bf16 = B_int8.to(torch.bfloat16)
mod.dispatch(A_bf16, B_bf16, a_scale, b_scale, C)
torch.cuda.synchronize()

md = (C.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_GEMM_A8W8_BLOCKSCALE = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_a8w8_blockscale_tk",
    "{KDIR}/gemm-basic/gemm_a8w8_blockscale/gemm_a8w8_blockscale_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_K=128
# dispatch(A_bf16, B_bf16, C, a_scale, b_scale)
# a_scale: fp32[M, K//128], b_scale: fp32[K//128, N]
M, N, K = 128, 128, 256
GROUP_K = 128
torch.manual_seed(42)
A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
B_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
num_k_groups = K // GROUP_K
a_scale = torch.rand(M, num_k_groups, dtype=torch.float32, device="cuda") * 0.1 + 0.01
b_scale = torch.rand(num_k_groups, N, dtype=torch.float32, device="cuda") * 0.1 + 0.01
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

C_ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
for kg in range(num_k_groups):
    ks, ke = kg * GROUP_K, (kg + 1) * GROUP_K
    partial = A_int8[:, ks:ke].float() @ B_int8[:, ks:ke].float().T
    C_ref += partial * a_scale[:, kg:kg+1] * b_scale[kg:kg+1, :]
C_ref = C_ref.bfloat16()

A_bf16 = A_int8.to(torch.bfloat16)
B_bf16 = B_int8.to(torch.bfloat16)
mod.dispatch(A_bf16, B_bf16, C, a_scale, b_scale)
torch.cuda.synchronize()

md = (C.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_GEMM_A8W8_PER_TOKEN_SCALE = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_a8w8_per_token_scale_tk",
    "{KDIR}/gemm-basic/gemm_a8w8_per_token_scale/gemm_a8w8_per_token_scale_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
# dispatch(A_bf16, B_bf16, C, a_scale, b_scale)
# a_scale: fp32[M], b_scale: fp32[N]
M, N, K = 128, 128, 128
torch.manual_seed(42)
A_int8 = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
B_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
a_scale = torch.rand(M, dtype=torch.float32, device="cuda") * 0.1 + 0.01
b_scale = torch.rand(N, dtype=torch.float32, device="cuda") * 0.1 + 0.01
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

C_ref = ((A_int8.float() @ B_int8.float().T) * a_scale[:, None] * b_scale[None, :]).bfloat16()

A_bf16 = A_int8.to(torch.bfloat16)
B_bf16 = B_int8.to(torch.bfloat16)
mod.dispatch(A_bf16, B_bf16, C, a_scale, b_scale)
torch.cuda.synchronize()

md = (C.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_GEMM_A16W8_BLOCKSCALE = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_a16w8_blockscale_tk",
    "{KDIR}/gemm-basic/gemm_a16w8_blockscale/gemm_a16w8_blockscale_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_K=128
# dispatch(A, B_bf16, C, b_scale)
# b_scale: fp32[N, K//128]
M, N, K = 128, 128, 256
GROUP_K = 128
torch.manual_seed(42)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
num_k_groups = K // GROUP_K
b_scale = torch.rand(N, num_k_groups, dtype=torch.float32, device="cuda") * 0.1 + 0.01
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

C_ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
for kg in range(num_k_groups):
    ks, ke = kg * GROUP_K, (kg + 1) * GROUP_K
    partial = A[:, ks:ke].float() @ B_int8[:, ks:ke].float().T
    C_ref += partial * b_scale[:, kg:kg+1].T
C_ref = C_ref.bfloat16()

B_bf16 = B_int8.to(torch.bfloat16)
mod.dispatch(A, B_bf16, C, b_scale)
torch.cuda.synchronize()

md = (C.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_GEMM_A16WFP4 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_a16wfp4_tk",
    "{KDIR}/gemm-basic/gemm_a16wfp4/gemm_a16wfp4_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, SCALE_GROUP=32
# dispatch(A, B, C, b_scales)
# b_scales: fp32[N, K//32]
M, N, K = 128, 128, 128
SCALE_GROUP = 32
torch.manual_seed(42)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
num_sg = K // SCALE_GROUP
b_scales = torch.ones(N, num_sg, dtype=torch.float32, device="cuda")
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

# With unit scales, kernel does A @ B^T
C_ref = (A.float() @ B.float().T).bfloat16()

mod.dispatch(A, B, C, b_scales)
torch.cuda.synchronize()

md = (C.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_GEMM_A8WFP4 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_a8wfp4_tk",
    "{KDIR}/gemm-basic/gemm_a8wfp4/gemm_a8wfp4_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
# dispatch(A, B, a_scale, C)
# a_scale: fp32[M]
M, N, K = 128, 128, 128
torch.manual_seed(42)
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
B = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
a_scale = torch.rand(M, dtype=torch.float32, device="cuda") * 0.1 + 0.01
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

C_ref = ((A.float() @ B.float().T) * a_scale[:, None].cuda()).bfloat16()

mod.dispatch(A, B, a_scale, C)
torch.cuda.synchronize()

md = (C.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_GEMM_AFP4WFP4 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("gemm_afp4wfp4_tk",
    "{KDIR}/gemm-basic/gemm_afp4wfp4/gemm_afp4wfp4_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
# dispatch(A, B, C)
M, N, K = 128, 128, 128
torch.manual_seed(42)
A = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
B = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

C_ref = (A.float() @ B.float().T).bfloat16()

mod.dispatch(A, B, C)
torch.cuda.synchronize()

md = (C.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

# ============================================================================
# Batched kernels
# ============================================================================

TEST_BATCHED_GEMM_BF16 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("batched_gemm_bf16_tk",
    "{KDIR}/gemm-batched/batched_gemm_bf16/batched_gemm_bf16_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
# dispatch(A_hk, B_hk, C_hk, bias_hk)
# HK layout: (1, B, M/N, K/N)
BATCH, M, N, K = 4, 128, 128, 128
torch.manual_seed(42)
A = torch.randn(BATCH, M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(BATCH, K, N, dtype=torch.bfloat16, device="cuda")
bias = torch.randn(BATCH, 1, N, dtype=torch.bfloat16, device="cuda")

C_ref = torch.bmm(A.float(), B.float()).bfloat16()
C_ref = (C_ref.float() + bias.float()).bfloat16()

A_hk = A.unsqueeze(0).contiguous()
B_hk = B.transpose(-2, -1).unsqueeze(0).contiguous()  # (1, B, N, K)
C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")
bias_hk = bias.unsqueeze(0).contiguous()

mod.dispatch(A_hk, B_hk, C_hk, bias_hk)
torch.cuda.synchronize()

C_out = C_hk.squeeze(0)
md = (C_out.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_BATCHED_GEMM_A8W8 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("batched_gemm_a8w8_tk",
    "{KDIR}/gemm-batched/batched_gemm_a8w8/batched_gemm_a8w8_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
# dispatch(A_hk, B_hk, a_scale_hk, b_scale_hk, C_hk, bias_hk)
BATCH, M, N, K = 4, 128, 128, 128
torch.manual_seed(42)
A_int8 = torch.randint(-128, 127, (BATCH, M, K), dtype=torch.int8)
B_int8 = torch.randint(-128, 127, (BATCH, K, N), dtype=torch.int8)
a_scale = torch.rand(BATCH, M, 1, dtype=torch.float32) * 0.1
b_scale = torch.rand(BATCH, 1, N, dtype=torch.float32) * 0.1
bias = torch.randn(BATCH, 1, N, dtype=torch.bfloat16)

C_ref = torch.bmm(A_int8.float(), B_int8.float()) * a_scale * b_scale
C_ref = (C_ref + bias.float()).bfloat16()

A_bf16 = A_int8.float().bfloat16().cuda()
B_bf16 = B_int8.float().bfloat16().cuda()
a_scale_gpu = a_scale.cuda()
b_scale_gpu = b_scale.cuda()
bias_gpu = bias.cuda()

A_hk = A_bf16.unsqueeze(0).contiguous()
B_hk = B_bf16.transpose(-2, -1).unsqueeze(0).contiguous()  # (1, B, N, K)
a_scale_hk = a_scale_gpu.squeeze(-1).unsqueeze(0).unsqueeze(2).contiguous()  # (1, B, 1, M)
b_scale_hk = b_scale_gpu.squeeze(-2).unsqueeze(0).unsqueeze(2).contiguous()  # (1, B, 1, N)
C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")
bias_hk = bias_gpu.unsqueeze(0).contiguous()

mod.dispatch(A_hk, B_hk, a_scale_hk, b_scale_hk, C_hk, bias_hk)
torch.cuda.synchronize()

C_out = C_hk.squeeze(0)
C_ref_gpu = C_ref.cuda()
md = (C_out.float() - C_ref_gpu.float()).abs().max().item()
re = md / (C_ref_gpu.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.10}}))
""")

TEST_BATCHED_GEMM_A16WFP4 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("batched_gemm_a16wfp4_tk",
    "{KDIR}/gemm-batched/batched_gemm_a16wfp4/batched_gemm_a16wfp4_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
# dispatch(A_hk, B_hk, C_hk)
BATCH, M, N, K = 4, 128, 128, 128
torch.manual_seed(42)
A = torch.randn(BATCH, M, K, dtype=torch.bfloat16, device="cuda")
B_dequant = (torch.randn(BATCH, K, N, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)

C_ref = torch.bmm(A.float(), B_dequant.float()).bfloat16()

A_hk = A.unsqueeze(0).contiguous()
B_hk = B_dequant.transpose(-2, -1).unsqueeze(0).contiguous()  # (1, B, N, K)
C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")

mod.dispatch(A_hk, B_hk, C_hk)
torch.cuda.synchronize()

C_out = C_hk.squeeze(0)
md = (C_out.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

TEST_BATCHED_GEMM_AFP4WFP4 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("batched_gemm_afp4wfp4_tk",
    "{KDIR}/gemm-batched/batched_gemm_afp4wfp4/batched_gemm_afp4wfp4_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
# dispatch(A_hk, B_hk, C_hk)
BATCH, M, N, K = 4, 128, 128, 128
torch.manual_seed(42)
A = (torch.randn(BATCH, M, K, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)
B = (torch.randn(BATCH, K, N, dtype=torch.bfloat16, device="cuda") * 0.5).clamp(-6, 6)

C_ref = torch.bmm(A.float(), B.float()).bfloat16()

A_hk = A.unsqueeze(0).contiguous()
B_hk = B.transpose(-2, -1).unsqueeze(0).contiguous()
C_hk = torch.zeros(1, BATCH, M, N, dtype=torch.bfloat16, device="cuda")

mod.dispatch(A_hk, B_hk, C_hk)
torch.cuda.synchronize()

C_out = C_hk.squeeze(0)
md = (C_out.float() - C_ref.float()).abs().max().item()
re = md / (C_ref.float().abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.05}}))
""")

# ============================================================================
# Fused kernels
# ============================================================================

TEST_FUSED_A8W8_BS_A16W16 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("fused_gemm_a8w8_blockscale_a16w16_tk",
    "{KDIR}/gemm-fused/fused_gemm_a8w8_blockscale_a16w16/fused_gemm_a8w8_blockscale_a16w16_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# dispatch(a_fp8, b_fp8, c_fp8, a_scale, b_scale, a_bf16, b_bf16, c_bf16)
M, N_fp8, N_bf16, K = 256, 256, 256, 256
GROUP_K = 128
torch.manual_seed(42)

a_fp8 = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)
b_fp8 = (torch.randn(N_fp8, K, device="cuda") * 0.1).to(torch.bfloat16)
scale_k = K // GROUP_K
a_scale = (torch.rand(M, scale_k, dtype=torch.float32, device="cuda") * 2.0)
b_scale = (torch.rand(scale_k, N_fp8, dtype=torch.float32, device="cuda") * 2.0)
a_bf16 = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)
b_bf16 = (torch.randn(N_bf16, K, device="cuda") * 0.1).to(torch.bfloat16)
c_fp8 = torch.zeros(M, N_fp8, dtype=torch.bfloat16, device="cuda")
c_bf16 = torch.zeros(M, N_bf16, dtype=torch.bfloat16, device="cuda")

# Reference: blockscale path
c_fp8_ref = torch.zeros(M, N_fp8, dtype=torch.float32, device="cuda")
for kb in range(scale_k):
    ks, ke = kb * GROUP_K, (kb + 1) * GROUP_K
    partial = a_fp8[:, ks:ke].float() @ b_fp8[:, ks:ke].float().T
    c_fp8_ref += partial * a_scale[:, kb:kb+1] * b_scale[kb:kb+1, :]
# Reference: bf16 path
c_bf16_ref = (a_bf16.float() @ b_bf16.float().T)

mod.dispatch(a_fp8, b_fp8, c_fp8, a_scale, b_scale, a_bf16, b_bf16, c_bf16)
torch.cuda.synchronize()

md_fp8 = (c_fp8.float() - c_fp8_ref).abs().max().item()
md_bf16 = (c_bf16.float() - c_bf16_ref).abs().max().item()
md = max(md_fp8, md_bf16)
re_fp8 = md_fp8 / (c_fp8_ref.abs().max().item() + 1e-8)
re_bf16 = md_bf16 / (c_bf16_ref.abs().max().item() + 1e-8)
ok = re_fp8 < 0.10 and re_bf16 < 0.10
print(json.dumps({{"max_diff": md, "rel_err": max(re_fp8, re_bf16),
    "passed": ok, "notes": f"fp8_re={{re_fp8:.4f}} bf16_re={{re_bf16:.4f}}"}}))
""")

TEST_FUSED_A8W8_BS_MUL_ADD = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("fused_gemm_a8w8_blockscale_mul_add_tk",
    "{KDIR}/gemm-fused/fused_gemm_a8w8_blockscale_mul_add/fused_gemm_a8w8_blockscale_mul_add_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# dispatch(a, b, c, a_scale, b_scale, c_a, c_b)
M, N, K = 256, 256, 256
GROUP_K = 128
torch.manual_seed(42)

a = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)
b = (torch.randn(N, K, device="cuda") * 0.1).to(torch.bfloat16)
scale_k = K // GROUP_K
a_scale = torch.rand(M, scale_k, dtype=torch.float32, device="cuda") * 2.0
b_scale = torch.rand(scale_k, N, dtype=torch.float32, device="cuda") * 2.0
c_a = (torch.randn(M, N, device="cuda") * 0.5).to(torch.bfloat16)
c_b = (torch.randn(M, N, device="cuda") * 0.5).to(torch.bfloat16)
c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

# Reference
gemm_out = torch.zeros(M, N, dtype=torch.float32, device="cuda")
for kb in range(scale_k):
    ks, ke = kb * GROUP_K, (kb + 1) * GROUP_K
    partial = a[:, ks:ke].float() @ b[:, ks:ke].float().T
    gemm_out += partial * a_scale[:, kb:kb+1] * b_scale[kb:kb+1, :]
ref = c_a.float() * gemm_out + c_b.float()

mod.dispatch(a, b, c, a_scale, b_scale, c_a, c_b)
torch.cuda.synchronize()

md = (c.float() - ref).abs().max().item()
re = md / (ref.abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.10}}))
""")

TEST_FUSED_AFP4WFP4_A16W16 = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("fused_gemm_afp4wfp4_a16w16_tk",
    "{KDIR}/gemm-fused/fused_gemm_afp4wfp4_a16w16/fused_gemm_afp4wfp4_a16w16_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# dispatch(a_fp4, b_fp4, c_fp4, a_fp4_scale, b_fp4_scale, a_bf16, b_bf16, c_bf16)
M, N_fp4, N_bf16, K = 256, 256, 256, 256
SCALE_GROUP = 32
torch.manual_seed(42)

a_fp4 = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)
b_fp4 = (torch.randn(N_fp4, K, device="cuda") * 0.1).to(torch.bfloat16)
scale_k = K // SCALE_GROUP
a_fp4_scale = torch.rand(M, scale_k, dtype=torch.float32, device="cuda") * 2.0
b_fp4_scale = torch.rand(N_fp4, scale_k, dtype=torch.float32, device="cuda") * 2.0
a_bf16 = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)
b_bf16 = (torch.randn(N_bf16, K, device="cuda") * 0.1).to(torch.bfloat16)
c_fp4 = torch.zeros(M, N_fp4, dtype=torch.bfloat16, device="cuda")
c_bf16 = torch.zeros(M, N_bf16, dtype=torch.bfloat16, device="cuda")

# Reference: FP4 path with per-block scales
c_fp4_ref = torch.zeros(M, N_fp4, dtype=torch.float32, device="cuda")
for kb in range(scale_k):
    ks, ke = kb * SCALE_GROUP, (kb + 1) * SCALE_GROUP
    partial = a_fp4[:, ks:ke].float() @ b_fp4[:, ks:ke].float().T
    c_fp4_ref += partial * a_fp4_scale[:, kb:kb+1] * b_fp4_scale[:, kb:kb+1].T
# Reference: bf16 path
c_bf16_ref = a_bf16.float() @ b_bf16.float().T

mod.dispatch(a_fp4, b_fp4, c_fp4, a_fp4_scale, b_fp4_scale, a_bf16, b_bf16, c_bf16)
torch.cuda.synchronize()

md_fp4 = (c_fp4.float() - c_fp4_ref).abs().max().item()
md_bf16 = (c_bf16.float() - c_bf16_ref).abs().max().item()
md = max(md_fp4, md_bf16)
re_fp4 = md_fp4 / (c_fp4_ref.abs().max().item() + 1e-8)
re_bf16 = md_bf16 / (c_bf16_ref.abs().max().item() + 1e-8)
ok = re_fp4 < 0.10 and re_bf16 < 0.10
print(json.dumps({{"max_diff": md, "rel_err": max(re_fp4, re_bf16),
    "passed": ok, "notes": f"fp4_re={{re_fp4:.4f}} bf16_re={{re_bf16:.4f}}"}}))
""")

TEST_FUSED_AFP4WFP4_MUL_ADD = textwrap.dedent(r"""
import torch, importlib.util, json
spec = importlib.util.spec_from_file_location("fused_gemm_afp4wfp4_mul_add_tk",
    "{KDIR}/gemm-fused/fused_gemm_afp4wfp4_mul_add/fused_gemm_afp4wfp4_mul_add_tk.cpython-312-x86_64-linux-gnu.so")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# dispatch(a, b, c, a_scale, b_scale, c_a, c_b)
M, N, K = 256, 256, 256
SCALE_GROUP = 32
torch.manual_seed(42)

a = (torch.randn(M, K, device="cuda") * 0.1).to(torch.bfloat16)
b = (torch.randn(N, K, device="cuda") * 0.1).to(torch.bfloat16)
scale_k = K // SCALE_GROUP
a_scale = torch.rand(M, scale_k, dtype=torch.float32, device="cuda") * 2.0
b_scale = torch.rand(N, scale_k, dtype=torch.float32, device="cuda") * 2.0
c_a = (torch.randn(M, N, device="cuda") * 0.5).to(torch.bfloat16)
c_b = (torch.randn(M, N, device="cuda") * 0.5).to(torch.bfloat16)
c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

# Reference
gemm_out = torch.zeros(M, N, dtype=torch.float32, device="cuda")
for kb in range(scale_k):
    ks, ke = kb * SCALE_GROUP, (kb + 1) * SCALE_GROUP
    partial = a[:, ks:ke].float() @ b[:, ks:ke].float().T
    gemm_out += partial * a_scale[:, kb:kb+1] * b_scale[:, kb:kb+1].T
ref = c_a.float() * gemm_out + c_b.float()

mod.dispatch(a, b, c, a_scale, b_scale, c_a, c_b)
torch.cuda.synchronize()

md = (c.float() - ref).abs().max().item()
re = md / (ref.abs().max().item() + 1e-8)
print(json.dumps({{"max_diff": md, "rel_err": re, "passed": re < 0.10}}))
""")


# ============================================================================
# Test runner
# ============================================================================

ALL_TESTS = [
    # (name, test_script)
    ("gemm_a16w16", TEST_GEMM_A16W16),
    ("gemm_a16w16_atomic", TEST_GEMM_A16W16_ATOMIC),
    ("gemm_a16w16_gated", TEST_GEMM_A16W16_GATED),
    ("gemm_a8w8", TEST_GEMM_A8W8),
    ("gemm_a8w8_blockscale", TEST_GEMM_A8W8_BLOCKSCALE),
    ("gemm_a8w8_per_token_scale", TEST_GEMM_A8W8_PER_TOKEN_SCALE),
    ("gemm_a16w8_blockscale", TEST_GEMM_A16W8_BLOCKSCALE),
    ("gemm_a16wfp4", TEST_GEMM_A16WFP4),
    ("gemm_a8wfp4", TEST_GEMM_A8WFP4),
    ("gemm_afp4wfp4", TEST_GEMM_AFP4WFP4),
    ("batched_gemm_bf16", TEST_BATCHED_GEMM_BF16),
    ("batched_gemm_a8w8", TEST_BATCHED_GEMM_A8W8),
    ("batched_gemm_a16wfp4", TEST_BATCHED_GEMM_A16WFP4),
    ("batched_gemm_afp4wfp4", TEST_BATCHED_GEMM_AFP4WFP4),
    ("fused_a8w8_bs_a16w16", TEST_FUSED_A8W8_BS_A16W16),
    ("fused_a8w8_bs_mul_add", TEST_FUSED_A8W8_BS_MUL_ADD),
    ("fused_afp4wfp4_a16w16", TEST_FUSED_AFP4WFP4_A16W16),
    ("fused_afp4wfp4_mul_add", TEST_FUSED_AFP4WFP4_MUL_ADD),
]


def run_test(name, script):
    """Run a test in a subprocess to isolate HIP errors."""
    code = script.replace("{KDIR}", KERNELS_DIR)
    try:
        result = subprocess.run(
            [PYTHON, "-c", code],
            capture_output=True, text=True, timeout=60
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            # Extract short error
            for line in stderr.split("\n"):
                if "Error" in line or "error" in line:
                    return {"status": "FAIL", "notes": line[:120]}
            return {"status": "FAIL", "notes": stderr[-200:] if stderr else "unknown error"}

        # Parse JSON output
        try:
            data = json.loads(stdout.split("\n")[-1])
            status = "PASS" if data.get("passed") else "FAIL"
            return {
                "status": status,
                "max_diff": data.get("max_diff"),
                "rel_err": data.get("rel_err"),
                "notes": data.get("notes", ""),
            }
        except (json.JSONDecodeError, IndexError):
            return {"status": "FAIL", "notes": f"bad output: {stdout[:100]}"}

    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "notes": "timeout (60s)"}
    except Exception as e:
        return {"status": "FAIL", "notes": str(e)[:120]}


def main():
    print("=" * 70)
    print("GEMM Kernel Parity Tests - HipKittens on AMD MI325X (gfx942)")
    print("=" * 70)

    results = []

    sections = [
        ("gemm-basic (10 kernels)", ALL_TESTS[:10]),
        ("gemm-batched (4 kernels)", ALL_TESTS[10:14]),
        ("gemm-fused (4 kernels)", ALL_TESTS[14:]),
    ]

    for section_name, tests in sections:
        print(f"\n--- {section_name} ---")
        for name, script in tests:
            r = run_test(name, script)
            r["name"] = name
            results.append(r)

            status = r["status"]
            md = r.get("max_diff")
            re = r.get("rel_err")
            notes = r.get("notes", "")
            md_str = f"md={md:.6f}" if md is not None else ""
            re_str = f"re={re:.6f}" if re is not None else ""
            detail = ", ".join(filter(None, [md_str, re_str, notes]))
            print(f"  [{status}] {name:<35} {detail}")

    # Summary
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    print(f"SUMMARY: PASS={passed} FAIL={failed} SKIP={skipped} TOTAL={len(results)}")
    print("=" * 70)

    print(f"\n{'Kernel':<35} {'Status':<6} {'Max Diff':<14} {'Rel Err':<14} {'Notes'}")
    print("-" * 100)
    for r in results:
        md = f"{r['max_diff']:.6f}" if r.get("max_diff") is not None else "N/A"
        re = f"{r['rel_err']:.6f}" if r.get("rel_err") is not None else "N/A"
        print(f"{r['name']:<35} {r['status']:<6} {md:<14} {re:<14} {r.get('notes', '')}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
