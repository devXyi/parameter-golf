"""Parameter Golf submission -- openai/parameter-golf
Train the best LM that fits in 16MB."""

from __future__ import annotations

# -- P100 / Pascal: force math SDPA before any CUDA ops ---------------------
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import glob, io, math, os, random, time, uuid, zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as grad_ckpt

try:
    import zstandard as _zstd
    _HAVE_ZSTD = True
except ImportError:
    _HAVE_ZSTD = False
    print("WARNING: install zstandard for best compression: pip install zstandard")


# HYPERPARAMETERS

class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH",
                                    "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE",  524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY",  1_000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations            = int(os.environ.get("ITERATIONS",           20_000))
    warmdown_iters        = int(os.environ.get("WARMDOWN_ITERS",        3_000))
    warmup_steps          = int(os.environ.get("WARMUP_STEPS",          20))
    train_batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS",    786_432))
    train_seq_len         = int(os.environ.get("TRAIN_SEQ_LEN",         2_048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init          = float(os.environ.get("QK_GAIN_INIT",          1.5))

    vocab_size       = int(os.environ.get("VOCAB_SIZE",        1_024))
    num_recurrences  = int(os.environ.get("NUM_RECURRENCES",   16))
    model_dim        = int(os.environ.get("MODEL_DIM",         1_024))
    num_heads        = int(os.environ.get("NUM_HEADS",         8))
    num_kv_heads     = int(os.environ.get("NUM_KV_HEADS",      1))
    mlp_intermediate = int(os.environ.get("MLP_INTERMEDIATE",  4_096))
    tie_embeddings   = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base        = float(os.environ.get("ROPE_BASE",        500_000.0))
    logit_softcap    = float(os.environ.get("LOGIT_SOFTCAP",    30.0))
    bigram_hash_size = int(os.environ.get("BIGRAM_HASH_SIZE",   16_384))
    bigram_hash_dim  = int(os.environ.get("BIGRAM_HASH_DIM",    256))

    mem_decay_init       = float(os.environ.get("MEM_DECAY_INIT",       0.9))
    mem_damping          = float(os.environ.get("MEM_DAMPING",          0.99))
    entropy_reg_weight   = float(os.environ.get("ENTROPY_REG_WEIGHT",   0.005))
    depth_dropout_base_p = float(os.environ.get("DEPTH_DROPOUT_BASE_P", 0.15))
    perturb_scale        = float(os.environ.get("PERTURB_SCALE",        0.01))
    qat_noise_scale      = float(os.environ.get("QAT_NOISE_SCALE",      0.01))
    qat_start_step       = int(os.environ.get("QAT_START_STEP",         8_000))
    qat_clip_val         = float(os.environ.get("QAT_CLIP_VAL",         1.5))
    swa_start_frac       = float(os.environ.get("SWA_START_FRAC",       0.4))
    swa_every            = int(os.environ.get("SWA_EVERY",               50))
    eval_stride_div      = int(os.environ.get("EVAL_STRIDE_DIV",        32))
    zstd_dict_size       = int(os.environ.get("ZSTD_DICT_SIZE",         65_536))
    mag_prune_frac       = float(os.environ.get("MAG_PRUNE_FRAC",        0.03))   # 3% pruning
    ent_weight           = float(os.environ.get("ENT_WEIGHT",           1e-4))
    ent_weight_qat       = float(os.environ.get("ENT_WEIGHT_QAT",       5e-5))
    pow2_weight          = float(os.environ.get("POW2_WEIGHT",          5e-5))
    dict_reg_weight      = float(os.environ.get("DICT_REG_WEIGHT",      2e-5))
    dict_reg_centroids   = int(os.environ.get("DICT_REG_CENTROIDS",     256))
    dyn_rec_sparse_w     = float(os.environ.get("DYN_REC_SPARSE_W",     1e-3))
    dyn_rec_threshold    = float(os.environ.get("DYN_REC_THRESHOLD",    0.05))
    # nuclear combo
    ent_route_w          = float(os.environ.get("ENT_ROUTE_W",          1e-3))  # entropy routing sparsity
    zipf_align_w         = float(os.environ.get("ZIPF_ALIGN_W",         2e-4))  # logit Zipf alignment
    freq_gate_init       = float(os.environ.get("FREQ_GATE_INIT",      -1.0))   # freq attention gate init
    dict_cotrain_w       = float(os.environ.get("DICT_COTRAIN_W",       1e-3))  
    dict_cotrain_every   = int(os.environ.get("DICT_COTRAIN_EVERY",     50))    
    mixed_bitwidth       = bool(int(os.environ.get("MIXED_BITWIDTH",    "1")))
    bit_cost_w           = float(os.environ.get("BIT_COST_W",           2e-4))  
    bit_ent_threshold    = float(os.environ.get("BIT_ENT_THRESHOLD",    3.5))   
    cross_ent_w          = float(os.environ.get("CROSS_ENT_W",          3e-5))
    temporal_corr_w      = float(os.environ.get("TEMPORAL_CORR_W",      5e-5))
    # new
    weight_reorder       = bool(int(os.environ.get("WEIGHT_REORDER",    "1")))  
    learned_centers_n    = int(os.environ.get("LEARNED_CENTERS_N",      64))    
    learned_centers_w    = float(os.environ.get("LEARNED_CENTERS_W",    3e-5))  

    embed_lr                   = float(os.environ.get("EMBED_LR",                   0.6))
    tied_embed_lr              = float(os.environ.get("TIED_EMBED_LR",              0.05))
    tied_embed_init_std        = float(os.environ.get("TIED_EMBED_INIT_STD",        0.005))
    matrix_lr                  = float(os.environ.get("MATRIX_LR",                  0.02))
    scalar_lr                  = float(os.environ.get("SCALAR_LR",                  0.02))
    muon_momentum              = float(os.environ.get("MUON_MOMENTUM",              0.99))
    muon_backend_steps         = int(os.environ.get("MUON_BACKEND_STEPS",           5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",   1_500))
    muon_wd                    = float(os.environ.get("MUON_WD",                    0.04))
    beta1                      = float(os.environ.get("BETA1",                      0.9))
    beta2                      = float(os.environ.get("BETA2",                      0.95))
    adam_eps                   = float(os.environ.get("ADAM_EPS",                   1e-8))
    grad_clip_norm             = float(os.environ.get("GRAD_CLIP_NORM",             0.3))


# MUON OPTIMIZER

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16(); X /= X.norm() + eps
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if G.size(0) > G.size(1) else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps,
                 weight_decay=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps,
                                      weight_decay=weight_decay,
                                      nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        dist_on  = dist.is_available() and dist.is_initialized()
        wsize    = dist.get_world_size() if dist_on else 1
        rank     = dist.get_rank()       if dist_on else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, mom = group["lr"], group["momentum"]
            bs, wd  = group["backend_steps"], group["weight_decay"]
            nst     = group["nesterov"]
            total   = sum(p.numel() for p in params)
            flat    = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            cur     = 0
            for i, p in enumerate(params):
                if i % wsize == rank and p.grad is not None:
                    g = p.grad
                    st = self.state[p]
                    if "buf" not in st: st["buf"] = torch.zeros_like(g)
                    buf = st["buf"]
                    buf.mul_(mom).add_(g)
                    if nst: g = g.add(buf, alpha=mom)
                    g = zeropower_via_newtonschulz5(g, steps=bs)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[cur : cur + p.numel()] = g.reshape(-1)
                cur += p.numel()
            if dist_on: dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            cur = 0
            for p in params:
                p.add_(flat[cur : cur + p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                if wd > 0: p.mul_(1.0 - lr * wd)
                cur += p.numel()
        return loss


# TRUE INT6 BIT-PACKING

_I6_OFF  = 31      # signed [-31,31] -> unsigned [0,62]
_I6_MASK = 0x3F    # 6-bit mask

def pack_int6(arr: np.ndarray) -> tuple[np.ndarray, int, tuple]:
    flat = arr.flatten().astype(np.int8)
    pad  = (4 - len(flat) % 4) % 4
    if pad: flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])
    u = (flat.astype(np.int16) + _I6_OFF).astype(np.uint8) & _I6_MASK
    a, b, c, d = u[0::4], u[1::4], u[2::4], u[3::4]
    out = np.empty(len(a) * 3, dtype=np.uint8)
    out[0::3] = (a << 2) | (b >> 4)
    out[1::3] = ((b & 0x0F) << 4) | (c >> 2)
    out[2::3] = ((c & 0x03) << 6) | d
    return out, pad, arr.shape


def unpack_int6(packed: np.ndarray, pad: int, orig_shape: tuple) -> np.ndarray:
    p0, p1, p2 = packed[0::3], packed[1::3], packed[2::3]
    flat = np.empty(len(p0) * 4, dtype=np.uint8)
    flat[0::4] = p0 >> 2
    flat[1::4] = ((p0 & 0x03) << 4) | (p1 >> 4)
    flat[2::4] = ((p1 & 0x0F) << 2) | (p2 >> 6)
    flat[3::4] = p2 & 0x3F
    signed = (flat.astype(np.int16) - _I6_OFF).astype(np.int8)
    n = len(signed) - pad if pad > 0 else len(signed)
    return signed[:n].reshape(orig_shape)


# ZSTD DICTIONARY COMPRESSION

def _zstd_compress(data: bytes, zdict: bytes | None = None, level: int = 22) -> bytes:
    if not _HAVE_ZSTD: return zlib.compress(data, level=9)
    if zdict:
        d = _zstd.ZstdCompressionDict(zdict)
        return _zstd.ZstdCompressor(level=level, dict_data=d).compress(data)
    return _zstd.ZstdCompressor(level=level).compress(data)


def _zstd_decompress(data: bytes, zdict: bytes | None = None) -> bytes:
    if not _HAVE_ZSTD: return zlib.decompress(data)
    if zdict:
        d = _zstd.ZstdCompressionDict(zdict)
        return _zstd.ZstdDecompressor(dict_data=d).decompress(data)
    return _zstd.ZstdDecompressor().decompress(data)


def train_zstd_dict(samples: list[bytes], dict_size: int) -> bytes | None:
    if not _HAVE_ZSTD or not samples: return None
    try:
        return _zstd.train_dictionary(dict_size, samples).as_bytes()
    except Exception: return None


# INT6 QUANTISATION + EXPORT

CONTROL_PATTERNS = tuple(p for p in os.environ.get(
    "CONTROL_TENSOR_NAME_PATTERNS",
    "smear_base,smear_proj,q_gain,mem_decay,mem_scale,depth_gate,step_bias,"
    "imp_w,hist_logit,logit_temp,freq_bias",
).split(",") if p)

INT_KEEP_FLOAT_MAX_NUMEL = 65_536
FP16_DTYPE  = torch.float16
INT6_CLIP_Q = 99.99 / 100.0
INT6_MAX    = 31


def _q_tensor(t32: np.ndarray):
    """Per-row INT6 with percentile clip -- fallback when no learned scale."""
    if t32.ndim == 2:
        ca  = np.quantile(np.abs(t32), INT6_CLIP_Q, axis=1)
        ca  = np.maximum(ca, 1e-6)
        sc  = (ca / INT6_MAX).astype(np.float16)
        cl  = np.clip(t32, -ca[:, None], ca[:, None])
        q   = np.clip(np.round(cl / sc[:, None].astype(np.float32)),
                      -INT6_MAX, INT6_MAX).astype(np.int8)
        return q, sc, "per_row"
    ca = float(np.quantile(np.abs(t32.flatten()), INT6_CLIP_Q)) or 1e-6
    sc = np.float16(ca / INT6_MAX)
    q  = np.clip(np.round(np.clip(t32, -ca, ca) / float(sc)),
                 -INT6_MAX, INT6_MAX).astype(np.int8)
    return q, sc, "scalar"


def _q_tensor_with_scale(t32: np.ndarray, scale_val: float):
    """Per-row INT6 using learned training scale -> zero train/export mismatch."""
    sc_scalar = max(scale_val, 1e-8)
    if t32.ndim == 2:
        sc  = np.full(t32.shape[0], sc_scalar, dtype=np.float16)
        q   = np.clip(np.round(t32 / sc_scalar), -INT6_MAX, INT6_MAX).astype(np.int8)
        return q, sc, "per_row"
    q = np.clip(np.round(t32 / sc_scalar), -INT6_MAX, INT6_MAX).astype(np.int8)
    return q, np.float16(sc_scalar), "scalar"


def extract_learned_scales(model: nn.Module) -> dict[str, float]:
    """Extract learned quant scales from registry for export alignment."""
    if not QuantLinear._scale_registry:
        return {}
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    result = {}
    for pid, log_scale_param in QuantLinear._scale_registry.items():
        name = id_to_name.get(pid)
        if name is None: continue
        sc = float(torch.exp(log_scale_param.detach()).clamp(
            0.3 / INT6_MAX, 3.0 / INT6_MAX).item())
        result[name] = sc
    return result


def quantize_state_dict(sd: dict, zstd_dict_size: int = 65_536,
                        learned_scales: dict | None = None,
                        weight_reorder: bool = True,
                        bit_selector_map: dict | None = None):
    """INT5/INT6 hybrid + magnitude-sort + zstd dict training."""
    quant, scales, dtypes, passthru, pod, qmeta = {}, {}, {}, {}, {}, {}
    stats  = {"param_count": 0, "raw_float_bytes": 0, "packed_bytes": 0}
    samps: list[bytes] = []
    ls  = learned_scales  or {}
    bsm = bit_selector_map or {}
    _INT5_CLIP_Q_EXP = 0.999

    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        if not t.is_floating_point():
            passthru[name] = t; continue
        if any(p in name for p in CONTROL_PATTERNS):
            passthru[name] = t.float().contiguous(); continue
        if t.numel() <= INT_KEEP_FLOAT_MAX_NUMEL:
            pod[name] = str(t.dtype).removeprefix("torch.")
            passthru[name] = t.to(FP16_DTYPE).contiguous(); continue

        stats["raw_float_bytes"] += t.numel() * t.element_size()
        t32 = t.float().numpy()

        # Magnitude-sort rows for zstd friendliness
        perm = None
        if weight_reorder and t32.ndim == 2:
            perm = np.argsort(np.abs(t32).mean(axis=1))
            t32  = t32[perm]

        # Fix 7: use INT5 scale if this layer's bit_selector chose INT5
        use_int5 = (float(torch.sigmoid(bsm[name]).item()) < 0.5) if name in bsm else False
        if use_int5:
            if t32.ndim == 2:
                ca  = np.maximum(np.quantile(np.abs(t32), _INT5_CLIP_Q_EXP, axis=1), 1e-6)
                sc  = (ca / 15).astype(np.float16)
                q   = np.clip(np.round(np.clip(t32, -ca[:,None], ca[:,None])
                                       / sc[:,None].astype(np.float32)), -15, 15).astype(np.int8)
                scheme = "per_row_int5"
            else:
                ca  = float(np.quantile(np.abs(t32.flatten()), _INT5_CLIP_Q_EXP)) or 1e-6
                sc  = np.float16(ca / 15)
                q   = np.clip(np.round(np.clip(t32, -ca, ca) / float(sc)), -15, 15).astype(np.int8)
                scheme = "scalar_int5"
        elif name in ls:
            q, sc, scheme = _q_tensor_with_scale(t32, ls[name])
        else:
            q, sc, scheme = _q_tensor(t32)
        packed, pad, shape = pack_int6(q)
        samps.append(packed.tobytes())
        entry = {"p": packed.tobytes(), "pad": pad, "sh": shape}
        if perm is not None:
            perm_b = perm.astype(np.int32).tobytes()
            entry["perm"]     = perm_b
            entry["perm_crc"] = zlib.crc32(perm_b) & 0xFFFFFFFF  # corruption guard
        quant[name]  = entry
        scales[name] = sc.tobytes() if hasattr(sc, "tobytes") else float(sc)
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        qmeta[name]  = scheme
        stats["packed_bytes"] += len(packed)

    zdict = train_zstd_dict(samps, zstd_dict_size) if samps else None
    obj = {"__fmt__": "int6_true_pack_v1",
           "q": quant, "sc": scales, "dt": dtypes,
           "pt": passthru, "qm": qmeta, "pod": pod, "zd": zdict}
    return obj, stats


def dequantize_state_dict(obj: dict) -> dict:
    """Reconstruct state dict; reverse magnitude-sort perm with CRC validation."""
    out = {}
    for name, entry in obj["q"].items():
        dtype  = getattr(torch, obj["dt"][name])
        packed = np.frombuffer(entry["p"], dtype=np.uint8)
        q      = unpack_int6(packed, entry["pad"], entry["sh"])
        sc     = obj["sc"][name]
        scheme = obj["qm"].get(name, "scalar")
        if scheme in ("per_row", "per_row_int5"):
            sc_np = np.frombuffer(sc, dtype=np.float16).astype(np.float32)
            recon = (q.astype(np.float32) * sc_np[:, None])
        else:  # scalar or scalar_int5
            recon = q.astype(np.float32) * float(sc)
        if "perm" in entry and recon.ndim == 2:
            perm_b = entry["perm"]
            if "perm_crc" in entry:
                assert (zlib.crc32(perm_b) & 0xFFFFFFFF) == entry["perm_crc"], \
                    f"perm CRC mismatch for {name} -- artifact may be corrupted"
            perm     = np.frombuffer(perm_b, dtype=np.int32)
            inv_perm = np.argsort(perm)
            recon    = recon[inv_perm]
        out[name] = torch.from_numpy(recon).to(dtype).contiguous()
    for name, t in obj["pt"].items():
        ot = t.detach().cpu().contiguous()
        if isinstance(obj["pod"].get(name), str):
            ot = ot.to(getattr(torch, obj["pod"][name])).contiguous()
        out[name] = ot
    return out


# DATA LOADING

def load_data_shard(file: Path) -> Tensor:
    hdr = np.fromfile(file, dtype="<i4", count=256)
    if hdr.size != 256 or int(hdr[0]) != 20240520 or int(hdr[1]) != 1:
        raise ValueError(f"Bad shard: {file}")
    n = int(hdr[2]); off = 256 * 4
    if file.stat().st_size != off + n * 2: raise ValueError(f"Size mismatch: {file}")
    return torch.from_numpy(
        np.fromfile(file, dtype="<u2", count=n, offset=off).astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(f"No files: {pattern}")
        self.idx = 0; self.tokens = load_data_shard(self.files[0]); self.pos = 0

    def _adv(self):
        self.idx = (self.idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.idx]); self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._adv(); continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank = rank; self.world_size = world_size
        self.device = device; self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, accum):
        local = global_tokens // (self.world_size * accum)
        span  = local + 1
        chunk = self.stream.take(span * self.world_size)
        s     = self.rank * span
        t     = chunk[s : s + span].to(torch.int64)
        x = t[:-1].reshape(-1, seq_len)
        y = t[1: ].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size()); tsz = max(sp_vs, vocab_size)
    bb = np.zeros((tsz,), dtype=np.int16)
    hs = np.zeros((tsz,), dtype=np.bool_)
    ib = np.ones((tsz,),  dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        ib[tid] = False
        if sp.is_byte(tid): bb[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("?"): hs[tid] = True; piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hs, dtype=torch.bool,  device=device),
            torch.tensor(ib, dtype=torch.bool,  device=device))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0: raise ValueError("Validation split too short")
    return tokens[: usable + 1]


# SLIDING WINDOW EVALUATION

def eval_val(args, model, rank, world_size, device, _ga,
             val_tokens, bbl, hsl, ibl):
    seq_len = args.train_seq_len
    stride  = max(1, seq_len // args.eval_stride_div)
    N       = val_tokens.numel() - 1
    first   = seq_len - stride
    pos_all = list(range(first, N, stride))
    local   = [p for i, p in enumerate(pos_all) if i % world_size == rank]

    ls = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for sp_pos in local:
            cs = max(0, sp_pos - seq_len + stride)
            ce = min(sp_pos + stride, N)
            wx = val_tokens[cs : ce].to(device, torch.int64, non_blocking=True)
            wy = val_tokens[cs + 1 : ce + 1].to(device, torch.int64, non_blocking=True)
            if wx.numel() < 2: continue
            x = wx[:-1].unsqueeze(0); y = wy[:-1].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x)
            n    = min(stride, x.shape[1])
            lo   = logits[0, -n:].float()
            tgt  = y[0, -n:]
            prev = x[0, -n - 1 : -1] if x.shape[1] > n else x[0, :n]
            ls  += F.cross_entropy(lo, tgt, reduction="sum").to(torch.float64)
            tc  += float(n)
            tb   = bbl[tgt].to(torch.int16)
            tb  += (hsl[tgt] & ~ibl[prev[:n]]).to(torch.int16)
            bc  += tb.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        for t in [ls, tc, bc]: dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = float((ls / tc.clamp_min(1)).item())
    bpb      = float((val_loss / math.log(2.0)) * (tc / bc.clamp_min(1)).item())
    model.train()
    return val_loss, bpb


# INT6 QAT  (noise injection + percentile clip, aligned with export)

def fake_quant_int6_ste(w: Tensor, clip_val: float, noise: float = 0.0) -> Tensor:
    wf = w.float().clamp(-clip_val, clip_val)
    if noise > 0: wf = wf + torch.randn_like(wf) * noise
    if wf.ndim == 2:
        ca  = torch.quantile(wf.abs(), INT6_CLIP_Q, dim=1, keepdim=True).clamp_min(1e-6)
        sc  = ca / INT6_MAX
        wdq = ((wf / sc).clamp(-INT6_MAX, INT6_MAX).round() * sc).to(w.dtype)
    else:
        ca  = float(torch.quantile(wf.abs().flatten(), INT6_CLIP_Q).item()) or 1e-6
        sc  = torch.tensor(ca / INT6_MAX, dtype=torch.float32)
        wdq = ((wf / sc).clamp(-INT6_MAX, INT6_MAX).round() * sc).to(w.dtype)
    return w + (wdq - w).detach()


class QuantLinear(nn.Linear):
    """Linear layer with INT6 fake-quant (STE) and per-matrix learned scale."""
    qat_enabled: bool  = False
    qat_clip:    float = 1.5
    qat_noise:   float = 0.0
    _scale_registry: dict = {}

    def _get_learned_scale(self) -> Tensor | None:
        if not QuantLinear.qat_enabled: return None
        pid = id(self.weight)
        if pid not in QuantLinear._scale_registry:
            with torch.no_grad():
                q99  = torch.quantile(self.weight.float().abs(), 0.9999).clamp_min(1e-6)
                init = math.log(float(q99.item()) / INT6_MAX)
            p = nn.Parameter(torch.tensor(init, device=self.weight.device))
            QuantLinear._scale_registry[pid] = p
        return QuantLinear._scale_registry[pid]

    _bit_selector: "Tensor | None" = None

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if QuantLinear.qat_enabled and w.numel() > INT_KEEP_FLOAT_MAX_NUMEL:
            bs = getattr(self, '_layer_bit_selector', None) or QuantLinear._bit_selector
            if bs is not None:
                w = apply_mixed_bitwidth_quant(w, bs, QuantLinear.qat_clip, QuantLinear.qat_noise)
            else:
                ls = self._get_learned_scale()
                if ls is not None:
                    scale = torch.exp(ls.to(w.device)).clamp(0.3 / INT6_MAX, 3.0 / INT6_MAX)
                    wf    = w.float().clamp(-QuantLinear.qat_clip, QuantLinear.qat_clip)
                    if QuantLinear.qat_noise > 0:
                        wf = wf + torch.randn_like(wf) * QuantLinear.qat_noise
                    wdq   = ((wf / scale).clamp(-INT6_MAX, INT6_MAX).round() * scale).to(w.dtype)
                    w     = w + (wdq - w).detach()
                else:
                    w = fake_quant_int6_ste(w, QuantLinear.qat_clip, QuantLinear.qat_noise)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), b)


# DIFFERENTIABLE ENTROPY REGULARISER

_ENT_BINS      = 63        # one bin per INT6 value [-31..31]
_ENT_SIGMA     = 62.0 / _ENT_BINS   # bin width for RBF kernel
_ENT_SAMPLE_N  = 8_192    # sample size: same gradient signal, 10x faster

def weight_entropy_loss(w: Tensor) -> Tensor:
    """Sampled RBF soft-histogram entropy over INT6 range. O(8192x63)."""
    wf = w.float().flatten()
    if wf.numel() > _ENT_SAMPLE_N:
        idx = torch.randint(0, wf.numel(), (_ENT_SAMPLE_N,), device=w.device)
        wf  = wf[idx]   # random sample -- O(N) -> O(8192), same effect
    centers = torch.linspace(-31.0, 31.0, _ENT_BINS, device=w.device)
    d       = ((wf.unsqueeze(1) - centers) / _ENT_SIGMA).pow(2)
    counts  = torch.exp(-0.5 * d).sum(0)
    p       = counts / counts.sum().clamp_min(1e-8)
    return -(p * (p + 1e-8).log()).sum()


def model_entropy_loss(module: nn.Module) -> Tensor:
    """Sampled entropy loss over large weight matrices."""
    total = None
    for name, p in module.named_parameters():
        if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL: continue
        if any(c in name for c in CONTROL_PATTERNS): continue
        if not p.is_floating_point(): continue
        e = weight_entropy_loss(p)
        total = e if total is None else total + e
    return total if total is not None else torch.tensor(0.0)


# BIT-PLANE ENTROPY REGULARISER

_BP_SHARP = 12.0   # sigmoid sharpness for soft bit extraction

def bitplane_entropy_loss(module: nn.Module) -> Tensor:
    """Differentiable bitplane entropy via fmod+sigmoid. Grad flows fully."""
    total = None
    for name, p in module.named_parameters():
        if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL: continue
        if any(c in name for c in CONTROL_PATTERNS): continue
        if not p.is_floating_point(): continue
        wf = p.float().flatten()
        if wf.numel() > 8192:
            idx = torch.randint(0, wf.numel(), (8192,), device=p.device)
            wf  = wf[idx]
        ws = wf.clamp(-31.0, 31.0) + 31.0   # in [0.0, 62.0], grads flow
        bp_ent = wf.new_zeros(1)
        for bit in range(6):
            step    = float(1 << bit)          # 2^bit:  1,2,4,8,16,32
            x_mod   = ws.fmod(2.0 * step)     # continuous fmod -- differentiable
            soft_bit = torch.sigmoid(_BP_SHARP * (x_mod - step))
            prob1    = soft_bit.mean().clamp(1e-6, 1.0 - 1e-6)
            prob0    = 1.0 - prob1
            bp_ent   = bp_ent - (prob1 * prob1.log() + prob0 * prob0.log())
        total = bp_ent if total is None else total + bp_ent
    return total if total is not None else torch.tensor(0.0)


# POWER-OF-TWO DISTRIBUTION SHAPING

_POW2_CENTERS = [
    0.,
    1., -1., 2., -2., 3., -3.,
    4., -4., 5., -5., 6., -6.,
    8., -8., 16., -16., 31., -31.
]

def pow2_cluster_loss(module: nn.Module, device: torch.device) -> Tensor:
    """Attract weights to hybrid centers {0,+-1-6,+-8,+-16,+-31}. Sampled."""
    centers = torch.tensor(_POW2_CENTERS, device=device)
    total   = None
    for name, p in module.named_parameters():
        if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL: continue
        if any(c in name for c in CONTROL_PATTERNS): continue
        if not p.is_floating_point(): continue
        wf   = p.float().flatten()
        if wf.numel() > 8192:
            wf = wf[torch.randint(0, wf.numel(), (8192,), device=device)]
        dist = (wf.unsqueeze(1) - centers).abs()
        loss = dist.min(dim=1).values.mean()
        total = loss if total is None else total + loss
    return total if total is not None else torch.tensor(0.0)


def dict_cotrain_loss(module: nn.Module) -> Tensor:
    """Byte-repeat proxy: adjacent-diff + 8-byte window loss -> LZ77."""
    eligible_names = [
        name for name, p in module.named_parameters()
        if p.numel() > INT_KEEP_FLOAT_MAX_NUMEL
        and not any(c in name for c in CONTROL_PATTERNS)
        and p.is_floating_point()
    ]
    if not eligible_names: return torch.tensor(0.0)
    per_tensor = (max(32, 4096 // len(eligible_names)) // 4) * 4

    chunks = []
    for name, p in module.named_parameters():
        if name not in eligible_names: continue
        wf  = p.float().flatten()
        n   = min(per_tensor, wf.numel() - (wf.numel() % 4))
        if n < 4: continue
        max_start = max(1, wf.numel() - n)
        start = int(torch.randint(0, max_start, (1,)).item())
        chunks.append(wf[start : start + n])
    if not chunks: return torch.tensor(0.0)

    seq  = torch.cat(chunks).clamp(-31.0, 31.0) + 31.0
    n4   = (seq.numel() // 4) * 4; seq = seq[:n4]
    a, b, c, d = seq[0::4], seq[1::4], seq[2::4], seq[3::4]

    byte0 = a * 4.0 + torch.floor(b / 16.0)
    byte1 = (b - torch.floor(b / 16.0) * 16.0) * 16.0 + torch.floor(c / 4.0)
    byte2 = (c - torch.floor(c / 4.0) * 4.0) * 64.0 + d
    bs    = torch.cat([byte0, byte1, byte2])

    # Adjacent run loss
    run_loss = (bs[1:] - bs[:-1]).abs().mean()

    if bs.numel() >= 32:
        w_len  = 8
        n_win  = (bs.numel() // (2 * w_len)) * 2 * w_len
        bsw    = bs[:n_win]
        win_a  = bsw[:n_win // 2].reshape(-1, w_len)
        win_b  = bsw[n_win // 2:].reshape(-1, w_len)
        win_loss = (win_a - win_b).abs().mean()
    else:
        win_loss = torch.tensor(0.0, device=seq.device)

    return run_loss + 0.5 * win_loss


# MIXED BIT-WIDTH COST LOSS

_INT5_MAX = 15   # INT5 symmetric range [-15, 15]

def bit_cost_loss(module: nn.Module, ent_threshold: float = 3.5) -> Tensor:
    """Per-layer bit cost loss with entropy-conditional INT5 forcing."""
    layer_ents: dict[str, float] = {}
    for name, p in module.named_parameters():
        if '_layer_bit_selector' in dir(p): continue  # skip non-weight params
        if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL: continue
        if any(c in name for c in CONTROL_PATTERNS): continue
        if not p.is_floating_point(): continue
        wf  = p.float().flatten()
        idx = torch.randint(0, wf.numel(), (min(1024, wf.numel()),), device=p.device)
        sample = wf[idx].clamp(-31.0, 31.0)
        cens   = torch.linspace(-31.0, 31.0, _ENT_BINS, device=p.device)
        d_     = ((sample.unsqueeze(1) - cens) / _ENT_SIGMA).pow(2)
        probs  = torch.exp(-0.5 * d_).sum(0)
        probs  = probs / probs.sum().clamp_min(1e-8)
        ent    = float(-(probs * (probs + 1e-8).log()).sum().item())
        layer_ents[name] = ent

    total = torch.tensor(0.0)
    layer_ent_list = list(layer_ents.values())   # ordered by named_parameters() traversal
    sel_idx = 0
    for name, p in module.named_parameters():
        if 'bit_selectors' not in name: continue
        prob_int6 = torch.sigmoid(p)
        base_cost = 5.0 + prob_int6
        paired_ent = layer_ent_list[sel_idx % max(1, len(layer_ent_list))]
        # Extra pressure when THIS layer is low-entropy (safe to use INT5)
        extra_pressure = F.relu(torch.tensor(ent_threshold - paired_ent,
                                             device=p.device)) * prob_int6
        total   = total + base_cost + extra_pressure
        sel_idx += 1
    return total


def apply_mixed_bitwidth_quant(w: Tensor, bit_selector: Tensor,
                                clip_val: float, noise: float) -> Tensor:
    """Soft blend: p6*INT6(w) + (1-p6)*INT5(w). Grad flows through p6."""
    wf = w.float().clamp(-clip_val, clip_val)
    if noise > 0: wf = wf + torch.randn_like(wf) * noise

    p6 = torch.sigmoid(bit_selector).clamp(1e-4, 1.0 - 1e-4)

    # INT6 branch
    if wf.ndim == 2:
        ca6  = torch.quantile(wf.abs(), INT6_CLIP_Q, dim=1, keepdim=True).clamp_min(1e-6)
        sc6  = ca6 / INT6_MAX
        w6   = ((wf / sc6).clamp(-INT6_MAX, INT6_MAX).round() * sc6).to(w.dtype)
    else:
        ca6  = float(torch.quantile(wf.abs().flatten(), INT6_CLIP_Q).item()) or 1e-6
        sc6  = torch.tensor(ca6 / INT6_MAX)
        w6   = ((wf / sc6).clamp(-INT6_MAX, INT6_MAX).round() * sc6).to(w.dtype)

    _INT5_CLIP_Q = 0.999   # tighter than INT6's 99.99% -- prevents outlier scale inflation
    if wf.ndim == 2:
        ca5  = torch.quantile(wf.abs(), _INT5_CLIP_Q, dim=1, keepdim=True).clamp_min(1e-6)
        sc5  = ca5 / _INT5_MAX
        w5   = ((wf / sc5).clamp(-_INT5_MAX, _INT5_MAX).round() * sc5).to(w.dtype)
    else:
        ca5  = float(torch.quantile(wf.abs().flatten(), _INT5_CLIP_Q).item()) or 1e-6
        sc5  = torch.tensor(ca5 / _INT5_MAX)
        w5   = ((wf / sc5).clamp(-_INT5_MAX, _INT5_MAX).round() * sc5).to(w.dtype)

    wdq = p6 * w6 + (1.0 - p6) * w5
    return w + (wdq - w).detach()


# CROSS-LAYER ENTROPY COUPLING

def cross_layer_entropy_loss(module: nn.Module) -> Tensor:
    """Penalise layers more entropic than global EMA pool."""
    layer_samples = []
    for name, p in module.named_parameters():
        if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL: continue
        if any(c in name for c in CONTROL_PATTERNS): continue
        if not p.is_floating_point(): continue
        wf  = p.float().flatten()
        idx = torch.randint(0, wf.numel(), (min(1024, wf.numel()),), device=p.device)
        layer_samples.append(wf[idx])

    if len(layer_samples) < 2: return torch.tensor(0.0)

    global_w = torch.cat(layer_samples)
    centers  = torch.linspace(-31.0, 31.0, _ENT_BINS, device=global_w.device)

    def _ent(wf_: Tensor) -> Tensor:
        d = ((wf_.unsqueeze(1) - centers) / _ENT_SIGMA).pow(2)
        p = torch.exp(-0.5 * d).sum(0)
        p = p / p.sum().clamp_min(1e-8)
        return -(p * (p + 1e-8).log()).sum()

    h_global_live = _ent(global_w)   # live -- not detached, co-evolves with model
    if not hasattr(cross_layer_entropy_loss, '_h_ema'):
        cross_layer_entropy_loss._h_ema = h_global_live.detach()
    cross_layer_entropy_loss._h_ema = (
        0.9 * cross_layer_entropy_loss._h_ema.to(global_w.device) +
        0.1 * h_global_live.detach()
    )
    h_global_target = cross_layer_entropy_loss._h_ema

    total = torch.tensor(0.0, device=global_w.device)
    for ls in layer_samples:
        h_local = _ent(ls)
        total   = total + F.relu(h_local - h_global_target)
    return total


# TEMPORAL WEIGHT CORRELATION LOSS

def temporal_correlation_loss(module: nn.Module) -> Tensor:
    """Multi-lag corr clamped at 0.9: lag1+0.5*lag2+0.25*lag4."""
    def _corr(x: Tensor, y: Tensor) -> Tensor:
        xc = x - x.mean(); yc = y - y.mean()
        return (xc * yc).sum() / (xc.norm() * yc.norm()).clamp_min(1e-8)

    total = torch.tensor(0.0); n_mats = 0
    for name, p in module.named_parameters():
        if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL: continue
        if any(c in name for c in CONTROL_PATTERNS): continue
        if not p.is_floating_point(): continue
        wf = p.float().flatten()
        if wf.numel() < 128: continue
        n  = min(8192, wf.numel()); wf = wf[:n]
        # Lag-1
        c1 = _corr(wf[:-1], wf[1:])
        c2 = _corr(wf[:-2], wf[2:]) if n >= 4 else torch.tensor(0.0)
        c4 = _corr(wf[:-4], wf[4:]) if n >= 8 else torch.tensor(0.0)
        total  = total - (c1.clamp(max=0.9) + 0.5 * c2.clamp(max=0.9) + 0.25 * c4.clamp(max=0.9))
        n_mats += 1
    return total / max(1, n_mats)


# LEARNABLE QUANTISATION CENTERS

class QuantCenters(nn.Module):
    """Trainable quant centers; tanh-constrained to [-31,31]."""
    def __init__(self, n: int = 64):
        super().__init__()
        base = torch.tensor(_POW2_CENTERS[:min(n, len(_POW2_CENTERS))], dtype=torch.float32)
        if n > len(base):
            base = torch.cat([base, torch.linspace(-31.0, 31.0, n - len(base))])
        self.centers = nn.Parameter(base[:n])

    def clamped(self) -> Tensor:
        return torch.tanh(self.centers / 31.0) * 31.0   # constrained [-31,31]

    def attraction_loss(self, module: nn.Module) -> Tensor:
        """Attraction to nearest center + repulsion between centers (anti-collapse)."""
        centers = self.clamped()
        repulsion = torch.pdist(centers.unsqueeze(1)).mean().clamp_min(1e-8)
        attract = None
        for name, p in module.named_parameters():
            if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL: continue
            if any(c in name for c in CONTROL_PATTERNS): continue
            if not p.is_floating_point(): continue
            wf   = p.float().flatten()
            if wf.numel() > 8192:
                wf = wf[torch.randint(0, wf.numel(), (8192,), device=p.device)]
            dist_mat = (wf.unsqueeze(1) - centers.to(wf.device)).abs()
            loss = dist_mat.min(1).values.mean()
            attract = loss if attract is None else attract + loss
        if attract is None: return torch.tensor(0.0)
        return attract - 0.01 * repulsion


class DictAwareReg:
    """EMA centroids over large weights; penalise deviation -> dict stays effective."""
    def __init__(self, n: int = 256, mom: float = 0.995, update_every: int = 50):
        self.n   = n
        self.mom = mom
        self.every = update_every
        self._step = 0
        self.centroids: Tensor | None = None

    @torch.no_grad()
    def _update(self, module: nn.Module):
        chunks = []
        for name, p in module.named_parameters():
            if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL: continue
            if any(c in name for c in CONTROL_PATTERNS): continue
            if not p.is_floating_point(): continue
            chunks.append(p.detach().float().flatten()[:4096])   # sample
        if not chunks: return
        wcat = torch.cat(chunks)
        q    = torch.linspace(0, 1, self.n, device=wcat.device)
        new_c = torch.quantile(wcat, q)
        if self.centroids is None:
            self.centroids = new_c
        else:
            self.centroids = self.mom * self.centroids.to(new_c.device) + (1 - self.mom) * new_c

    def __call__(self, module: nn.Module) -> Tensor:
        self._step += 1
        if self._step % self.every == 0 or self.centroids is None:
            self._update(module)
        if self.centroids is None:
            return torch.tensor(0.0)
        total = None
        for name, p in module.named_parameters():
            if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL: continue
            if any(c in name for c in CONTROL_PATTERNS): continue
            if not p.is_floating_point(): continue
            wf   = p.float().flatten()
            dist = (wf.unsqueeze(1) - self.centroids.to(wf.device)).abs()
            loss = dist.min(1).values.mean()
            total = loss if total is None else total + loss
        return total if total is not None else torch.tensor(0.0)


# TRANSFORMER MODULES

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.weight, eps=self.eps)


def build_rope_cache(seq_len, hd, base, device):
    half  = hd // 2
    freq  = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t     = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    B, T, nh, hd = x.shape; half = hd // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:T].view(1, T, 1, half); s = sin[:T].view(1, T, 1, half)
    return torch.cat([x1*c - x2*s, x1*s + x2*c], dim=-1)


class Attention(nn.Module):
    def __init__(self, hp: Hyperparameters):
        super().__init__()
        self.nh  = hp.num_heads; self.nkv = hp.num_kv_heads
        self.hd  = hp.model_dim // hp.num_heads
        d, kd    = hp.model_dim, hp.num_kv_heads * self.hd
        self.Wq  = QuantLinear(d, d,  bias=False)
        self.Wk  = QuantLinear(d, kd, bias=False)
        self.Wv  = QuantLinear(d, kd, bias=False)
        self.Wo  = QuantLinear(d, d,  bias=False)
        self.q_gain = nn.Parameter(torch.tensor(hp.qk_gain_init))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, _ = x.shape; nh, nkv, hd = self.nh, self.nkv, self.hd
        gain = self.q_gain.abs().clamp_min(0.1)
        q = apply_rope(self.Wq(x).view(B,T,nh, hd), cos, sin) * gain
        k = apply_rope(self.Wk(x).view(B,T,nkv,hd), cos, sin)
        v = self.Wv(x).view(B,T,nkv,hd)
        if nh > nkv:
            r = nh // nkv; k = k.repeat_interleave(r, dim=2); v = v.repeat_interleave(r, dim=2)
        out = F.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True)
        return self.Wo(out.transpose(1,2).contiguous().view(B,T,nh*hd))


class MLP(nn.Module):
    """Full-rank SwiGLU  intermediate = 4 x d = 4096."""
    def __init__(self, hp: Hyperparameters):
        super().__init__()
        d, h       = hp.model_dim, hp.mlp_intermediate
        self.gate  = QuantLinear(d, h, bias=False)
        self.up    = QuantLinear(d, h, bias=False)
        self.down  = QuantLinear(h, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        g = F.relu(self.gate(x)); return self.down(g * g * self.up(x))


class Block(nn.Module):
    def __init__(self, hp: Hyperparameters):
        super().__init__()
        d = hp.model_dim
        self.an = RMSNorm(d); self.attn = Attention(hp)
        self.mn = RMSNorm(d); self.mlp  = MLP(hp)

    def forward(self, x: Tensor, cos, sin) -> tuple[Tensor, Tensor]:
        a = self.attn(self.an(x), cos, sin)
        m = self.mlp(self.mn(x + a))
        return a, m


class NgramHash(nn.Module):
    """3-gram Knuth hash -> embed -> d. Richer than bigram, same params."""
    def __init__(self, hash_size: int, hash_dim: int, model_dim: int):
        super().__init__()
        self.hs    = hash_size
        self.table = nn.Embedding(hash_size, hash_dim)
        self.proj  = QuantLinear(hash_dim, model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        p1   = torch.cat([x[:, :1],  x[:, :-1]],  dim=1)   # prev1
        p2   = torch.cat([x[:, :2],  x[:, :-2]],  dim=1)   # prev2 (pad first 2)
        idx  = (((x.long() ^ (p1.long() << 1) ^ (p2.long() << 2))
                 * 2_654_435_761) % self.hs).clamp(0, self.hs - 1)
        return self.proj(self.table(idx))


#  RECURRENT GPT
class GPT(nn.Module):
    """Recurrent GPT. 1 shared block x 16 recurrences, d=1024, h=4096."""

    def __init__(self, hp: Hyperparameters):
        super().__init__()
        self.hp = hp; self.R = hp.num_recurrences
        d, R    = hp.model_dim, hp.num_recurrences

        self.embed      = nn.Embedding(hp.vocab_size, d)
        self.trigram    = NgramHash(hp.bigram_hash_size, hp.bigram_hash_dim, d)  
        self.depth_emb  = nn.Embedding(R, d)
        self.depth_gate = nn.Parameter(torch.zeros(R, d))

        # SmearGate low-rank: (R,1) x (1,d)
        _sg  = math.log(max(1e-7, (1.0/math.sqrt(R)) / (1.0 - 1.0/math.sqrt(R))))
        self.attn_smear_base = nn.Parameter(torch.full((R, 1), _sg))
        self.attn_smear_proj = nn.Parameter(torch.ones(1, d))
        self.mlp_smear_base  = nn.Parameter(torch.full((R, 1), _sg))
        self.mlp_smear_proj  = nn.Parameter(torch.ones(1, d))

        self.router    = nn.Linear(d, 1, bias=False)
        self.step_bias = nn.Parameter(torch.zeros(R))
        self.mem_decay = nn.Parameter(
            torch.tensor(math.log(hp.mem_decay_init / (1.0 - hp.mem_decay_init))))
        self.mem_scale  = nn.Parameter(torch.tensor(0.1))
        self.hist_logit = nn.Parameter(torch.tensor(-2.944))
        self.imp_w      = nn.Parameter(torch.zeros(d))

        # calibration + prior
        self.logit_temp = nn.Parameter(torch.ones(1))
        self.freq_bias  = nn.Parameter(torch.zeros(hp.vocab_size))

        self.freq_embed = nn.Embedding(hp.vocab_size, 32)
        self.freq_proj  = nn.Linear(32, d, bias=False)
        self.freq_gate_w = nn.Linear(32, d, bias=False)  # maps freq embed -> per-dim gate
        self.ent_router  = nn.Linear(d, 1, bias=True)    # h -> scalar entropy estimate

        self.step_continue = nn.Parameter(torch.ones(R) * 2.0)

        
        _n_large_fixed = 7   # Wq, Wk, Wv, Wo, gate, up, down -- all large (d*d or d*h)
        self.bit_selectors = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(_n_large_fixed)]
        )

        self.block    = Block(hp)
        self.out_norm = RMSNorm(d)
        self.skip_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ~ 0.12 at init

        if not hp.tie_embeddings:
            self.head = QuantLinear(d, hp.vocab_size, bias=False)

        hd = d // hp.num_heads
        cos, sin = build_rope_cache(4096, hd, hp.rope_base, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self._init_weights()

    def _init_weights(self):
        std = self.hp.tied_embed_init_std if self.hp.tie_embeddings else 0.02
        nn.init.orthogonal_(self.embed.weight)
        self.embed.weight.data.mul_(
            std / self.embed.weight.data.norm(dim=1).mean().clamp_min(1e-8))
        nn.init.normal_(self.depth_emb.weight, std=0.005)
        nn.init.zeros_(self.depth_gate)
        nn.init.uniform_(self.step_bias, -0.1, 0.1)
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.zeros_(self.imp_w); nn.init.zeros_(self.freq_bias)
        nn.init.zeros_(self.freq_gate_w.weight)
        nn.init.normal_(self.ent_router.weight, std=0.01)
        nn.init.zeros_(self.ent_router.bias)
        nn.init.normal_(self.freq_embed.weight, std=0.01)
        nn.init.zeros_(self.freq_proj.weight)
        if not self.hp.tie_embeddings: nn.init.zeros_(self.head.weight)
        bstd = 0.02 / math.sqrt(2 * self.R)
        for name, p in self.block.named_parameters():
            if p.ndim >= 2 and "weight" in name:
                if "down.weight" in name:
                    nn.init.zeros_(p)
                else:
                    nn.init.orthogonal_(p)
                    p.data.mul_(bstd * max(p.shape) / max(p.data.norm().item(), 1e-8))

    def _head_weight(self) -> Tensor:
        return self.embed.weight if self.hp.tie_embeddings else self.head.weight

    def _step(self, h, depth_sig, cos, sin, a_sg, m_sg, step_b) -> Tensor:
        h_in        = h + depth_sig
        a_out, m_out = self.block(h_in, cos, sin)
        h_a = torch.lerp(h_in, h_in + a_out, a_sg)
        h_m = torch.lerp(h_a,  h_a  + m_out, m_sg)
        gate = torch.sigmoid(self.router(h) + step_b)
        return h * (1.0 - gate) + h_m * gate

    def forward(self, x: Tensor, y: Tensor | None = None,
                entropy_reg: bool = False) -> Tensor | tuple:
        B, T  = x.shape
        cos   = self.rope_cos[:T].to(x.device)
        sin   = self.rope_sin[:T].to(x.device)
        freq_e = self.freq_embed(x)                          # (B,T,32)
        h      = self.embed(x) + self.trigram(x)
        h      = h + self.freq_proj(freq_e)                  # passive freq signal
        freq_gate = torch.sigmoid(self.freq_gate_w(freq_e))  # (B,T,d) in (0,1)
        h         = h * (1.0 + 0.1 * freq_gate)             # gentle multiplicative gate
        decay = torch.sigmoid(self.mem_decay)
        msc   = self.mem_scale
        mem   = torch.zeros_like(h)
        damp  = self.hp.mem_damping

        gd       = self.depth_emb.weight * torch.sigmoid(self.depth_gate)
        h_skips  = []     # U-Net multi-scale captures
        hc       = torch.sigmoid(self.hist_logit)
        h_prev   = h.detach()
        a_sgs    = torch.sigmoid(self.attn_smear_base * self.attn_smear_proj)
        m_sgs    = torch.sigmoid(self.mlp_smear_base  * self.mlp_smear_proj)

        cont_probs  = torch.sigmoid(self.step_continue)   # (R,) global step gates
        ent_score   = torch.sigmoid(self.ent_router(h.detach()))  # (B,T,1) no grad through h
        sparse_loss = torch.zeros(1, device=x.device)
        ent_route_loss = torch.zeros(1, device=x.device)  # sparsity on entropy router

        for step in range(self.R):
            drop_p = self.hp.depth_dropout_base_p * (step / max(1, self.R - 1))
            if self.training and drop_p > 0 and torch.rand(1).item() < drop_p:
                h_prev = h.detach(); continue

            cp = cont_probs[step]                             # global step gate
            cp_eff = (cp * (0.5 + ent_score.squeeze(-1))).clamp(0, 1)  # (B,T) local gate
            if not self.training and cp.item() < self.hp.dyn_rec_threshold:
                continue

            h_in = h + msc * mem
            if self.training and self.hp.perturb_scale > 0:
                h_in = h_in + torch.randn_like(h_in) * self.hp.perturb_scale * (step / self.R)

            ds = gd[step].view(1, 1, -1)
            ag = a_sgs[step].view(1, 1, -1)
            mg = m_sgs[step].view(1, 1, -1)
            sb = self.step_bias[step].view(1, 1, 1)

            if self.training:
                h_new = grad_ckpt(self._step, h_in, ds, cos, sin, ag, mg, sb, use_reentrant=False)
            else:
                h_new = self._step(h_in, ds, cos, sin, ag, mg, sb)

            h      = cp_eff.unsqueeze(-1) * h_new + (1.0 - cp_eff.unsqueeze(-1)) * h.detach()
            h      = h + hc * h_prev
            h_prev = h.detach()
            if step == self.R // 4 or step == self.R // 2 or step == 3 * self.R // 4:
                h_skips.append(h.detach())   # multi-scale U-Net captures

            mem      = damp * (decay * mem + (1.0 - decay) * h)
            mem_norm = mem.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            mem      = mem / mem_norm.clamp_min(1.0)

            if self.training:
                sparse_loss = sparse_loss - (
                    cp * (cp + 1e-6).log() + (1.0 - cp) * (1.0 - cp + 1e-6).log())
                es = ent_score.squeeze(-1).mean()
                ent_route_loss = ent_route_loss - (
                    es * (es + 1e-6).log() + (1.0 - es) * (1.0 - es + 1e-6).log())

        if h_skips:
            h_blend = torch.stack(h_skips).mean(0)  # average across captured depths
            h = h + torch.sigmoid(self.skip_gate) * h_blend
        importance = torch.sigmoid((h * self.imp_w).mean(dim=-1, keepdim=True))
        h = self.out_norm(h * importance)

        logits = F.linear(h, self._head_weight().to(h.dtype))
        logits = logits + self.freq_bias.to(logits.dtype)
        logits = logits / self.logit_temp.clamp(0.5, 2.0)

        if self.hp.logit_softcap > 0:
            logits = torch.tanh(logits / self.hp.logit_softcap) * self.hp.logit_softcap

        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            if entropy_reg and self.hp.entropy_reg_weight > 0:
                pr  = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)
                ent = -(pr * (pr + 1e-8).log()).sum(-1).mean()
                loss = loss - self.hp.entropy_reg_weight * ent
            return loss, sparse_loss.squeeze(), ent_route_loss.squeeze(), logits
        return logits


def zipf_alignment_loss(logits_sample: Tensor) -> Tensor:
    """KL divergence from model logit distribution to Zipf law."""
    V = logits_sample.size(-1)
    probs = F.softmax(logits_sample.float(), dim=-1).mean(0)  # (V,) mean over batch
    ranks  = torch.arange(1, V + 1, device=logits_sample.device, dtype=torch.float32)
    zipf_p = (1.0 / ranks)
    probs_sorted = probs.sort(descending=True).values.clamp_min(1e-8)
    zipf_p       = (zipf_p / zipf_p.sum()).clamp_min(1e-8)
    return (probs_sorted * (probs_sorted / zipf_p).log()).sum()

# SWA

class SWAState:
    def __init__(self, m: nn.Module):
        self.p = {n: p.data.clone().cpu() for n, p in m.named_parameters()}
        self.n = 0

    def update(self, m: nn.Module):
        self.n += 1; a = 1.0 / self.n
        for n, p in m.named_parameters():
            self.p[n].lerp_(p.data.cpu(), a)

    def apply(self, m: nn.Module):
        for n, p in m.named_parameters():
            p.data.copy_(self.p[n].to(p.device))


def restore_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(c in name for c in CONTROL_PATTERNS)) \
               and p.dtype != torch.float32:
                p.data = p.data.float()


# MAIN

def main() -> None:
    args = Hyperparameters()
    assert torch.cuda.is_available(), "CUDA required"
    if dist.is_available() and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank(); ws = dist.get_world_size()
    else:
        rank = 0; ws = 1
    master = rank == 0
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    model     = GPT(args).to(device)
    raw_model = model
    restore_fp32(model)
    QuantLinear.qat_clip  = args.qat_clip_val
    QuantLinear.qat_noise = args.qat_noise_scale

    if master:
        tp = sum(p.numel() for p in model.parameters())
        print(f"GPT [{'zstd' if _HAVE_ZSTD else 'zlib fallback'}]")
        print(f"  unique params   : {tp:,} ({tp/1e6:.2f}M)")
        print(f"  effective @R={args.num_recurrences}  : {tp*args.num_recurrences:,} ({tp*args.num_recurrences/1e6:.0f}M)")
        print(f"  mlp_intermediate: {args.mlp_intermediate}  (full-rank 4xd)")
        print(f"  seqlen={args.train_seq_len} | stride=1/{args.eval_stride_div} | swa_frac={args.swa_start_frac} | damping={args.mem_damping}")

    learned_cents = QuantCenters(n=args.learned_centers_n).to(device)

    if ws > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    embed_p = [raw_model.embed.weight]
    sc_p, mx_p = [], []
    for name, p in raw_model.named_parameters():
        if p is raw_model.embed.weight: continue
        (mx_p if p.ndim >= 2 and p.numel() > INT_KEEP_FLOAT_MAX_NUMEL else sc_p).append(p)
    if not args.tie_embeddings: sc_p.append(raw_model.head.weight)

    elr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_adam = torch.optim.AdamW(
        [{"params": embed_p, "lr": elr,           "base_lr": elr,           "weight_decay": args.muon_wd},
         {"params": sc_p,    "lr": args.scalar_lr, "base_lr": args.scalar_lr, "weight_decay": 0.01}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon([{"params": mx_p}], lr=args.matrix_lr,
                    momentum=args.muon_momentum_warmup_start,
                    backend_steps=args.muon_backend_steps,
                    weight_decay=args.muon_wd)
    opts = [opt_adam, opt_muon]

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    sp = spm.SentencePieceProcessor(); sp.Load(args.tokenizer_path)
    bbl, hsl, ibl = build_sentencepiece_luts(sp, args.vocab_size, device)
    loader = DistributedTokenLoader(args.train_files, rank, ws, device)


    dict_reg = DictAwareReg(n=args.dict_reg_centroids, mom=0.995, update_every=50)
    opt_adam.add_param_group({
        "params": list(learned_cents.parameters()),
        "lr": args.scalar_lr, "base_lr": args.scalar_lr
    })

    la, lc = 0.0, 0; t0 = time.time()
    token_freq = torch.ones(args.vocab_size, device=device)  # EMA token counts for TFAL
    swa_start = int(args.iterations * (1.0 - args.swa_start_frac))  # start at 60% done
    swa: SWAState | None = None                    # fix: must be initialised before loop
    if master: print(f"\nTraining {args.iterations} iters | cap {args.max_wallclock_seconds}s")

    for step in range(args.iterations):
        elapsed = time.time() - t0
        if args.max_wallclock_seconds > 0 and elapsed >= args.max_wallclock_seconds:
            if master: print(f"Wallclock cap @ step {step}."); break

        if step == args.qat_start_step:
            QuantLinear.qat_enabled = True
            if master:
                print(f"step {step}: INT6 QAT ON (clip=+-{args.qat_clip_val}, noise={args.qat_noise_scale})")
            if args.mixed_bitwidth and hasattr(raw_model, 'bit_selectors'):
                large_qat = [
                    m for m in raw_model.modules()
                    if isinstance(m, QuantLinear) and m.weight.numel() > INT_KEEP_FLOAT_MAX_NUMEL
                ]
                for i, m in enumerate(large_qat):
                    sel_idx = i % len(raw_model.bit_selectors)
                    m._layer_bit_selector = raw_model.bit_selectors[sel_idx]
                QuantLinear._bit_selector = None   # per-layer attrs take priority
                if master: print(f"  Per-layer mixed bitwidth: {len(large_qat)} matrices, {len(raw_model.bit_selectors)} selectors")
            with torch.no_grad():
                _dummy = torch.zeros(1, 1, device=device, dtype=torch.int64)
                raw_model.eval()
                try: raw_model(_dummy)
                except Exception: pass
                raw_model.train()
            lqs_params = list(QuantLinear._scale_registry.values())
            if lqs_params:
                opt_adam.add_param_group({
                    "params": lqs_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr
                })
                if master: print(f"  Registered {len(lqs_params)} learned quant scales")

        model.train()

        if step < args.warmup_steps:
            lrs = (step + 1) / args.warmup_steps
        elif step < args.iterations - args.warmdown_iters:
            lrs = 1.0
        else:
            lrs = max(0.0, (args.iterations - step) / args.warmdown_iters)

        for pg in opt_adam.param_groups: pg["lr"] = pg["base_lr"] * lrs
        for pg in opt_muon.param_groups: pg["lr"] = args.matrix_lr * lrs
        mm = args.muon_momentum_warmup_start + (
            args.muon_momentum - args.muon_momentum_warmup_start
        ) * min(1.0, step / max(1, args.muon_momentum_warmup_steps))
        for pg in opt_muon.param_groups: pg["momentum"] = mm

        x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, 1)
        for o in opts: o.zero_grad(set_to_none=True)

        with torch.no_grad():
            cnts = torch.bincount(y.view(-1), minlength=args.vocab_size).float()
            token_freq.mul_(0.99).add_(0.01 * cnts)
            if ws > 1: dist.all_reduce(token_freq, op=dist.ReduceOp.AVG)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, sparse_loss, ent_route_loss, raw_logits = model(
                x, y, entropy_reg=QuantLinear.qat_enabled)
        tok_w = (token_freq.clamp_min(1e-6) ** -0.5)
        tok_w = (tok_w / tok_w.mean()).clamp(max=10.0)
        flat_y      = y.view(-1)
        ce_per_tok  = F.cross_entropy(raw_logits.float().view(-1, raw_logits.size(-1)),
                                      flat_y, reduction="none")
        loss = (ce_per_tok * tok_w[flat_y]).mean()

        total_loss = (loss
                      + args.dyn_rec_sparse_w * sparse_loss
                      + args.ent_route_w * ent_route_loss)

        ew = args.ent_weight_qat if QuantLinear.qat_enabled else args.ent_weight
        if ew > 0:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ent_l = model_entropy_loss(raw_model)
            total_loss = total_loss + ew * ent_l

        if QuantLinear.qat_enabled and step % 50 == 0 and args.zipf_align_w > 0:
            total_loss = total_loss + args.zipf_align_w * zipf_alignment_loss(
                raw_logits.detach()[:, ::8].reshape(-1, raw_logits.size(-1))[:512])
        if QuantLinear.qat_enabled:
            if args.pow2_weight > 0:
                total_loss = total_loss + args.pow2_weight * pow2_cluster_loss(raw_model, device)
            if args.dict_reg_weight > 0:
                total_loss = total_loss + args.dict_reg_weight * dict_reg(raw_model)
            bp_w = args.ent_weight_qat * 0.5
            if bp_w > 0:
                total_loss = total_loss + bp_w * bitplane_entropy_loss(raw_model)
            if args.dict_cotrain_w > 0 and (step % args.dict_cotrain_every == 0):
                total_loss = total_loss + args.dict_cotrain_w * dict_cotrain_loss(raw_model)
            if args.mixed_bitwidth and args.bit_cost_w > 0:
                total_loss = total_loss + args.bit_cost_w * bit_cost_loss(
                    raw_model, ent_threshold=args.bit_ent_threshold)
            if args.cross_ent_w > 0:
                total_loss = total_loss + args.cross_ent_w * cross_layer_entropy_loss(raw_model)
            if args.temporal_corr_w > 0:
                total_loss = total_loss + args.temporal_corr_w * temporal_correlation_loss(raw_model)
            if args.learned_centers_w > 0:
                total_loss = total_loss + args.learned_centers_w * learned_cents.attraction_loss(raw_model)

        total_loss.backward()
        if ws > 1 and learned_cents.centers.grad is not None:
            dist.all_reduce(learned_cents.centers.grad, op=dist.ReduceOp.AVG)
        if args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip_norm)
        for o in opts: o.step()
        restore_fp32(raw_model)
        if QATLinear.qat_enabled and step % 500 == 0:
            with torch.no_grad():
                for p in raw_model.parameters():
                    if p.ndim == 2 and p.numel() > INT_KEEP_FLOAT_MAX_NUMEL:
                        threshold = (p.abs().max(dim=1, keepdim=True).values * 0.03).clamp_min(1e-6)
                        p.data *= (p.abs() >= threshold).float()
        if QuantLinear.qat_enabled and args.mag_prune_frac > 0 and (step % 100 == 0):
            with torch.no_grad():
                for p in raw_model.parameters():
                    if p.numel() <= INT_KEEP_FLOAT_MAX_NUMEL or not p.is_floating_point(): continue
                    thresh = torch.quantile(p.abs().flatten(), args.mag_prune_frac)
                    p.data[p.abs() < thresh] = 0.0

        if step >= swa_start and (step - swa_start) % args.swa_every == 0:
            if swa is None: swa = SWAState(raw_model)
            swa.update(raw_model)
            # Also EMA-average learned_cents (outside DDP scope)
            if swa.n > 1:
                for p_swa, p_cur in zip([learned_cents.centers], [learned_cents.centers.data]):
                    p_swa.data.lerp_(p_cur.cpu(), 1.0 / swa.n)

        la += loss.item(); lc += 1
        if master and (step + 1) % args.train_log_every == 0:
            print(f"step {step+1:6d} | loss {la/lc:.4f} | lrx{lrs:.3f} | {elapsed:.0f}s | "
                  f"qat={'on' if QuantLinear.qat_enabled else 'off'} | swa={'on' if swa else 'off'}")
            la = 0.0; lc = 0

        if args.val_loss_every > 0 and (step + 1) % args.val_loss_every == 0:
            vl, vb = eval_val(args, model, rank, ws, device, 1, val_tokens, bbl, hsl, ibl)
            if master:
                print(f"step {step+1:6d} | val_loss {vl:.4f} | val_bpb {vb:.4f} | {elapsed:.0f}s")

    if swa is not None:
        if master: print("\nApplying SWA weights...")
        swa.apply(raw_model)

    vl, vb = eval_val(args, model, rank, ws, device, 1, val_tokens, bbl, hsl, ibl)
    if master:
        print(f"\nFinal | val_loss {vl:.4f} | val_bpb {vb:.4f} | "
              f"time {time.time()-t0:.0f}s | run_id {args.run_id}")

    # -- INT6 true-pack + zstd-dict compression -----------------------------
    if master:
        sd = {k: v.detach() for k, v in raw_model.state_dict().items()}
        ls = extract_learned_scales(raw_model)
        if master and ls:
            print(f"Exporting with {len(ls)} learned quant scales (train/export aligned)")
        # Build bit_selector_map: name -> selector param for INT5 export path
        bsm: dict = {}
        if hasattr(raw_model, 'bit_selectors'):
            large_qat_names = [
                n for n, p in raw_model.named_parameters()
                if p.numel() > INT_KEEP_FLOAT_MAX_NUMEL
                and not any(c in n for c in CONTROL_PATTERNS)
                and p.is_floating_point()
                and 'bit_selector' not in n
            ]
            for i, n in enumerate(large_qat_names):
                sel_idx = i % len(raw_model.bit_selectors)
                bsm[n]  = raw_model.bit_selectors[sel_idx].detach()
        obj, stats = quantize_state_dict(sd, args.zstd_dict_size,
                                          learned_scales=ls,
                                          weight_reorder=args.weight_reorder,
                                          bit_selector_map=bsm)
        rec        = dequantize_state_dict(obj)
        raw_model.load_state_dict(rec, strict=True)

        vl_rt, vb_rt = eval_val(args, raw_model, 0, 1, device, 1, val_tokens, bbl, hsl, ibl)
        print(f"\nint6_truepack_roundtrip | val_loss {vl_rt:.4f} | val_bpb {vb_rt:.4f}")

        zdict     = obj.get("zd")
        buf       = io.BytesIO(); torch.save(obj, buf)
        mc        = _zstd_compress(buf.getvalue(), zdict)
        sc_bytes  = _zstd_compress(open(__file__, "rb").read())
        zd_bytes  = len(zdict) if zdict else 0
        tot       = len(mc) + len(sc_bytes) + zd_bytes
        method    = ("zstd-22+dict" if (zdict and _HAVE_ZSTD)
                     else ("zstd-22" if _HAVE_ZSTD else "zlib-9"))
        print(f"\nArtifact ({method}):")
        print(f"  model compressed : {len(mc)/1e6:.3f} MB")
        print(f"  script compressed: {len(sc_bytes)/1e3:.1f} KB")
        print(f"  zstd dict        : {zd_bytes/1e3:.1f} KB")
        print(f"  TOTAL            : {tot/1e6:.3f} MB  "
              f"({'PASS OK' if tot < 16_000_000 else 'FAIL X'})")
        print(f"\nParams : {stats['param_count']:,} unique x {args.num_recurrences} = "
              f"{stats['param_count']*args.num_recurrences:,} effective")
        raw_mb = stats["raw_float_bytes"] / 1e6
        pack_mb = stats["packed_bytes"] / 1e6
        print(f"INT6 true-pack: {raw_mb:.2f} MB -> {pack_mb:.2f} MB "
              f"({(1-pack_mb/max(0.01,raw_mb))*100:.1f}% saved vs float)")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()