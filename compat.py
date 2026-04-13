# compat.py
# Patches for transformers 5.x + colpali_engine + Windows mmap issue.

import json
import re
import struct
import torch
from transformers.integrations import peft as _peft

# ---------------------------------------------------------------------------
# Fix 1: qwen2_5_vl incorrectly listed in _MODEL_TO_CONVERSION_PATTERN which
# triggers a broken MoE-PEFT conversion corrupting target_modules into chars.
# ---------------------------------------------------------------------------
_peft._MODEL_TO_CONVERSION_PATTERN.pop("qwen2_5_vl", None)

# ---------------------------------------------------------------------------
# Fix 2: safetensors.safe_open always memory-maps the file to CPU virtual
# memory.  On Windows this requires paging file space equal to the whole model
# (~6 GB) which triggers OSError 1455 on memory-starved systems.
#
# Replacement: read each tensor with ordinary file I/O (seek + read) — no mmap.
# Peak RAM = one tensor at a time (a few hundred MB at most).
#
# Additionally, checkpoint keys are remapped inside the reader so that both
# base-model and LoRA-adapter loading see the names that transformers 5.x
# expects.  The colqwen2.5-base checkpoint uses old Qwen2.5-VL naming
# (model.layers, model.embed_tokens, model.norm) but the current code has
# self.language_model.*. ColQwen2_5._checkpoint_conversion_mapping only covers
# model.layers and the global qwen2_5_vl mapping produces model.language_model
# (designed for ForConditionalGeneration, not the bare VLModel that ColQwen2_5
# inherits).  And the adapter loading path doesn't apply _checkpoint_conversion
# at all.  Doing the rename at the file-reader level fixes all paths at once.
# ---------------------------------------------------------------------------

_SAFETENSORS_DTYPE = {
    "F64":  torch.float64,
    "F32":  torch.float32,
    "F16":  torch.float16,
    "BF16": torch.bfloat16,
    "I64":  torch.int64,
    "I32":  torch.int32,
    "I16":  torch.int16,
    "I8":   torch.int8,
    "U8":   torch.uint8,
    "BOOL": torch.bool,
}

# Applied in order to every key read from a safetensors file.
_KEY_REMAP = [
    # 1. Strip PEFT base_model.model. wrapper (adapter checkpoints)
    (re.compile(r"^base_model\.model\."), ""),
    # 2. Old Qwen2.5-VL naming → new (model.X → language_model.X)
    (re.compile(r"^model\."), "language_model."),
]


def _remap_key(key: str) -> str:
    for pattern, replacement in _KEY_REMAP:
        key = pattern.sub(replacement, key)
    return key


class _SafeSlice:
    """Lazy tensor slice — materialised only when indexed (mirrors PySafeSlice)."""

    def __init__(self, owner: "_FileIOSafetensors", name: str):
        self._owner = owner
        self._name = name

    def get_shape(self):
        return list(self._owner._tensor_info(self._name)["shape"])

    def get_dtype(self):
        return self._owner._tensor_info(self._name)["dtype"]

    def __getitem__(self, idx):
        tensor = self._owner.get_tensor(self._name)
        if idx == () or idx is Ellipsis or idx == slice(None):
            return tensor
        return tensor[idx]


class _FileIOSafetensors:
    """Drop-in replacement for safetensors.safe_open using plain file I/O."""

    def __init__(self, path: str, framework: str, device: str = "cpu"):
        self._f = None
        self._path = path
        self._device = device
        self._f = open(path, "rb")

        header_size = struct.unpack("<Q", self._f.read(8))[0]
        header = json.loads(self._f.read(header_size).decode("utf-8"))

        self._data_start = 8 + header_size
        self._meta = header.pop("__metadata__", {})
        self._tensors = header  # {original_name: {dtype, shape, data_offsets}}

        # Build key remapping: remapped_name → original_name
        self._to_orig = {}
        for orig in self._tensors:
            remapped = _remap_key(orig)
            self._to_orig[remapped] = orig

    def _tensor_info(self, name: str) -> dict:
        return self._tensors[self._to_orig.get(name, name)]

    # ---- context manager ----
    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._f is not None:
            self._f.close()

    def __del__(self):
        try:
            if getattr(self, "_f", None) is not None:
                self._f.close()
        except Exception:
            pass

    # ---- safetensors API ----
    def keys(self):
        return list(self._to_orig.keys())

    def metadata(self):
        return self._meta

    def get_slice(self, name: str) -> _SafeSlice:
        return _SafeSlice(self, name)

    def get_tensor(self, name: str) -> torch.Tensor:
        info = self._tensor_info(name)
        dtype_str = info["dtype"]
        shape = info["shape"]
        begin, end = info["data_offsets"]
        size = end - begin

        # Try CPU allocation first.  On memory-starved Windows systems the
        # Python allocator may fail for mid-size tensors after heap fragmentation,
        # so fall back to streaming directly to GPU in 64 MiB chunks.
        try:
            buf = bytearray(size)
        except MemoryError:
            if torch.cuda.is_available():
                return self._stream_to_gpu(dtype_str, shape, begin, end)
            raise

        self._f.seek(self._data_start + begin)
        self._f.readinto(buf)

        if dtype_str == "BF16":
            t = torch.frombuffer(buf, dtype=torch.int16).view(torch.bfloat16)
        else:
            t = torch.frombuffer(buf, dtype=_SAFETENSORS_DTYPE[dtype_str])

        if shape:
            t = t.reshape(shape)
        return t

    def _stream_to_gpu(self, dtype_str, shape, begin, end):
        dtype = torch.bfloat16 if dtype_str == "BF16" else _SAFETENSORS_DTYPE[dtype_str]
        elem = dtype.itemsize if hasattr(dtype, "itemsize") else torch.tensor([], dtype=dtype).element_size()
        total_bytes = end - begin
        total_elems = total_bytes // elem

        result = torch.empty(total_elems, dtype=dtype, device="cuda")
        chunk_bytes = 2 * 1024 * 1024  # 2 MiB — keep small for RAM-starved systems
        self._f.seek(self._data_start + begin)
        pos = 0

        while pos < total_elems:
            n_bytes = min(chunk_bytes, (total_elems - pos) * elem)
            n_elems = n_bytes // elem
            buf = bytearray(n_bytes)
            self._f.readinto(buf)

            if dtype_str == "BF16":
                chunk = torch.frombuffer(buf, dtype=torch.int16).view(torch.bfloat16)
            else:
                chunk = torch.frombuffer(buf, dtype=dtype)

            result[pos:pos + n_elems] = chunk
            pos += n_elems
            del buf, chunk

        if shape:
            result = result.reshape(shape)
        return result


import transformers.modeling_utils as _mu
_mu.safe_open = _FileIOSafetensors

# ---------------------------------------------------------------------------
# Fix 3: caching_allocator_warmup pre-allocates the full adapter size on GPU
# before loading weights.  On a 4 GB GPU already holding the quantized base
# (~2.6 GB), this OOMs.  No-op it — the adapter is only ~100 MB of real data.
# ---------------------------------------------------------------------------
if hasattr(_mu, "caching_allocator_warmup"):
    _mu.caching_allocator_warmup = lambda *a, **k: None
