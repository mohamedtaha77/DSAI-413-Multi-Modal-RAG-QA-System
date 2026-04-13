# retrieval/retriever.py
# Encodes the user query with ColQwen2.5, runs MaxSim search against the vector store,
# and returns top-k page images with source metadata (doc name, page number).

import base64
import gc
import io
import os
# Shrink cuBLAS scratch workspace so small reductions (proj.norm) don't fail
# for VRAM-contiguity reasons after the 4-bit base has been loaded.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
import torch
import compat  # noqa: F401 — installs safe_open + allocator_warmup patches
from dataclasses import dataclass
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import BitsAndBytesConfig
from qdrant_client import QdrantClient
from config import get_settings


@dataclass
class RetrievedPage:
    doc_name: str
    page_num: int
    score: float
    image: Image.Image


def _get_qdrant_client(settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60)
    return QdrantClient(path="./qdrant_db")


def _b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


class Retriever:
    def __init__(self):
        settings = get_settings()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # 4-bit: model fits in ~1.5 GB VRAM; CPU peak is one layer at a time
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = ColQwen2_5.from_pretrained(
                settings.colpali_model,
                quantization_config=quant_config,
                device_map="cuda",
            ).eval()
        else:
            self.model = ColQwen2_5.from_pretrained(
                settings.colpali_model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to(device).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(settings.colpali_model)
        self.device = device
        self.client = _get_qdrant_client(settings)
        self.settings = settings

        # Release residual load-time CPU buffers (bytearrays from compat.py)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # Warmup: force CUDA kernel JIT compilation now — before real queries —
        # while CPU RAM is relatively freshest. NVRTC/PTXAS allocates on CPU,
        # and compiling on a starved system is what raises "fatal: Memory
        # allocation failure" during the first user query.
        try:
            with torch.no_grad():
                warm = self.processor.process_queries(["warmup"]).to(
                    next(self.model.parameters()).device
                )
                _ = self.model(**warm)
            del warm, _
        except Exception as e:
            print(f"[retriever] warmup pass skipped: {e}", flush=True)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    def retrieve(self, query: str) -> list[RetrievedPage]:
        if self.device == "cuda":
            torch.cuda.empty_cache()
        with torch.no_grad():
            input_device = next(self.model.parameters()).device
            inputs = self.processor.process_queries([query]).to(input_device)
            query_vecs = self.model(**inputs)  # (1, seq_len, 128)

        query_vecs_list = query_vecs[0].cpu().float().tolist()

        results = self.client.query_points(
            collection_name=self.settings.collection_name,
            query=query_vecs_list,
            using="colpali",
            limit=self.settings.top_k,
            with_payload=True,
        )

        pages: list[RetrievedPage] = []
        for hit in results.points:
            payload = hit.payload
            pages.append(
                RetrievedPage(
                    doc_name=payload["doc_name"],
                    page_num=payload["page_num"],
                    score=hit.score,
                    image=_b64_to_image(payload["image_b64"]),
                )
            )
        return pages
