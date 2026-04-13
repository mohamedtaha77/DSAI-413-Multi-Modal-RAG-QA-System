# ingestion/embedder.py
# Loads ColQwen2.5, processes page images, generates multi-vector embeddings,
# and upserts them into the vector store with page-level metadata.

import base64
import gc
import io
import os
import torch
os.environ["SAFETENSORS_FAST_GPU"] = "1"  # map weights directly to GPU, bypass CPU mmap
import compat  # noqa: F401 — patches transformers qwen2_vl PEFT bug
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import BitsAndBytesConfig
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorConfig,
    MultiVectorComparator,
    PointStruct,
    VectorParams,
)
from config import get_settings
from ingestion.pdf_converter import pdf_to_images


def _get_qdrant_client(settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=120)
    return QdrantClient(path="./qdrant_db")


def _image_to_b64(img: Image.Image) -> str:
    # JPEG at quality 85 is ~10x smaller than PNG — keeps payload uploadable
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ingest_pdf(pdf_path: str) -> None:
    """
    Embed all pages of a PDF with ColQwen2.5 and store them in Qdrant.
    """
    settings = get_settings()
    doc_name = os.path.basename(pdf_path)

    # 4-bit quantization: ~1.5GB VRAM, fits on RTX 3050 Ti. Falls back to bfloat16 on CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = ColQwen2_5.from_pretrained(
            settings.colpali_model,
            quantization_config=quant_config,
            device_map="cuda",
        ).eval()
    else:
        model = ColQwen2_5.from_pretrained(
            settings.colpali_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device).eval()
    processor = ColQwen2_5_Processor.from_pretrained(
        settings.colpali_model, max_num_visual_tokens=768,
    )

    # Set up Qdrant collection
    client = _get_qdrant_client(settings)
    vector_size = 128  # ColQwen2.5 output dim

    if not client.collection_exists(settings.collection_name):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                "colpali": VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                )
            },
        )

    # Convert PDF pages to images
    pages = pdf_to_images(pdf_path)

    # Embed one page at a time to stay within CPU RAM limits
    batch_size = 1
    points: list[PointStruct] = []

    # Determine starting ID (avoid collisions if collection already has points)
    existing = client.count(collection_name=settings.collection_name, exact=True)
    id_offset = existing.count

    total = len(pages)
    for i in range(0, total, batch_size):
        batch = pages[i : i + batch_size]
        images = [img for _, img in batch]

        with torch.no_grad():
            input_device = next(model.parameters()).device
            batch_inputs = processor.process_images(images).to(input_device)
            embeddings = model(**batch_inputs)  # (B, seq_len, 128)

        batch_points: list[PointStruct] = []
        for j, (page_num, img) in enumerate(batch):
            vecs = embeddings[j].cpu().float().tolist()  # list of 128-dim lists
            point_id = id_offset + i + j
            batch_points.append(
                PointStruct(
                    id=point_id,
                    vector={"colpali": vecs},
                    payload={
                        "doc_name": doc_name,
                        "page_num": page_num,
                        "image_b64": _image_to_b64(img),
                    },
                )
            )
            points.append(batch_points[-1])

        client.upsert(collection_name=settings.collection_name, points=batch_points)
        del batch_inputs, embeddings
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  Page {i + len(batch)}/{total} ingested", flush=True)

    print(f"Ingested {len(points)} pages from '{doc_name}'")
