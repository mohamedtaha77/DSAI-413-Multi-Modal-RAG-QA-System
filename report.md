# Multi-Modal RAG-Based QA System -- Technical Report

**DSAI 413 -- Assignment 1 | SPR 26 | Mohammed Taha | 202201788**

## 1. Architecture Overview

The system implements a multi-modal Retrieval-Augmented Generation pipeline that answers questions about complex PDF documents containing text, tables, charts, and images. Rather than extracting each modality separately, we adopt a **vision-first** approach: every PDF page is rendered as an image and embedded holistically by a vision-language model, preserving spatial layout and cross-modal relationships that text extraction would lose.

**Pipeline stages:**

```
PDF --> Page Images --> ColQwen2.5 Embeddings --> Qdrant (MaxSim) --> Top-K Pages --> LLM Answer
         (pymupdf)       (multi-vector 128d)      (cloud vector DB)                  (Llama 4 Maverick)
```

| Component | Technology | Role |
|-----------|-----------|------|
| Document parsing | PyMuPDF | Renders each page to a 150 DPI RGB image |
| Embedding model | ColQwen2.5-v0.2 (3B params, 4-bit quantized) | Produces 128-dim multi-vector embeddings per page via late interaction |
| Vector store | Qdrant Cloud | Stores multi-vector embeddings with MaxSim (cosine) retrieval |
| Answer generation | Llama 4 Maverick 17B (NVIDIA NIM) | Multimodal LLM that receives query + retrieved page images and generates cited answers |
| Interface | Streamlit | Web UI for document upload, question input, and answer display with source pages |

## 2. Design Choices

**Why ColPali / ColQwen2.5?** Traditional RAG pipelines rely on text extraction followed by chunking and text-only embeddings. This fails on tables, charts, and figures. ColPali (Contextual Late Interaction over Pages) embeds entire page images using a vision-language model, treating retrieval as a visual matching problem. ColQwen2.5-v0.2 is a LoRA adapter on Qwen2.5-VL-3B that produces multi-vector representations (one 128-dim vector per visual token), enabling fine-grained late interaction matching via MaxSim -- the same mechanism used in ColBERT but extended to vision.

**Why page-level chunking?** ColPali is designed for page-level granularity. Each page becomes one embedding unit. This naturally preserves document structure -- a table, its caption, and surrounding text stay together. No semantic chunking heuristics are needed.

**Why 4-bit quantization?** The target hardware has only 4 GB VRAM (RTX 3050 Ti). BitsAndBytes 4-bit quantization reduces the 3B-parameter model from ~6 GB to ~2.6 GB, leaving enough headroom for inference activations. A custom safetensors loader (`compat.py`) bypasses Windows memory-mapping limitations by using streaming file I/O with automatic GPU offloading for large tensors.

**Why a multimodal LLM for generation?** The retrieved pages are sent as images (not extracted text) to Llama 4 Maverick via NVIDIA NIM. This preserves all visual information -- the LLM can read tables, charts, and figures directly, producing answers grounded in what it actually sees rather than lossy text extraction.

## 3. System Modules

The codebase follows a modular design with five packages:

- **`ingestion/`** -- `pdf_converter.py` renders pages; `embedder.py` encodes them with ColQwen2.5 and upserts into Qdrant.
- **`retrieval/`** -- `retriever.py` encodes text queries, performs MaxSim search against Qdrant, and returns ranked pages with metadata.
- **`generation/`** -- `generator.py` builds a multimodal prompt (images + query) and calls the NVIDIA NIM API with citation instructions.
- **`evaluation/`** -- `benchmark.py` runs 12 predefined queries across text, table, and image modalities, measuring Hit@1 retrieval accuracy.
- **`interface/`** -- `app.py` provides a Streamlit web UI for interactive QA with retrieved page visualization.
- **`compat.py`** -- Compatibility layer that patches transformers 5.x for Windows memory constraints: custom safetensors reader with streaming I/O, checkpoint key remapping, and memory management fixes.

## 4. Benchmark Results

Evaluated on 12 benchmark queries against the NASA JSC 2024 Annual Report (23 pages):

| Modality | Queries | Hit@1 | Accuracy |
|----------|---------|-------|----------|
| Text | 6 | 6 | **100%** |
| Table | 4 | 3 | **75%** |
| Image | 2 | 2 | **100%** |
| **Overall** | **12** | **11** | **92%** |

The single miss (Query 8) retrieved page 4 instead of page 5, but the answer was still factually correct since the workforce data appears on both pages. All generated answers included accurate page-level citations.

## 5. Key Observations

1. **Vision-first RAG eliminates modality silos.** By embedding page images rather than extracted text, the system handles text, tables, charts, and figures uniformly without separate extraction pipelines.

2. **Late interaction (MaxSim) excels at fine-grained matching.** Multi-vector representations allow token-level alignment between query terms and page regions, outperforming single-vector approaches for complex queries that reference specific details.

3. **Memory-constrained deployment is feasible.** With 4-bit quantization, streaming I/O, and careful memory management, a 3B-parameter vision-language model runs on consumer hardware (4 GB VRAM, 12 GB RAM) without quality degradation.

4. **Sending images (not text) to the generator preserves fidelity.** The LLM reads tables and figures directly from page images, avoiding OCR errors and layout reconstruction artifacts that plague text-extraction approaches.
