# generation/generator.py
# Takes the user query + retrieved page images, sends them to the multimodal LLM,
# and returns a grounded answer with page-level citations.

import base64
import io
from openai import OpenAI
from PIL import Image
from config import get_settings
from retrieval.retriever import RetrievedPage


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def generate_answer(query: str, pages: list[RetrievedPage]) -> str:
    """
    Build a multimodal prompt (query + page images) and call Kimi K2.
    Returns the answer text with inline citations like [Doc: X, Page: Y].
    """
    settings = get_settings()
    client = OpenAI(
        api_key=settings.kimi_api_key,
        base_url=settings.kimi_base_url,
    )

    # Build message content: interleave images with their citation labels
    content = []
    for page in pages:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{_pil_to_b64(page.image)}"
            },
        })
        content.append({
            "type": "text",
            "text": f"[Doc: {page.doc_name}, Page: {page.page_num}]",
        })

    citation_hint = ", ".join(
        f"[Doc: {p.doc_name}, Page: {p.page_num}]" for p in pages
    )
    content.append({
        "type": "text",
        "text": (
            f"Using only the page images above, answer the following question.\n"
            f"Cite the source of each claim using the format shown (e.g. {citation_hint}).\n\n"
            f"Question: {query}"
        ),
    })

    response = client.chat.completions.create(
        model=settings.kimi_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""
