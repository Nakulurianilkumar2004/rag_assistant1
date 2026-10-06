"""Query pipeline: answer questions about Zoiko Rooms using the index built by ingest.py.

RAG steps handled here (on every question):
  5. RETRIEVE - embed the question and find the most similar chunks
  6. AUGMENT  - put those chunks into the prompt as context
  7. GENERATE - stream an answer from a Groq-hosted LLM
"""

import json
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq

from ingest import CHUNKS_FILE, EMBED_MODEL, VECTORS_FILE

load_dotenv(Path(__file__).parent / ".env")  # picks up GROQ_API_KEY from the project's .env

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = 5
MIN_SCORE = 0.6  # bge cosine scores below this are usually unrelated (off-topic queries score ~0.55)

SYSTEM_PROMPT = """You are the help assistant for Zoiko Rooms (zoikorooms.com), a platform for \
verified private rooms rented for 30+ nights.

Answer using only the website excerpts provided in the user's message. If the excerpts don't \
contain the answer, say you couldn't find it on the website and suggest contacting Zoiko Rooms \
support. Keep answers concise and friendly.

End with a line "Sources:" followed by each distinct source page URL you relied on (each URL once), one per line as "- https://...". Write URLs plainly, with no brackets around them."""


# ---------- Step 5: RETRIEVE ----------
class Retriever:
    def __init__(self):
        if not (CHUNKS_FILE.exists() and VECTORS_FILE.exists()):
            raise FileNotFoundError("Index not found - run `python ingest.py` first.")
        self.chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        self.vectors = np.load(VECTORS_FILE)
        self.model = TextEmbedding(EMBED_MODEL)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        q = np.array(next(iter(self.model.query_embed(query))), dtype=np.float32)
        scores = self.vectors @ (q / np.linalg.norm(q))
        best = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in best if scores[i] >= MIN_SCORE]


# ---------- Step 6: AUGMENT ----------
def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f'<excerpt source="{h["url"]}" title="{h["title"]}">\n{h["text"]}\n</excerpt>' for h in hits
    ) or "(no relevant excerpts found)"
    return f"<website_excerpts>\n{context}\n</website_excerpts>\n\nQuestion: {question}"


# ---------- Step 7: GENERATE ----------
def answer_stream(
    client: Groq, retriever: Retriever, question: str, history: list[dict] | None = None
) -> tuple[Iterator[str], list[dict]]:
    """Return (text stream, retrieved hits). `history` is prior user/assistant text turns."""
    hits = retriever.search(question)
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + (history or [])
        + [{"role": "user", "content": build_prompt(question, hits)}]
    )

    # gpt-oss models think before answering: keep that short and out of the visible reply.
    reasoning = {"reasoning_effort": "low", "include_reasoning": False} if "gpt-oss" in GROQ_MODEL else {}

    def gen() -> Iterator[str]:
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,  # keep answers close to the source text
            max_completion_tokens=2048,
            stream=True,
            **reasoning,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return gen(), hits


def main() -> None:
    client = Groq()  # reads GROQ_API_KEY
    retriever = Retriever()
    print(f"Loaded {len(retriever.chunks)} chunks. Model: {GROQ_MODEL}. Ask about Zoiko Rooms (Ctrl+C to quit).")
    history: list[dict] = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        stream, _ = answer_stream(client, retriever, question, history)
        print("Assistant: ", end="", flush=True)
        reply = ""
        for text in stream:
            reply += text
            print(text, end="", flush=True)
        print()
        # Keep plain question/answer text so follow-ups have context without resending old excerpts.
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
