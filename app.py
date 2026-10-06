"""Web server for the Zoiko Rooms RAG assistant. Run: python app.py  ->  http://localhost:8080 (set PORT in .env to change)

Serves the HTML/CSS/JS frontend from static/ and exposes POST /api/chat, which streams
newline-delimited JSON events: {"type": "sources"}, then {"type": "token"}..., then {"type": "done"}.
"""

import json
import os
import sys
from pathlib import Path
from typing import Iterator, Literal

import groq
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import GROQ_MODEL, Retriever, answer_stream

STATIC_DIR = Path(__file__).parent / "static"
MAX_HISTORY = 10  # past messages sent back to the model for follow-up context

if not os.getenv("GROQ_API_KEY"):
    sys.exit(
        "GROQ_API_KEY is not set.\n"
        "Add this line to the .env file in this folder (and save it), then run again:\n"
        "  GROQ_API_KEY=gsk_your_key_here\n"
        "Get a key at https://console.groq.com/keys"
    )

app = FastAPI(title="Zoiko Rooms Assistant")
retriever = Retriever()
client = groq.Groq()  # reads GROQ_API_KEY (loaded from .env by rag.py)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


def event(**data) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    history = [m.model_dump() for m in req.history[-MAX_HISTORY:]]

    def events() -> Iterator[str]:
        try:
            stream, hits = answer_stream(client, retriever, req.question, history)
            yield event(type="sources", sources=[
                {"url": h["url"], "title": h["title"], "score": round(h["score"], 2)} for h in hits
            ])
            for text in stream:
                yield event(type="token", text=text)
            yield event(type="done")
        except groq.AuthenticationError:
            yield event(type="error", message="Invalid GROQ_API_KEY - check your .env file.")
        except groq.RateLimitError:
            yield event(type="error", message="Groq rate limit reached - wait a moment and try again.")
        except groq.APIStatusError as e:
            yield event(type="error", message=f"Groq API error ({e.status_code}): {e.message}")
        except groq.APIConnectionError:
            yield event(type="error", message="Could not reach the Groq API - check your connection.")

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/api/info")
def info() -> dict:
    return {"model": GROQ_MODEL, "chunks": len(retriever.chunks)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Open http://localhost:{port} in your browser")
    uvicorn.run(app, host="127.0.0.1", port=port)
