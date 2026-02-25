"""
FastAPI application — AI Leadership Insight & Decision Agent
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import rag_engine
from models import AskRequest, AskResponse, HealthResponse

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Frontend"))


# ── Lifespan: initialise RAG engine on startup ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_engine.init_engine()
    yield


app = FastAPI(
    title="AI Leadership Insight & Decision Agent",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow the frontend) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Answer a leadership / business question using the selected RAG method."""
    valid = ["naive_dense", "bm25", "hybrid", "hybrid_reranked", "agentic"]
    if req.method not in valid:
        raise HTTPException(400, f"Invalid method '{req.method}'. Choose from {valid}")
    try:
        result = rag_engine.ask(req.question, method=req.method)
        return AskResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            question_type=result.get("question_type", ""),
            latency_seconds=result.get("latency_seconds", 0.0),
            cache_hit=result.get("cache_hit", False),
            matched_query=result.get("matched_query"),
            method=result.get("method", req.method),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check system health."""
    return HealthResponse(
        status="healthy" if rag_engine.llm else "not_ready",
        llm_ok=rag_engine.llm is not None,
        embeddings_ok=rag_engine.embeddings is not None,
        documents_loaded=len(set(c.metadata["source"] for c in rag_engine.all_chunks)) if rag_engine.all_chunks else 0,
        chunks_indexed=len(rag_engine.all_chunks),
    )


@app.get("/documents")
async def documents():
    """List loaded documents and chunk counts."""
    return rag_engine.get_doc_info()


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or Word document to the knowledge base."""
    fname = file.filename or "upload"
    ext = os.path.splitext(fname)[1].lower()
    if ext not in (".pdf", ".docx"):
        raise HTTPException(400, "Only .pdf and .docx files are supported.")
    content = await file.read()
    try:
        result = rag_engine.add_document(fname, content, ext)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Serve frontend static files ──
@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "gl-insights-benchmark.html"))


app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")


# ── Run directly with: python main.py ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
