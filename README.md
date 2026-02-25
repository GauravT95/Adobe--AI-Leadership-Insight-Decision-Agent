# AI Leadership Insight & Decision Agent

AI-powered assistant that ingests company documents and answers leadership questions with source-grounded responses. Built with **Azure OpenAI**, **LangChain**, **LangGraph**, and **FastAPI**.

---

## Project Structure

```
├── Notebook/                            # Standalone Jupyter Notebook
│   ├── AI_Leadership_Insight_Agent.ipynb
│   └── sample_company_docs/             # Sample docs for the notebook
│
├── Complete code/                       # Full-stack application (Backend + Frontend UI)
│   ├── Backend/
│   │   ├── config.py                    # Azure OpenAI config (reads from env vars)
│   │   ├── main.py                      # FastAPI server
│   │   ├── models.py                    # Pydantic schemas
│   │   ├── rag_engine.py               # Core RAG engine (5 methods + LangGraph agent)
│   │   ├── requirements.txt
│   │   └── sample_company_docs/         # Pre-loaded company documents
│   └── Frontend/
│       ├── gl-insights-benchmark.html   # Main UI
│       ├── css/gl-insights.css
│       └── js/app.js
│
├── langflow_pipeline_export.json        # LangFlow visual pipeline export
└── README.md
```

---

## Two Ways to Run

### Option 1: Notebook (Standalone)

The notebook in `Notebook/` is fully self-contained — all 5 RAG methods, LangGraph agent, benchmarking, and visualizations in one file.

1. Install dependencies:
   ```bash
   pip install langchain-openai langchain-community langchain-text-splitters langgraph faiss-cpu rank-bm25 numpy pandas matplotlib seaborn pymupdf4llm python-docx
   ```

2. Open `Notebook/AI_Leadership_Insight_Agent.ipynb`

3. **Set your Azure OpenAI API key in Cell 1:**
   ```python
   AZURE_OPENAI_API_KEY = "your-api-key-here"
   ```

4. Run all cells sequentially.

---

### Option 2: Full-Stack App (Backend API + Frontend UI)

1. Create virtual environment & install dependencies:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate

   pip install -r "Complete code/Backend/requirements.txt"
   pip install pymupdf4llm python-docx python-multipart
   ```

2. **Set your Azure OpenAI API key** — either:
   - **Environment variable** (recommended):
     ```bash
     # Windows
     set AZURE_OPENAI_API_KEY=your-api-key-here
     # macOS/Linux
     export AZURE_OPENAI_API_KEY=your-api-key-here
     ```
   - **Or edit** `Complete code/Backend/config.py` directly.

3. Start the server:
   ```bash
   cd "Complete code/Backend"
   python -m uvicorn main:app --host 0.0.0.0 --port 8080
   ```

4. Open [http://localhost:8080](http://localhost:8080) in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question. Body: `{"question": "...", "method": "agentic"}` |
| `POST` | `/upload` | Upload a PDF or DOCX file |
| `GET` | `/health` | System health check |
| `GET` | `/documents` | List indexed documents |

### RAG Methods

| Value | Description |
|-------|-------------|
| `naive_dense` | FAISS vector similarity search |
| `bm25` | Keyword-based BM25 ranking |
| `hybrid` | Dense + Sparse with Reciprocal Rank Fusion |
| `hybrid_reranked` | Hybrid + LLM reranking |
| `agentic` | Full LangGraph agent (classify → decompose → retrieve → rerank → synthesize → quality-check) |

---

## Configuration

All Azure OpenAI settings are in `Complete code/Backend/config.py`. The notebook has its own config in Cell 1. Both default to reading `AZURE_OPENAI_API_KEY` from environment variables — **just set your API key and everything runs**.

| Setting | Default |
|---------|---------|
| `AZURE_OPENAI_ENDPOINT` | `https://ue2daoipocaoa0l.openai.azure.com/` |
| `AZURE_CHAT_MODEL` | `gpt-5` |
| `AZURE_EMBEDDING_MODEL` | `text-embedding` |

Update the endpoint and model names in config to match your Azure deployment.

---

## Tech Stack

Python 3.11+ · Azure OpenAI · LangChain · LangGraph · FAISS · BM25 · FastAPI · HTML/CSS/JS
