# shl-recommender

SHL Assessment Recommender API (skeleton).

This project will expose a FastAPI service that can:
- Scrape/build an SHL product catalog (`catalog.json`)
- Embed catalog items using Google Embedding API
- Store/search embeddings in a FAISS vector store
- Use a conversational agent (Gemini 1.5 Flash via LangChain) to clarify needs and recommend assessments

## Status

This is **scaffold-only**: files, imports, and TODOs are present, but core logic is intentionally not implemented yet.

## Quickstart (once implemented)

1) Create a virtual environment and install deps:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Configure environment:
- Copy `.env.example` to `.env`
- Fill in `GEMINI_API_KEY`

3) Run the server:

```bash
uvicorn main:app --reload
```

## API

- `GET /health`: basic liveness check
- `POST /chat`: chat with the recommender agent

`POST /chat` response schema is:

```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "string",
      "test_type": "string"
    }
  ],
  "end_of_conversation": true
}
```

