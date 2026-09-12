# Frontend (Svelte + Vite)

Requires Node.js 18+.

## First time

    cd frontend
    npm install

## Run (needs TWO terminals)

Terminal 1 - backend:

    cd thai-legal-rag
    .venv\Scripts\activate
    python -m uvicorn api.main:app --port 8000

Terminal 2 - frontend:

    cd thai-legal-rag\frontend
    npm run dev

Then open http://localhost:5173

The Vite dev server proxies /api to http://127.0.0.1:8000,
so the browser sees a single origin and there are no CORS problems.

## Build for the presentation

    npm run build

Output goes to frontend/dist. Serving it from FastAPI is not set up;
running `npm run dev` during the demo is simpler and hot-reloads.
