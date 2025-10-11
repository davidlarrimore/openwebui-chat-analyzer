"""FastAPI application for the Open WebUI Chat Analyzer backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import API_ALLOWED_ORIGINS, API_HOST, API_PORT
from .routes import router
from .services import data_service

app = FastAPI(
    title="Open WebUI Chat Analyzer API",
    version="1.0.0",
    description="Backend service powering the Open WebUI Chat Analyzer dashboard.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def ensure_data_loaded() -> None:
    """Ensure the singleton data service loads default datasets on startup."""
    # The singleton already loads data in its __init__, but calling here captures reload scenarios.
    data_service.load_initial_data()


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Basic health-check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host=API_HOST, port=API_PORT, reload=True)
