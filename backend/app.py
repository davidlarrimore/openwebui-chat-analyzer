"""FastAPI application for the Open WebUI Chat Analyzer backend."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import API_ALLOWED_ORIGINS, API_HOST, API_PORT
from .logging_config import configure_logging
from .routes import router
from .services import data_service

configure_logging()
LOGGER = logging.getLogger(__name__)
LOGGER.info("Creating Open WebUI Chat Analyzer FastAPI application")

app = FastAPI(
    title="Open WebUI Chat Analyzer API",
    version="1.0.0",
    description="Backend service powering the Open WebUI Chat Analyzer dashboard.",
)

# Configure CORS so that the Next.js dashboard (and other approved origins) can call the API.
# We deliberately keep the allow list driven by configuration so deployments can constrain access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the API router that exposes all data related endpoints.
app.include_router(router)


@app.on_event("startup")
def ensure_data_loaded() -> None:
    """Warm the shared DataService so the app is ready to serve requests.

    Ensures the singleton data service hydrates its caches when the ASGI server starts.
    Returns:
        None
    """
    LOGGER.info("Backend startup hook triggered – ensuring data service is hydrated")
    try:
        # The singleton already loads data in its __init__, but calling here captures reload scenarios.
        data_service.load_initial_data()
    except Exception:  # pylint: disable=broad-except
        LOGGER.exception("Data service failed to load during startup")
        raise
    LOGGER.info("Data service hydration complete")


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Report a simple OK status used for readiness checks.

    Returns:
        dict[str, str]: A service status payload.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    LOGGER.info("Launching Uvicorn development server on %s:%s", API_HOST, API_PORT)
    uvicorn.run("backend.app:app", host=API_HOST, port=API_PORT, reload=True)
