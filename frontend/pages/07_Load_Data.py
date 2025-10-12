from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

st.set_page_config(
    page_title="Load Data | Open WebUI Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from frontend.core.api import BackendError, get_dataset_meta, is_backend_unavailable_error
from frontend.core.config import get_config
from frontend.core.processing import build_dataset_panel, determine_dataset_source
from frontend.ui import components
from frontend.ui.controls import render_admin_section, render_direct_connect_section, render_upload_section
from frontend.utils import state as app_state
from frontend.utils.logging import get_logger


LOGGER = get_logger("frontend.load_data")


def render_page() -> None:
    sidebar = st.sidebar
    components.render_sidebar_branding(sidebar)

    config = get_config()
    app_state.set_default_openwebui_values(
        config.direct_host_default,
        config.direct_api_key_default,
    )

    try:
        dataset_meta = get_dataset_meta()
    except BackendError as exc:
        LOGGER.error("Unable to fetch dataset metadata: %s", exc)
        if is_backend_unavailable_error(exc):
            components.render_backend_wait_splash(config.api_base_url, exc)
        else:
            st.error(
                "Unable to connect to the backend API. "
                f"Ensure the FastAPI service is running and reachable at `{config.api_base_url}`.\n\nDetails: {exc}"
            )
        return

    app_state.set_dataset_meta(dataset_meta)
    components.render_sidebar_status(dataset_meta, container=sidebar)

    dataset_ready = dataset_meta.chat_count > 0 or (dataset_meta.app_metadata or {}).get("chat_count", 0) > 0
    source_info = determine_dataset_source(dataset_meta)

    st.header("Load Data")
    st.caption("Connect to Open WebUI or upload exported files to populate the analyzer.")

    panel = build_dataset_panel(dataset_meta)
    col_dataset, col_log = st.columns(2)
    components.render_dataset_panel(panel, container=col_dataset)
    log_renderer = components.render_processing_log_panel(
        container=col_log,
        description="This panel will show data processing logs",
    )

    render_admin_section(
        dataset=dataset_meta,
        on_refresh=app_state.trigger_rerun,
        log_renderer=log_renderer,
    )
    
    render_direct_connect_section(
        dataset_meta,
        dataset_ready=dataset_ready,
        config=config,
        on_refresh=app_state.trigger_rerun,
        log_renderer=log_renderer,
    )

    render_upload_section(
        dataset_ready=dataset_ready,
        source_label=source_info.label,
        on_refresh=app_state.trigger_rerun,
        log_renderer=log_renderer,
    )

    if not dataset_ready:
        components.render_instructions()


if __name__ == "__main__":
    render_page()
