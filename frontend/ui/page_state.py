from __future__ import annotations

import streamlit as st

from frontend.core.api import BackendError, get_dataset_meta, is_backend_unavailable_error
from frontend.core.config import get_config
from frontend.core.models import DatasetMeta, ProcessedData
from frontend.ui import components
from frontend.ui.data_access import load_processed_data
from frontend.utils import state as app_state
from frontend.utils.logging import get_logger


LOGGER = get_logger("frontend.page_state")


def ensure_data_ready() -> tuple[DatasetMeta, ProcessedData]:
    """Ensure dataset metadata and processed data are available for a page."""
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
        st.stop()

    app_state.set_dataset_meta(dataset_meta)
    sidebar = st.sidebar
    components.render_sidebar_branding(sidebar)
    components.render_sidebar_status(dataset_meta, container=sidebar)

    dataset_ready = dataset_meta.chat_count > 0 or (dataset_meta.app_metadata or {}).get("chat_count", 0) > 0

    if not dataset_ready:
        st.info(
            "No data loaded yet. Visit the Load Data page to connect to Open WebUI or upload exports."
        )
        components.render_instructions()
        st.stop()

    with st.spinner("🔄 Processing chat data..."):
        processed_data = load_processed_data(dataset_meta.dataset_id)

    app_state.set_processed_data(processed_data)
    dataset_changed = app_state.update_dataset_signature(
        dataset_meta.dataset_id,
        processed_data.chats,
        processed_data.messages,
    )
    if dataset_changed:
        app_state.ensure_filter_defaults()
        app_state.reset_model_filter()

    return dataset_meta, processed_data
