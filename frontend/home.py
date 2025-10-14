from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is available on sys.path for `frontend.*` imports.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from frontend.core.api import BackendError, get_dataset_meta, is_backend_unavailable_error
from frontend.core.config import get_config
from frontend.ui import components
from frontend.utils import state as app_state
from frontend.utils.logging import get_logger


LOGGER = get_logger("frontend.home")


def main() -> None:
    st.set_page_config(
        page_title="Home | Open WebUI Chat Analyzer",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = get_config()
    app_state.set_default_openwebui_values(
        config.direct_host_default,
        config.direct_api_key_default,
    )

    dataset_meta = None

    try:
        dataset_meta = get_dataset_meta()
    except BackendError as exc:
        LOGGER.warning("Unable to fetch dataset metadata: %s", exc)
        if is_backend_unavailable_error(exc):
            components.render_backend_wait_splash(config.api_base_url, exc)
            st.stop()
        else:
            st.warning(
                "We couldn't reach the analytics backend yet. "
                "Double-check that the FastAPI service is running, then refresh to try again.\n\n"
                f"Details: {exc}"
            )

    st.title("Welcome to Open WebUI Chat Analyzer")
    st.caption("A friendly dashboard for turning Open WebUI conversations into insight.")

    if dataset_meta is not None:
        app_state.set_dataset_meta(dataset_meta)
        sidebar = st.sidebar
        components.render_sidebar_branding(sidebar)
        components.render_sidebar_status(dataset_meta, container=sidebar)

        dataset_ready = dataset_meta.chat_count > 0 or (dataset_meta.app_metadata or {}).get("chat_count", 0) > 0
        if dataset_ready:
            st.success(
                f"You're all set! {dataset_meta.chat_count:,} chats, {dataset_meta.message_count:,} messages, "
                f"and {dataset_meta.model_count:,} models are ready to explore on the Overview page."
            )
        else:
            st.info(
                "No chats detected yet. Visit **Load Data** to connect to Open WebUI "
                "and pull your workspace data into the analyzer."
            )
    else:
        st.info(
            "Once the backend responds, this page will automatically reflect your chat data status. "
            "If you're still starting things up, keep this tab open and refresh in a moment."
        )

    st.divider()

    st.subheader("What you can explore")
    st.markdown(
        """
        - **Activity at a glance** – See how busy conversations get across your workspace.
        - **Popular models** – Spot which models teammates rely on most.
        - **Conversation mood** – Follow the overall sentiment and highlight feel-good or tense stretches.
        - **Searchable history** – Jump into specific chats, filter by person or model, and download threads.
        """
    )

    st.subheader("Bring your data in")
    st.markdown("**Direct Connect**")
    st.write(
        "Point the app at a running Open WebUI instance. Add the base URL and API key on the Load Data page, "
        "and the analyzer will pull chats plus user details for you."
    )
    st.caption(
        "Need automation? The upload API endpoints still exist for scripted imports, but the UI flow is now deprecated."
    )

    st.subheader("First-time setup")
    st.markdown(
        """
        1. Make sure the `.env` file is in place (start with `.env.example` if you're setting things up for the first time).
        2. Start the analyzer using your usual approach—Docker Compose, the bundled Makefile helpers, or the local Python commands.
        3. Visit **Load Data**, add your Open WebUI URL plus API token, and let the importer finish.
        4. Open the **Overview** page for live metrics, then browse the other tabs for deeper dives.
        """
    )

    st.subheader("Helpful tips")
    st.markdown(
        """
        - Leave this window open while the backend warms up—the status card will refresh automatically.
        - Working offline? Everything stays on your machine; exports never leave your environment.
        - Want to experiment first? Use the sample files in `sample_data/` to see the charts with demo content.
        """
    )

    st.caption(
        "Need more detail? The project README covers advanced configuration, customization ideas, and troubleshooting steps."
    )

    st.markdown(
        """
        <div style="margin-top:1.5rem;">
            <a href="https://github.com/davidlarrimore/openwebui-chat-analyzer" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="22" style="vertical-align:middle;margin-right:0.4rem;" alt="GitHub logo" />
                View this project on GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
