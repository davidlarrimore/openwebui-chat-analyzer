from __future__ import annotations

import pandas as pd

from frontend.core.constants import ALL_MODELS_OPTION, ALL_USERS_OPTION
from frontend.core.processing import (
    calculate_engagement_metrics,
    filter_dataframes_by_user_model,
    perform_sentiment_analysis,
)


def test_calculate_engagement_metrics(sample_chats_df, sample_messages_df) -> None:
    metrics = calculate_engagement_metrics(sample_chats_df, sample_messages_df)
    assert metrics["total_chats"] == 2
    assert metrics["total_messages"] == 3
    assert metrics["unique_users"] == 2
    assert metrics["user_messages"] == 2
    assert metrics["assistant_messages"] == 1
    assert metrics["files_uploaded"] == 1


def test_filter_dataframes_by_user_and_model(sample_chats_df, sample_messages_df) -> None:
    filtered_chats, filtered_messages = filter_dataframes_by_user_model(
        sample_chats_df,
        sample_messages_df,
        user_id="user-1",
        model_name="model-a",
    )
    assert len(filtered_chats) == 1
    assert len(filtered_messages) == 2
    assert set(filtered_messages["chat_id"].unique()) == {"chat-1"}

    # Filtering with no matches returns empty dataframes.
    empty_chats, empty_messages = filter_dataframes_by_user_model(
        sample_chats_df,
        sample_messages_df,
        user_id="user-1",
        model_name="non-existent",
    )
    assert empty_chats.empty
    assert empty_messages.empty


def test_perform_sentiment_analysis(sample_messages_df) -> None:
    sentiment_df = perform_sentiment_analysis(sample_messages_df)
    assert not sentiment_df.empty
    assert "sentiment" in sentiment_df.columns
    assert "sentiment_category" in sentiment_df.columns
    assert sentiment_df["sentiment_category"].isin(["Positive", "Neutral", "Negative"]).all()

