#!/usr/bin/env python3
"""
Open WebUI Chat Data Analyzer
A modern Streamlit dashboard for analyzing Open WebUI chat export

Installation:
pip install streamlit pandas plotly wordcloud textblob networkx

Usage:
streamlit run openwebui_chat_analyzer.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import re
from textblob import TextBlob
from wordcloud import WordCloud
import networkx as nx
import base64
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="Open WebUI Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)




def create_header():
    st.title("💬 Open WebUI Chat Analyzer")
    st.subheader("Transform your conversation data into actionable insights with beautiful visualizations")


@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process Open WebUI JSON data"""
    try:
        # Load JSON data
        data = json.load(uploaded_file)
        
        # Initialize lists for different data types
        chats = []
        messages = []
        
        # Process each chat
        for item in data:
            chat_info = {
                'chat_id': item.get('id', ''),
                'user_id': item.get('user_id', ''),
                'title': item.get('title', ''),
                'created_at': pd.to_datetime(item.get('created_at', 0), unit='s'),
                'updated_at': pd.to_datetime(item.get('updated_at', 0), unit='s'),
                'archived': item.get('archived', False),
                'pinned': item.get('pinned', False),
                'tags': item.get('meta', {}).get('tags', [])
            }
            chats.append(chat_info)
            
            # Process messages in this chat
            chat_data = item.get('chat', {})
            chat_messages = chat_data.get('messages', [])
            
            for msg in chat_messages:
                message_info = {
                    'chat_id': item.get('id', ''),
                    'message_id': msg.get('id', ''),
                    'parent_id': msg.get('parentId'),
                    'role': msg.get('role', ''),
                    'content': msg.get('content', ''),
                    'timestamp': pd.to_datetime(msg.get('timestamp', 0), unit='s'),
                    'model': msg.get('model', ''),
                    'models': msg.get('models', [])
                }
                messages.append(message_info)
        
        # Create DataFrames
        chats_df = pd.DataFrame(chats)
        messages_df = pd.DataFrame(messages)
        
        return chats_df, messages_df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None

def calculate_engagement_metrics(chats_df, messages_df):
    """Calculate user engagement metrics"""
    if messages_df.empty:
        return {}
    
    total_chats = len(chats_df)
    total_messages = len(messages_df)
    unique_users = chats_df['user_id'].nunique()
    
    # Messages per chat
    msg_per_chat = messages_df.groupby('chat_id').size().mean()
    
    # User messages vs assistant messages
    user_messages = len(messages_df[messages_df['role'] == 'user'])
    assistant_messages = len(messages_df[messages_df['role'] == 'assistant'])
    
    # Average conversation length
    conversations = messages_df.groupby('chat_id').size()
    avg_conversation_length = conversations.mean()
    
    return {
        'total_chats': total_chats,
        'total_messages': total_messages,
        'unique_users': unique_users,
        'avg_messages_per_chat': msg_per_chat,
        'user_messages': user_messages,
        'assistant_messages': assistant_messages,
        'avg_conversation_length': avg_conversation_length
    }

def create_time_series_chart(messages_df):
    """Create time series visualization with modern styling"""
    if messages_df.empty:
        return go.Figure()
    
    # Group messages by date
    daily_messages = messages_df.groupby(messages_df['timestamp'].dt.date).size().reset_index()
    daily_messages.columns = ['date', 'message_count']
    
    fig = px.line(daily_messages, x='date', y='message_count',
                  title='Daily Message Activity')
    
    fig.update_traces(
        line_color='#3b82f6', 
        line_width=3,
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    )
    
    fig.update_layout(
        hovermode='x unified',
        showlegend=False,
        template='plotly_white',
        title_font_size=18,
        title_font_color='#1f2937',
        title_font_weight=600,
        xaxis_title="Date",
        yaxis_title="Number of Messages",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, l=50, r=50, b=50)
    )
    
    return fig

def create_user_activity_chart(messages_df):
    """Create user activity heatmap with modern styling"""
    if messages_df.empty:
        return go.Figure()
    
    # Extract hour and day of week
    messages_df['hour'] = messages_df['timestamp'].dt.hour
    messages_df['day_of_week'] = messages_df['timestamp'].dt.day_name()
    
    # Create activity matrix
    activity_matrix = messages_df.groupby(['day_of_week', 'hour']).size().reset_index()
    activity_matrix.columns = ['day_of_week', 'hour', 'count']
    
    # Pivot for heatmap
    heatmap_data = activity_matrix.pivot(index='day_of_week', columns='hour', values='count').fillna(0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=list(range(24)),
        y=heatmap_data.index,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>Hour: %{x}:00<br>Messages: %{z}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title="Messages",
            tickmode="linear",
            tick0=0,
            len=0.7,
            thickness=15,
            x=1.02,
            xanchor='left',
            y=0.5,
            yanchor='middle'
        )
    ))
    
    fig.update_layout(
        title='Message Activity Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        template='plotly_white',
        title_font_size=18,
        title_font_color='#1f2937',
        title_font_weight=600,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, l=50, r=50, b=50)
    )
    
    return fig

def create_model_usage_chart(messages_df):
    """Create model usage visualization with modern styling"""
    if messages_df.empty:
        return go.Figure()
    
    # Count model usage
    model_counts = messages_df[messages_df['model'] != '']['model'].value_counts()
    
    if len(model_counts) == 0:
        return go.Figure()
    
    # Create color palette
    colors = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#6366f1', '#ec4899']
    
    fig = px.bar(
        x=model_counts.values,
        y=model_counts.index,
        orientation='h',
        title='Model Usage Distribution',
        color=model_counts.index,
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'},
        title_font_size=18,
        title_font_color='#1f2937',
        title_font_weight=600,
        xaxis_title="Number of Messages",
        yaxis_title="Model",
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, l=50, r=50, b=50)
    )
    
    return fig

def create_conversation_length_distribution(messages_df):
    """Create conversation length distribution with modern styling"""
    if messages_df.empty:
        return go.Figure()
    
    # Calculate conversation lengths
    conv_lengths = messages_df.groupby('chat_id').size()
    
    fig = px.histogram(
        x=conv_lengths.values,
        nbins=20,
        title='Conversation Length Distribution',
        color_discrete_sequence=['#3b82f6']
    )
    
    fig.update_layout(
        template='plotly_white',
        title_font_size=18,
        title_font_color='#1f2937',
        title_font_weight=600,
        xaxis_title="Messages per Conversation",
        yaxis_title="Number of Conversations",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, l=50, r=50, b=50)
    )
    
    return fig

def generate_word_cloud(messages_df):
    """Generate word cloud from user messages"""
    if messages_df.empty:
        return None
    
    # Get user messages only
    user_messages = messages_df[messages_df['role'] == 'user']['content']
    
    if len(user_messages) == 0:
        return None
    
    # Combine all text
    text = ' '.join(user_messages.astype(str))
    
    # Clean text
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text)
    
    if len(text.strip()) == 0:
        return None
    
    # Generate word cloud with modern colors
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis',
            font_path=None,
            prefer_horizontal=0.9
        ).generate(text)
        
        return wordcloud
    except:
        return None

def perform_sentiment_analysis(messages_df):
    """Perform sentiment analysis on messages"""
    if messages_df.empty:
        return pd.DataFrame()
    
    # Get user messages for sentiment analysis
    user_messages = messages_df[messages_df['role'] == 'user'].copy()
    
    if len(user_messages) == 0:
        return pd.DataFrame()
    
    # Calculate sentiment scores
    sentiments = []
    for content in user_messages['content']:
        try:
            blob = TextBlob(str(content))
            sentiment = blob.sentiment.polarity
            sentiments.append(sentiment)
        except:
            sentiments.append(0)
    
    user_messages['sentiment'] = sentiments
    
    # Categorize sentiments
    user_messages['sentiment_category'] = pd.cut(
        user_messages['sentiment'],
        bins=[-1, -0.1, 0.1, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    return user_messages

def create_search_interface(messages_df):
    """Create enhanced search interface"""
    if messages_df.empty:
        return
    
    st.header("Search Messages")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input("Search in message content:", placeholder="Enter search terms...", key="search_input")
    with col2:
        role_filter = st.selectbox("Filter by role:", ["All", "user", "assistant"])
    with col3:
        max_results = st.selectbox("Max results:", [10, 25, 50, 100])
    if search_query:
        filtered_messages = messages_df[
            messages_df['content'].str.contains(search_query, case=False, na=False)
        ]
        if role_filter != "All":
            filtered_messages = filtered_messages[filtered_messages['role'] == role_filter]
        st.success(f"✅ Found {len(filtered_messages)} messages containing '{search_query}'")
        for i, (_, msg) in enumerate(filtered_messages.head(max_results).iterrows()):
            with st.expander(f"#{i+1} {msg['role'].title()} - {msg['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(msg['content'])
                if msg['model']:
                    st.caption(f"Model: {msg['model']}")

def create_export_section(chats_df, messages_df):
    """Create modern export section"""
    st.header("Export Data")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Chat Data")
        csv_chats = chats_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Chat Data CSV",
            data=csv_chats,
            file_name="openwebui_chat_data.csv",
            mime="text/csv",
            key="download_chats"
        )
    with col2:
        st.subheader("Message Data")
        csv_messages = messages_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Message Data CSV",
            data=csv_messages,
            file_name="openwebui_message_data.csv",
            mime="text/csv",
            key="download_messages"
        )

def create_instructions():
    """Create modern instructions section"""
    st.header("How to export data from Open WebUI")
    st.write(
        """
        1. **Access your Open WebUI admin panel**
        2. **Go to Settings → Data & Privacy**
        3. **Click "Export All Chats"** to download your chat data as JSON
        4. **Upload the downloaded JSON file** using the file uploader above
        """
    )
    st.header("What you'll get from this analysis")
    st.write(
        """
        - **Overview Metrics**: Total chats, messages, unique users, average messages per conversation, user vs assistant message ratios
        - **Time Analysis**: Daily message activity trends, activity heatmaps by hour and day, conversation length distributions
        - **Model Usage**: Distribution of different AI models used, usage statistics and percentages, model performance insights
        - **Content Analysis**: Word clouds from user messages, message length analysis, content pattern identification
        - **Sentiment Analysis**: Positive, neutral, negative message ratios, sentiment trends over time, user satisfaction indicators
        - **Search & Filter**: Full-text search across all messages, filter by user role or time period, export filtered results
        """
    )

def main():
    # Create modern header
    create_header()
    
    
    uploaded_file = st.file_uploader(
        "Select JSON file",
        type=['json'],
        help="Upload the JSON file exported from your Open WebUI instance",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Load and process data with enhanced loading state
        with st.spinner("🔄 Processing chat data..."):
            chats_df, messages_df = load_and_process_data(uploaded_file)
        
        if chats_df is not None and messages_df is not None:
            st.success(f"✅ Data loaded successfully! Found {len(chats_df)} chats with {len(messages_df)} messages")
            metrics = calculate_engagement_metrics(chats_df, messages_df)
            st.header("Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Total Chats", value=f"{metrics['total_chats']:,}")
            with col2:
                st.metric(label="Total Messages", value=f"{metrics['total_messages']:,}")
            with col3:
                st.metric(label="Unique Users", value=f"{metrics['unique_users']:,}")
            with col4:
                st.metric(label="Avg Msgs/Chat", value=f"{metrics['avg_messages_per_chat']:.1f}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                user_pct = (metrics['user_messages'] / metrics['total_messages']) * 100
                st.metric(label="User Messages", value=f"{user_pct:.1f}%")
            with col2:
                assistant_pct = (metrics['assistant_messages'] / metrics['total_messages']) * 100
                st.metric(label="AI Messages", value=f"{assistant_pct:.1f}%")
            with col3:
                st.metric(label="Avg Conv Length", value=f"{metrics['avg_conversation_length']:.1f}")
            with col4:
                if not messages_df.empty:
                    active_days = messages_df['timestamp'].dt.date.nunique()
                    st.metric(label="Active Days", value=f"{active_days:,}")
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Time Analysis", 
                "🤖 Model Usage", 
                "💭 Content Analysis", 
                "😊 Sentiment", 
                "🔍 Search"
            ])
            
            with tab1:
                st.subheader("Time-based Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    time_fig = create_time_series_chart(messages_df)
                    st.plotly_chart(time_fig, use_container_width=True)
                with col2:
                    length_fig = create_conversation_length_distribution(messages_df)
                    st.plotly_chart(length_fig, use_container_width=True)
                activity_fig = create_user_activity_chart(messages_df)
                st.plotly_chart(activity_fig, use_container_width=True)
            with tab2:
                st.subheader("Model Usage Analysis")
                col1, col2 = st.columns([2, 1])
                with col1:
                    model_fig = create_model_usage_chart(messages_df)
                    st.plotly_chart(model_fig, use_container_width=True)
                with col2:
                    if not messages_df[messages_df['model'] != ''].empty:
                        model_stats = messages_df[messages_df['model'] != '']['model'].value_counts()
                        st.subheader("Model Statistics")
                        for i, (model, count) in enumerate(model_stats.head(5).items()):
                            percentage = (count / len(messages_df)) * 100
                            st.write(f"{model}: {count:,} ({percentage:.1f}%)")
                    else:
                        st.warning("ℹ️ No model information available in the uploaded data")
            with tab3:
                st.subheader("Content Analysis")
                wordcloud = generate_word_cloud(messages_df)
                if wordcloud:
                    st.markdown("### ☁️ Word Cloud (User Messages)")
                    img = wordcloud.to_image()
                    st.image(img, use_container_width=True)
                else:
                    st.warning("⚠️ Word cloud unavailable - No user messages found or text processing failed")
                if not messages_df.empty:
                    messages_df['content_length'] = messages_df['content'].str.len()
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_length = messages_df.groupby('role')['content_length'].mean()
                        fig = px.bar(
                            x=avg_length.index,
                            y=avg_length.values,
                            title="Average Message Length by Role",
                            color=avg_length.index,
                            color_discrete_sequence=['#3b82f6', '#8b5cf6']
                        )
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=18,
                            title_font_color='#1f2937',
                            title_font_weight=600,
                            xaxis_title="Role",
                            yaxis_title="Average Characters",
                            font=dict(family="Inter, sans-serif"),
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=60, l=50, r=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.histogram(
                            messages_df,
                            x='content_length',
                            title="Message Length Distribution",
                            color_discrete_sequence=['#3b82f6']
                        )
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=18,
                            title_font_color='#1f2937',
                            title_font_weight=600,
                            xaxis_title="Message Length (characters)",
                            yaxis_title="Number of Messages",
                            font=dict(family="Inter, sans-serif"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=60, l=50, r=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
            with tab4:
                st.subheader("Sentiment Analysis")
                sentiment_df = perform_sentiment_analysis(messages_df)
                if not sentiment_df.empty:
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment_counts = sentiment_df['sentiment_category'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution (User Messages)",
                            color_discrete_sequence=['#ef4444', '#6b7280', '#22c55e']
                        )
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=18,
                            title_font_color='#1f2937',
                            title_font_weight=600,
                            font=dict(family="Inter, sans-serif"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=60, l=50, r=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        sentiment_time = sentiment_df.groupby(sentiment_df['timestamp'].dt.date)['sentiment'].mean().reset_index()
                        fig = px.line(
                            sentiment_time,
                            x='timestamp',
                            y='sentiment',
                            title="Average Sentiment Over Time"
                        )
                        fig.update_traces(line_color='#3b82f6', line_width=3)
                        fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", annotation_text="Neutral")
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=18,
                            title_font_color='#1f2937',
                            title_font_weight=600,
                            xaxis_title="Date",
                            yaxis_title="Average Sentiment",
                            font=dict(family="Inter, sans-serif"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=60, l=50, r=50, b=50)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown("### Sentiment Breakdown")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_pct = (sentiment_counts.get('Positive', 0) / len(sentiment_df)) * 100
                        st.metric(label="Positive Messages", value=f"{positive_pct:.1f}%")
                    with col2:
                        neutral_pct = (sentiment_counts.get('Neutral', 0) / len(sentiment_df)) * 100
                        st.metric(label="Neutral Messages", value=f"{neutral_pct:.1f}%")
                    with col3:
                        negative_pct = (sentiment_counts.get('Negative', 0) / len(sentiment_df)) * 100
                        st.metric(label="Negative Messages", value=f"{negative_pct:.1f}%")
                else:
                    st.warning("⚠️ Sentiment analysis unavailable - No user messages found for analysis")
            with tab5:
                create_search_interface(messages_df)
            create_export_section(chats_df, messages_df)
    
    else:
        st.warning("Please upload your Open WebUI JSON export file to begin analysis")
        create_instructions()

if __name__ == "__main__":
    main()