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


# Simplified Streamlit-compliant CSS injection
def inject_custom_css():
    st.markdown(
        """
        <style></style>

        """,
        unsafe_allow_html=True
    )

inject_custom_css()


def create_header():
    st.title("💬 Open WebUI Chat Analyzer")
    st.subheader("Transform your conversation data into actionable insights with beautiful visualizations")

def create_metric_card(title, value, icon="📊", change=None):
    """Create a modern metric card"""
    change_html = ""
    if change:
        change_color = "#22c55e" if change >= 0 else "#ef4444"
        change_icon = "↗️" if change >= 0 else "↘️"
        change_html = f"""
        <div style="font-size: 0.875rem; color: {change_color}; font-weight: 500; margin-top: 0.5rem;">
            {change_icon} {change:+.1f}%
        </div>
        """
    
    return f"""
    <div class="metric-card fade-in">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <span style="font-size: 0.875rem; color: #6b7280; font-weight: 500;">{title}</span>
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: #1f2937;">{value}</div>
        {change_html}
    </div>
    """

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
        title_font_size=16,
        title_font_color='#1f2937',
        xaxis_title="Date",
        yaxis_title="Number of Messages",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=50, r=50, b=50)
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
        x=1.02,             # positions it slightly outside the plot
        xanchor='left',     # anchors colorbar's left edge to x-position
        y=0.5,              # vertically centered
        yanchor='middle'    # anchors colorbar vertically in the middle
    )
))
    
    fig.update_layout(
        title='Message Activity Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        template='plotly_white',
        title_font_size=16,
        title_font_color='#1f2937',
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=50, r=50, b=50)
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
    colors = ['#3b82f6', '#8b5cf6', '#06d6a0', '#f59e0b', '#ef4444', '#6366f1', '#ec4899']
    
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
        title_font_size=16,
        title_font_color='#1f2937',
        xaxis_title="Number of Messages",
        yaxis_title="Model",
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=50, r=50, b=50)
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
        title_font_size=16,
        title_font_color='#1f2937',
        xaxis_title="Messages per Conversation",
        yaxis_title="Number of Conversations",
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, l=50, r=50, b=50)
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
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">🔍</span>
        <h2>Search Messages</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_query = st.text_input("Search in message content:", placeholder="Enter search terms...", key="search_input")
    
    with col2:
        role_filter = st.selectbox("Filter by role:", ["All", "user", "assistant"])
    
    with col3:
        max_results = st.selectbox("Max results:", [10, 25, 50, 100])
    
    if search_query:
        # Filter messages
        filtered_messages = messages_df[
            messages_df['content'].str.contains(search_query, case=False, na=False)
        ]
        
        if role_filter != "All":
            filtered_messages = filtered_messages[filtered_messages['role'] == role_filter]
        
        st.markdown(f"""
        <div class="status-success">
            <strong>Found {len(filtered_messages)} messages</strong> containing '{search_query}'
        </div>
        """, unsafe_allow_html=True)
        
        # Display results
        for i, (_, msg) in enumerate(filtered_messages.head(max_results).iterrows()):
            with st.expander(f"#{i+1} {msg['role'].title()} - {msg['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(msg['content'])
                if msg['model']:
                    st.caption(f"Model: {msg['model']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def create_export_section(chats_df, messages_df):
    """Create modern export section"""
    st.markdown("""
    <div class="section-header">
        <span class="icon">📤</span>
        <h2>Export Data</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.subheader("📊 Chat Data")
            st.write(f"Export {len(chats_df)} conversations with metadata")
            
            csv_chats = chats_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Chat Data CSV",
                data=csv_chats,
                file_name="openwebui_chat_data.csv",
                mime="text/csv",
                key="download_chats"
            )
    
    with col2:
        with st.container():
            st.subheader("💬 Message Data")
            st.write(f"Export {len(messages_df)} individual messages")
            
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
    st.markdown("""
    <div class="info-box">
        <h3>📋 How to export data from Open WebUI</h3>
        <ol style="line-height: 1.8;">
            <li><strong>Access your Open WebUI admin panel</strong></li>
            <li><strong>Go to Settings</strong> → <strong>Data & Privacy</strong></li>
            <li><strong>Click "Export All Chats"</strong> to download your chat data as JSON</li>
            <li><strong>Upload the downloaded JSON file</strong> using the file uploader above</li>
        </ol>
        <p style="margin-top: 1rem; font-style: italic;">The exported file will contain all your conversations, messages, and metadata.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 What you'll get from this analysis</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div>
                <h4 style="color: #1d4ed8; margin-bottom: 0.5rem;">📊 Overview Metrics</h4>
                <ul style="margin: 0;">
                    <li>Total chats, messages, and unique users</li>
                    <li>Average messages per conversation</li>
                    <li>User vs assistant message ratios</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #1d4ed8; margin-bottom: 0.5rem;">📈 Time Analysis</h4>
                <ul style="margin: 0;">
                    <li>Daily message activity trends</li>
                    <li>Activity heatmaps by hour and day</li>
                    <li>Conversation length distributions</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #1d4ed8; margin-bottom: 0.5rem;">🤖 Model Usage</h4>
                <ul style="margin: 0;">
                    <li>Distribution of different AI models used</li>
                    <li>Usage statistics and percentages</li>
                    <li>Model performance insights</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #1d4ed8; margin-bottom: 0.5rem;">💭 Content Analysis</h4>
                <ul style="margin: 0;">
                    <li>Word clouds from user messages</li>
                    <li>Message length analysis</li>
                    <li>Content pattern identification</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #1d4ed8; margin-bottom: 0.5rem;">😊 Sentiment Analysis</h4>
                <ul style="margin: 0;">
                    <li>Positive, neutral, negative message ratios</li>
                    <li>Sentiment trends over time</li>
                    <li>User satisfaction indicators</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #1d4ed8; margin-bottom: 0.5rem;">🔍 Search & Filter</h4>
                <ul style="margin: 0;">
                    <li>Full-text search across all messages</li>
                    <li>Filter by user role or time period</li>
                    <li>Export filtered results</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Create modern header
    create_header()
    
    # File uploader inside the styled container
    uploaded_file = st.file_uploader(
        "Select JSON file",
        type=['json'],
        help="Upload the JSON file exported from your Open WebUI instance"
    )
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("🔄 Processing chat data..."):
            chats_df, messages_df = load_and_process_data(uploaded_file)
        
        if chats_df is not None and messages_df is not None:
            # Success message
            st.markdown(f"""
            <div class="status-success fade-in">
                <strong>✅ Data loaded successfully!</strong> Found {len(chats_df)} chats with {len(messages_df)} messages
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate metrics
            metrics = calculate_engagement_metrics(chats_df, messages_df)
            
            # Overview section with modern metric cards
            st.markdown("""
            <div class="section-header">
                <span class="icon">📊</span>
                <h2>Overview</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(create_metric_card(
                    "Total Chats", 
                    f"{metrics['total_chats']:,}", 
                    "💬"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_metric_card(
                    "Total Messages", 
                    f"{metrics['total_messages']:,}", 
                    "📝"
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card(
                    "Unique Users", 
                    f"{metrics['unique_users']:,}", 
                    "👥"
                ), unsafe_allow_html=True)
            
            with col4:
                st.markdown(create_metric_card(
                    "Avg Msgs/Chat", 
                    f"{metrics['avg_messages_per_chat']:.1f}", 
                    "📊"
                ), unsafe_allow_html=True)
            
            # Additional metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                user_pct = (metrics['user_messages'] / metrics['total_messages']) * 100
                st.markdown(create_metric_card(
                    "User Messages", 
                    f"{user_pct:.1f}%", 
                    "👤"
                ), unsafe_allow_html=True)
            
            with col2:
                assistant_pct = (metrics['assistant_messages'] / metrics['total_messages']) * 100
                st.markdown(create_metric_card(
                    "AI Messages", 
                    f"{assistant_pct:.1f}%", 
                    "🤖"
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card(
                    "Avg Conv Length", 
                    f"{metrics['avg_conversation_length']:.1f}", 
                    "🔗"
                ), unsafe_allow_html=True)
            
            with col4:
                # Calculate active days
                if not messages_df.empty:
                    active_days = messages_df['timestamp'].dt.date.nunique()
                    st.markdown(create_metric_card(
                        "Active Days", 
                        f"{active_days:,}", 
                        "📅"
                    ), unsafe_allow_html=True)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Time Analysis", 
                "🤖 Model Usage", 
                "💭 Content Analysis", 
                "😊 Sentiment", 
                "🔍 Search"
            ])
            
            with tab1:
                st.markdown("""
                <div class="section-header">
                    <span class="icon">📈</span>
                    <h2>Time-based Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container():
                        time_fig = create_time_series_chart(messages_df)
                        st.plotly_chart(time_fig, use_container_width=True)
                
                with col2:
                    with st.container():
                        length_fig = create_conversation_length_distribution(messages_df)
                        st.plotly_chart(length_fig, use_container_width=True)
                
                # Activity heatmap
                with st.container():
                    activity_fig = create_user_activity_chart(messages_df)
                    st.plotly_chart(activity_fig, use_container_width=True)
            
            with tab2:
                st.markdown("""
                <div class="section-header">
                    <span class="icon">🤖</span>
                    <h2>Model Usage Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    with st.container():
                        model_fig = create_model_usage_chart(messages_df)
                        st.plotly_chart(model_fig, use_container_width=True)
                
                with col2:
                    with st.container():
                        if not messages_df[messages_df['model'] != ''].empty:
                            model_stats = messages_df[messages_df['model'] != '']['model'].value_counts()
                            st.subheader("📊 Model Statistics")
                            
                            for i, (model, count) in enumerate(model_stats.head(5).items()):
                                percentage = (count / len(messages_df)) * 100
                                # Create progress bar effect
                                st.markdown(f"""
                                <div style="margin: 1rem 0;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                        <span style="font-weight: 500; color: #1f2937;">{model}</span>
                                        <span style="color: #6b7280;">{count:,} ({percentage:.1f}%)</span>
                                    </div>
                                    <div style="background: #e5e7eb; border-radius: 0.5rem; height: 8px;">
                                        <div style="background: linear-gradient(90deg, #3b82f6, #8b5cf6); height: 100%; width: {percentage}%; border-radius: 0.5rem; transition: width 0.5s ease;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No model information available in the data")
            
            with tab3:
                st.markdown("""
                <div class="section-header">
                    <span class="icon">💭</span>
                    <h2>Content Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Word cloud
                wordcloud = generate_word_cloud(messages_df)
                if wordcloud:
                    st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
                    st.subheader("☁️ Word Cloud (User Messages)")
                    img = wordcloud.to_image()
                    st.image(img, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="status-warning">
                        <strong>⚠️ Word cloud unavailable</strong> - No user messages found or text processing failed
                    </div>
                    """, unsafe_allow_html=True)
                
                # Message length analysis
                if not messages_df.empty:
                    messages_df['content_length'] = messages_df['content'].str.len()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.container():
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
                                title_font_size=16,
                                title_font_color='#1f2937',
                                xaxis_title="Role",
                                yaxis_title="Average Characters",
                                font=dict(family="Inter, sans-serif"),
                                showlegend=False,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=50, l=50, r=50, b=50)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        with st.container():
                            fig = px.histogram(
                                messages_df,
                                x='content_length',
                                title="Message Length Distribution",
                                color_discrete_sequence=['#3b82f6']
                            )
                            fig.update_layout(
                                template='plotly_white',
                                title_font_size=16,
                                title_font_color='#1f2937',
                                xaxis_title="Message Length (characters)",
                                yaxis_title="Number of Messages",
                                font=dict(family="Inter, sans-serif"),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=50, l=50, r=50, b=50)
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.markdown("""
                <div class="section-header">
                    <span class="icon">😊</span>
                    <h2>Sentiment Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Perform sentiment analysis
                sentiment_df = perform_sentiment_analysis(messages_df)
                
                if not sentiment_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.container():
                            sentiment_counts = sentiment_df['sentiment_category'].value_counts()
                            fig = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="Sentiment Distribution (User Messages)",
                                color_discrete_sequence=['#ef4444', '#6b7280', '#22c55e']
                            )
                            fig.update_layout(
                                template='plotly_white',
                                title_font_size=16,
                                title_font_color='#1f2937',
                                font=dict(family="Inter, sans-serif"),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=50, l=50, r=50, b=50)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        with st.container():
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
                                title_font_size=16,
                                title_font_color='#1f2937',
                                xaxis_title="Date",
                                yaxis_title="Average Sentiment",
                                font=dict(family="Inter, sans-serif"),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=50, l=50, r=50, b=50)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment statistics with modern cards
                    st.markdown("### Sentiment Breakdown")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        positive_pct = (sentiment_counts.get('Positive', 0) / len(sentiment_df)) * 100
                        st.markdown(create_metric_card(
                            "Positive Messages", 
                            f"{positive_pct:.1f}%", 
                            "😊"
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        neutral_pct = (sentiment_counts.get('Neutral', 0) / len(sentiment_df)) * 100
                        st.markdown(create_metric_card(
                            "Neutral Messages", 
                            f"{neutral_pct:.1f}%", 
                            "😐"
                        ), unsafe_allow_html=True)
                    
                    with col3:
                        negative_pct = (sentiment_counts.get('Negative', 0) / len(sentiment_df)) * 100
                        st.markdown(create_metric_card(
                            "Negative Messages", 
                            f"{negative_pct:.1f}%", 
                            "😢"
                        ), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="status-warning">
                        <strong>⚠️ Sentiment analysis unavailable</strong> - No user messages found for analysis
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab5:
                create_search_interface(messages_df)
            
            # Export section
            create_export_section(chats_df, messages_df)
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        <div class="status-warning">
            <strong>👆 Please upload your Open WebUI JSON export file to begin analysis</strong>
        </div>
        """, unsafe_allow_html=True)
        
        create_instructions()

if __name__ == "__main__":
    main()