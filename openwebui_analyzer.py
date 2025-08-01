#!/usr/bin/env python3
"""
Open WebUI Chat Data Analyzer
A Streamlit dashboard for analyzing Open WebUI chat exports

Installation:
pip install streamlit pandas plotly wordcloud textblob networkx

Usage:
streamlit run openwebui_analyzer.py
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

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

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
    """Create time series visualization of message activity"""
    if messages_df.empty:
        return go.Figure()
    
    # Group messages by date
    daily_messages = messages_df.groupby(messages_df['timestamp'].dt.date).size().reset_index()
    daily_messages.columns = ['date', 'message_count']
    
    fig = px.line(daily_messages, x='date', y='message_count',
                  title='Daily Message Activity',
                  labels={'message_count': 'Number of Messages', 'date': 'Date'})
    
    fig.update_traces(line_color='#1f77b4', line_width=2)
    fig.update_layout(
        hovermode='x unified',
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_user_activity_chart(messages_df):
    """Create user activity heatmap"""
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
        hovertemplate='Hour: %{x}<br>Day: %{y}<br>Messages: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Message Activity Heatmap (by Hour and Day)',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        template='plotly_white'
    )
    
    return fig

def create_model_usage_chart(messages_df):
    """Create model usage visualization"""
    if messages_df.empty:
        return go.Figure()
    
    # Count model usage
    model_counts = messages_df[messages_df['model'] != '']['model'].value_counts()
    
    if len(model_counts) == 0:
        return go.Figure()
    
    fig = px.bar(
        x=model_counts.values,
        y=model_counts.index,
        orientation='h',
        title='Model Usage Distribution',
        labels={'x': 'Number of Messages', 'y': 'Model'}
    )
    
    fig.update_layout(
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_conversation_length_distribution(messages_df):
    """Create conversation length distribution"""
    if messages_df.empty:
        return go.Figure()
    
    # Calculate conversation lengths
    conv_lengths = messages_df.groupby('chat_id').size()
    
    fig = px.histogram(
        x=conv_lengths.values,
        nbins=20,
        title='Conversation Length Distribution',
        labels={'x': 'Number of Messages per Conversation', 'y': 'Number of Conversations'}
    )
    
    fig.update_layout(template='plotly_white')
    
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
    
    # Generate word cloud
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis'
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
    """Create search interface for messages"""
    if messages_df.empty:
        return
    
    st.subheader("🔍 Search Messages")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search in message content:", placeholder="Enter search terms...")
    
    with col2:
        role_filter = st.selectbox("Filter by role:", ["All", "user", "assistant"])
    
    if search_query:
        # Filter messages
        filtered_messages = messages_df[
            messages_df['content'].str.contains(search_query, case=False, na=False)
        ]
        
        if role_filter != "All":
            filtered_messages = filtered_messages[filtered_messages['role'] == role_filter]
        
        st.write(f"Found {len(filtered_messages)} messages containing '{search_query}'")
        
        # Display results
        for _, msg in filtered_messages.head(10).iterrows():
            with st.expander(f"{msg['role'].title()} - {msg['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(msg['content'])
                if msg['model']:
                    st.caption(f"Model: {msg['model']}")

def main():
    st.title("💬 Open WebUI Chat Data Analyzer")
    st.markdown("Upload your Open WebUI chat export JSON file to analyze conversation patterns and trends.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your Open WebUI JSON export file",
        type=['json'],
        help="Upload the JSON file exported from your Open WebUI instance"
    )
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Processing chat data..."):
            chats_df, messages_df = load_and_process_data(uploaded_file)
        
        if chats_df is not None and messages_df is not None:
            # Calculate metrics
            metrics = calculate_engagement_metrics(chats_df, messages_df)
            
            # Overview section
            st.header("📊 Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Chats", f"{metrics['total_chats']:,}")
            
            with col2:
                st.metric("Total Messages", f"{metrics['total_messages']:,}")
            
            with col3:
                st.metric("Unique Users", f"{metrics['unique_users']:,}")
            
            with col4:
                st.metric("Avg Messages/Chat", f"{metrics['avg_messages_per_chat']:.1f}")
            
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
                    # Time series chart
                    time_fig = create_time_series_chart(messages_df)
                    st.plotly_chart(time_fig, use_container_width=True)
                
                with col2:
                    # Conversation length distribution
                    length_fig = create_conversation_length_distribution(messages_df)
                    st.plotly_chart(length_fig, use_container_width=True)
                
                # Activity heatmap
                activity_fig = create_user_activity_chart(messages_df)
                st.plotly_chart(activity_fig, use_container_width=True)
            
            with tab2:
                st.subheader("Model Usage Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Model usage chart
                    model_fig = create_model_usage_chart(messages_df)
                    st.plotly_chart(model_fig, use_container_width=True)
                
                with col2:
                    # Model statistics
                    if not messages_df[messages_df['model'] != ''].empty:
                        model_stats = messages_df[messages_df['model'] != '']['model'].value_counts()
                        st.subheader("Model Statistics")
                        for model, count in model_stats.head(5).items():
                            percentage = (count / len(messages_df)) * 100
                            st.write(f"**{model}**: {count:,} messages ({percentage:.1f}%)")
            
            with tab3:
                st.subheader("Content Analysis")
                
                # Word cloud
                wordcloud = generate_word_cloud(messages_df)
                if wordcloud:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.subheader("Word Cloud (User Messages)")
                        # Convert to image
                        img = wordcloud.to_image()
                        st.image(img, use_container_width=True)
                
                # Message length analysis
                if not messages_df.empty:
                    messages_df['content_length'] = messages_df['content'].str.len()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Average message length by role
                        avg_length = messages_df.groupby('role')['content_length'].mean()
                        fig = px.bar(
                            x=avg_length.index,
                            y=avg_length.values,
                            title="Average Message Length by Role",
                            labels={'x': 'Role', 'y': 'Average Characters'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Message length distribution
                        fig = px.histogram(
                            messages_df,
                            x='content_length',
                            title="Message Length Distribution",
                            labels={'content_length': 'Message Length (characters)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Sentiment Analysis")
                
                # Perform sentiment analysis
                sentiment_df = perform_sentiment_analysis(messages_df)
                
                if not sentiment_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution
                        sentiment_counts = sentiment_df['sentiment_category'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution (User Messages)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Sentiment over time
                        sentiment_time = sentiment_df.groupby(sentiment_df['timestamp'].dt.date)['sentiment'].mean().reset_index()
                        fig = px.line(
                            sentiment_time,
                            x='timestamp',
                            y='sentiment',
                            title="Average Sentiment Over Time"
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment statistics
                    st.subheader("Sentiment Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        positive_pct = (sentiment_counts.get('Positive', 0) / len(sentiment_df)) * 100
                        st.metric("Positive Messages", f"{positive_pct:.1f}%")
                    
                    with col2:
                        neutral_pct = (sentiment_counts.get('Neutral', 0) / len(sentiment_df)) * 100
                        st.metric("Neutral Messages", f"{neutral_pct:.1f}%")
                    
                    with col3:
                        negative_pct = (sentiment_counts.get('Negative', 0) / len(sentiment_df)) * 100
                        st.metric("Negative Messages", f"{negative_pct:.1f}%")
            
            with tab5:
                create_search_interface(messages_df)
            
            # Data export section
            st.header("📤 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Processed Chat Data"):
                    csv = chats_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="chat_data.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Download Message Data"):
                    csv = messages_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="message_data.csv",
                        mime="text/csv"
                    )
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload your Open WebUI JSON export file to begin analysis.")
        
        with st.expander("📋 How to export data from Open WebUI"):
            st.markdown("""
            1. **Access your Open WebUI admin panel**
            2. **Go to Settings** → **Data & Privacy** 
            3. **Click "Export All Chats"** to download your chat data as JSON
            4. **Upload the downloaded JSON file** using the file uploader above
            
            The exported file will contain all your conversations, messages, and metadata.
            """)
        
        with st.expander("🎯 What you'll get from this analysis"):
            st.markdown("""
            **📊 Overview Metrics**
            - Total chats, messages, and unique users
            - Average messages per conversation
            - User vs assistant message ratios
            
            **📈 Time Analysis**
            - Daily message activity trends
            - Activity heatmaps by hour and day
            - Conversation length distributions
            
            **🤖 Model Usage**
            - Distribution of different AI models used
            - Usage statistics and percentages
            - Model performance insights
            
            **💭 Content Analysis**
            - Word clouds from user messages
            - Message length analysis
            - Content pattern identification
            
            **😊 Sentiment Analysis**
            - Positive, neutral, negative message ratios
            - Sentiment trends over time
            - User satisfaction indicators
            
            **🔍 Search & Filter**
            - Full-text search across all messages
            - Filter by user role or time period
            - Export filtered results
            """)

if __name__ == "__main__":
    main()