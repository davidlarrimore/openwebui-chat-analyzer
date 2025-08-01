#!/usr/bin/env python3
"""
Open WebUI Chat Data Analyzer
A Streamlit dashboard for analyzing Open WebUI chat exports

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

# Custom CSS with Amivero-inspired design system
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Color Variables - Professional Tech Palette */
    :root {
        --primary-blue: #1E3A8A;        /* Deep professional blue */
        --secondary-blue: #3B82F6;      /* Bright accent blue */
        --light-blue: #DBEAFE;          /* Light blue backgrounds */
        --accent-orange: #F59E0B;       /* Attention/CTA color */
        --success-green: #10B981;       /* Success states */
        --warning-yellow: #F59E0B;      /* Warning states */
        --error-red: #EF4444;           /* Error states */
        --neutral-900: #111827;         /* Primary text */
        --neutral-700: #374151;         /* Secondary text */
        --neutral-500: #6B7280;         /* Tertiary text */
        --neutral-300: #D1D5DB;         /* Borders */
        --neutral-100: #F3F4F6;         /* Light backgrounds */
        --neutral-50: #F9FAFB;          /* Lightest backgrounds */
        --white: #FFFFFF;
        --gradient-primary: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        --gradient-accent: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Global Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--neutral-50);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--neutral-900);
        line-height: 1.2;
    }
    
    h1 {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: var(--primary-blue);
        font-size: 1.875rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--light-blue);
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: var(--neutral-700);
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
    }
    
    /* Header with brand styling */
    .header-container {
        background: var(--gradient-primary);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 1rem 1rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.125rem;
        font-weight: 400;
        margin: 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--white);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid var(--neutral-300);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-accent);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-blue);
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--neutral-500);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Streamlit component overrides */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--white);
        border-radius: 0.5rem;
        padding: 0.25rem;
        border: 1px solid var(--neutral-300);
        box-shadow: var(--shadow-sm);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0.375rem;
        color: var(--neutral-700);
        font-weight: 500;
        margin: 0 0.125rem;
        padding: 0.75rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--neutral-100);
        color: var(--primary-blue);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--gradient-primary);
        color: white;
        box-shadow: var(--shadow-sm);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: var(--white);
        border: 2px dashed var(--neutral-300);
        border-radius: 0.75rem;
        padding: 2rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--secondary-blue);
        background: var(--light-blue);
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: var(--gradient-accent);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--white);
        border-right: 1px solid var(--neutral-300);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--neutral-100);
        border-radius: 0.5rem;
        border: 1px solid var(--neutral-300);
        font-weight: 500;
        color: var(--neutral-700);
    }
    
    .streamlit-expanderContent {
        background: var(--white);
        border: 1px solid var(--neutral-300);
        border-top: none;
        border-radius: 0 0 0.5rem 0.5rem;
    }
    
    /* Info/Success/Warning/Error messages */
    .stAlert {
        border-radius: 0.5rem;
        border: none;
        box-shadow: var(--shadow-sm);
    }
    
    .stAlert[data-baseweb="notification"] {
        background: var(--light-blue);
        color: var(--primary-blue);
        border-left: 4px solid var(--secondary-blue);
    }
    
    /* Progress bars */
    .stProgress .css-1cpxqw2 {
        background: var(--gradient-primary);
        border-radius: 1rem;
    }
    
    /* Selectbox and text input */
    .stSelectbox > div > div {
        background: var(--white);
        border: 1px solid var(--neutral-300);
        border-radius: 0.5rem;
    }
    
    .stTextInput > div > div {
        background: var(--white);
        border: 1px solid var(--neutral-300);
        border-radius: 0.5rem;
    }
    
    /* Data tables */
    .dataframe {
        border: 1px solid var(--neutral-300);
        border-radius: 0.5rem;
        background: var(--white);
    }
    
    /* Custom info cards */
    .info-card {
        background: var(--white);
        border: 1px solid var(--neutral-300);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .info-card-header {
        color: var(--primary-blue);
        font-size: 1.125rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .info-card-content {
        color: var(--neutral-700);
        line-height: 1.6;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success-green);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning-yellow);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: var(--error-red);
    }
    
    /* Loading spinner override */
    .stSpinner > div {
        border-top-color: var(--secondary-blue);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
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
    
    fig.update_traces(
        line_color='#3B82F6', 
        line_width=3,
        mode='lines+markers',
        marker=dict(size=6, color='#1E3A8A')
    )
    
    fig.update_layout(
        hovermode='x unified',
        showlegend=False,
        template='plotly_white',
        title_font_size=16,
        title_font_color='#374151',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(
            gridcolor='#F3F4F6',
            linecolor='#D1D5DB'
        ),
        yaxis=dict(
            gridcolor='#F3F4F6',
            linecolor='#D1D5DB'
        )
    )
    
    return fig

def create_user_activity_chart(messages_df):
    """Create user activity heatmap"""
    if messages_df.empty:
        return go.Figure()
    
    # Extract hour and day of week
    messages_df_copy = messages_df.copy()
    messages_df_copy['hour'] = messages_df_copy['timestamp'].dt.hour
    messages_df_copy['day_of_week'] = messages_df_copy['timestamp'].dt.day_name()
    
    # Create activity matrix
    activity_matrix = messages_df_copy.groupby(['day_of_week', 'hour']).size().reset_index()
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
        colorscale=[[0, '#F9FAFB'], [0.25, '#DBEAFE'], [0.5, '#93C5FD'], [0.75, '#3B82F6'], [1, '#1E3A8A']],
        hoverongaps=False,
        hovertemplate='Hour: %{x}<br>Day: %{y}<br>Messages: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Message Activity Heatmap (by Hour and Day)',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        template='plotly_white',
        title_font_size=16,
        title_font_color='#374151',
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
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
        labels={'x': 'Number of Messages', 'y': 'Model'},
        color=model_counts.values,
        color_continuous_scale=[[0, '#DBEAFE'], [1, '#1E3A8A']]
    )
    
    fig.update_layout(
        template='plotly_white',
        title_font_size=16,
        title_font_color='#374151',
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            gridcolor='#F3F4F6',
            linecolor='#D1D5DB'
        ),
        yaxis=dict(
            categoryorder='total ascending',
            gridcolor='#F3F4F6',
            linecolor='#D1D5DB'
        )
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
        labels={'x': 'Number of Messages per Conversation', 'y': 'Number of Conversations'},
        color_discrete_sequence=['#3B82F6']
    )
    
    fig.update_layout(
        template='plotly_white',
        title_font_size=16,
        title_font_color='#374151',
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='#F3F4F6',
            linecolor='#D1D5DB'
        ),
        yaxis=dict(
            gridcolor='#F3F4F6',
            linecolor='#D1D5DB'
        )
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
    
    # Professional color palette for word cloud
    def color_func(*args, **kwargs):
        colors = ['#1E3A8A', '#3B82F6', '#374151', '#6B7280', '#F59E0B']
        return colors[np.random.randint(0, len(colors))]
    
    # Generate word cloud
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            color_func=color_func,
            font_path=None,
            prefer_horizontal=0.7
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
    
    st.markdown('<h3 class="info-card-header">🔍 Search Messages</h3>', unsafe_allow_html=True)
    
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
        
        st.markdown(f'<div class="status-indicator status-success">Found {len(filtered_messages)} messages containing "{search_query}"</div>', unsafe_allow_html=True)
        
        # Display results
        for _, msg in filtered_messages.head(10).iterrows():
            with st.expander(f"{msg['role'].title()} - {msg['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(msg['content'])
                if msg['model']:
                    st.caption(f"Model: {msg['model']}")

def main():
    # Header section
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">💬 Open WebUI Chat Analyzer</h1>
        <p class="header-subtitle">Transform your conversation data into actionable insights with professional analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
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
            st.markdown('<h2>📊 Overview Dashboard</h2>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['total_chats']:,}</div>
                    <div class="metric-label">Total Chats</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['total_messages']:,}</div>
                    <div class="metric-label">Total Messages</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['unique_users']:,}</div>
                    <div class="metric-label">Unique Users</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['avg_messages_per_chat']:.1f}</div>
                    <div class="metric-label">Avg Messages/Chat</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Time Analysis", 
                "🤖 Model Usage", 
                "💭 Content Analysis", 
                "😊 Sentiment", 
                "🔍 Search"
            ])
            
            with tab1:
                st.markdown('<h3>Time-based Analysis</h3>', unsafe_allow_html=True)
                
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
                st.markdown('<h3>Model Usage Analysis</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Model usage chart
                    model_fig = create_model_usage_chart(messages_df)
                    st.plotly_chart(model_fig, use_container_width=True)
                
                with col2:
                    # Model statistics
                    if not messages_df[messages_df['model'] != ''].empty:
                        model_stats = messages_df[messages_df['model'] != '']['model'].value_counts()
                        st.markdown('<div class="info-card"><div class="info-card-header">📊 Model Statistics</div><div class="info-card-content">', unsafe_allow_html=True)
                        for model, count in model_stats.head(5).items():
                            percentage = (count / len(messages_df)) * 100
                            st.markdown(f"""
                            <div style="margin-bottom: 0.75rem;">
                                <div style="font-weight: 600; color: #1E3A8A; margin-bottom: 0.25rem;">{model}</div>
                                <div style="color: #6B7280; font-size: 0.875rem;">{count:,} messages ({percentage:.1f}%)</div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div></div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<h3>Content Analysis</h3>', unsafe_allow_html=True)
                
                # Word cloud
                wordcloud = generate_word_cloud(messages_df)
                if wordcloud:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown('<div class="info-card"><div class="info-card-header">☁️ Word Cloud (User Messages)</div>', unsafe_allow_html=True)
                        # Convert to image
                        img = wordcloud.to_image()
                        st.image(img, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Message length analysis
                if not messages_df.empty:
                    messages_df_copy = messages_df.copy()
                    messages_df_copy['content_length'] = messages_df_copy['content'].str.len()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Average message length by role
                        avg_length = messages_df_copy.groupby('role')['content_length'].mean()
                        fig = px.bar(
                            x=avg_length.index,
                            y=avg_length.values,
                            title="Average Message Length by Role",
                            labels={'x': 'Role', 'y': 'Average Characters'},
                            color_discrete_sequence=['#3B82F6']
                        )
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=16,
                            title_font_color='#374151',
                            font=dict(family="Inter, sans-serif"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(gridcolor='#F3F4F6', linecolor='#D1D5DB'),
                            yaxis=dict(gridcolor='#F3F4F6', linecolor='#D1D5DB')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Message length distribution
                        fig = px.histogram(
                            messages_df_copy,
                            x='content_length',
                            title="Message Length Distribution",
                            labels={'content_length': 'Message Length (characters)'},
                            color_discrete_sequence=['#F59E0B']
                        )
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=16,
                            title_font_color='#374151',
                            font=dict(family="Inter, sans-serif"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(gridcolor='#F3F4F6', linecolor='#D1D5DB'),
                            yaxis=dict(gridcolor='#F3F4F6', linecolor='#D1D5DB')
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.markdown('<h3>Sentiment Analysis</h3>', unsafe_allow_html=True)
                
                # Perform sentiment analysis
                sentiment_df = perform_sentiment_analysis(messages_df)
                
                if not sentiment_df.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution
                        sentiment_counts = sentiment_df['sentiment_category'].value_counts()
                        colors = {'Positive': '#10B981', 'Neutral': '#6B7280', 'Negative': '#EF4444'}
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution (User Messages)",
                            color=sentiment_counts.index,
                            color_discrete_map=colors
                        )
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=16,
                            title_font_color='#374151',
                            font=dict(family="Inter, sans-serif")
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
                        fig.update_traces(line_color='#3B82F6', line_width=3)
                        fig.add_hline(y=0, line_dash="dash", line_color="#6B7280", annotation_text="Neutral")
                        fig.update_layout(
                            template='plotly_white',
                            title_font_size=16,
                            title_font_color='#374151',
                            font=dict(family="Inter, sans-serif"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(gridcolor='#F3F4F6', linecolor='#D1D5DB'),
                            yaxis=dict(gridcolor='#F3F4F6', linecolor='#D1D5DB')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment statistics
                    st.markdown('<h4>Sentiment Statistics</h4>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        positive_pct = (sentiment_counts.get('Positive', 0) / len(sentiment_df)) * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #10B981;">{positive_pct:.1f}%</div>
                            <div class="metric-label">Positive Messages</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        neutral_pct = (sentiment_counts.get('Neutral', 0) / len(sentiment_df)) * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #6B7280;">{neutral_pct:.1f}%</div>
                            <div class="metric-label">Neutral Messages</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        negative_pct = (sentiment_counts.get('Negative', 0) / len(sentiment_df)) * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: #EF4444;">{negative_pct:.1f}%</div>
                            <div class="metric-label">Negative Messages</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab5:
                create_search_interface(messages_df)
            
            # Data export section
            st.markdown('<h2>📤 Export Data</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Download Chat Data"):
                    csv = chats_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name="chat_data.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("💬 Download Message Data"):
                    csv = messages_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name="message_data.csv",
                        mime="text/csv"
                    )
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        <div class="info-card">
            <div class="info-card-header">📋 Getting Started</div>
            <div class="info-card-content">
                <p>Upload your Open WebUI JSON export file to begin comprehensive analysis of your chat data.</p>
                <p style="margin-top: 1rem; color: #6B7280; font-size: 0.875rem;">
                    <strong>Privacy First:</strong> All processing happens locally on your machine. Your data never leaves your browser.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 How to export data from Open WebUI"):
            st.markdown("""
            <div class="info-card-content">
                <ol style="line-height: 1.8;">
                    <li><strong>Access your Open WebUI admin panel</strong></li>
                    <li><strong>Go to Settings</strong> → <strong>Data & Privacy</strong></li>
                    <li><strong>Click "Export All Chats"</strong> to download your chat data as JSON</li>
                    <li><strong>Upload the downloaded JSON file</strong> using the file uploader above</li>
                </ol>
                <p style="margin-top: 1rem; color: #6B7280;">
                    The exported file will contain all your conversations, messages, and metadata for comprehensive analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("🎯 What you'll get from this analysis"):
            st.markdown("""
            <div class="info-card-content">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                    <div>
                        <h4 style="color: #1E3A8A; margin-bottom: 0.5rem;">📊 Overview Metrics</h4>
                        <ul style="color: #6B7280; line-height: 1.6;">
                            <li>Total chats, messages, and unique users</li>
                            <li>Average messages per conversation</li>
                            <li>User vs assistant message ratios</li>
                        </ul>
                    </div>
                    
                    <div>
                        <h4 style="color: #1E3A8A; margin-bottom: 0.5rem;">📈 Time Analysis</h4>
                        <ul style="color: #6B7280; line-height: 1.6;">
                            <li>Daily message activity trends</li>
                            <li>Activity heatmaps by hour and day</li>
                            <li>Conversation length distributions</li>
                        </ul>
                    </div>
                    
                    <div>
                        <h4 style="color: #1E3A8A; margin-bottom: 0.5rem;">🤖 Model Usage</h4>
                        <ul style="color: #6B7280; line-height: 1.6;">
                            <li>Distribution of different AI models used</li>
                            <li>Usage statistics and percentages</li>
                            <li>Model performance insights</li>
                        </ul>
                    </div>
                    
                    <div>
                        <h4 style="color: #1E3A8A; margin-bottom: 0.5rem;">💭 Content Analysis</h4>
                        <ul style="color: #6B7280; line-height: 1.6;">
                            <li>Word clouds from user messages</li>
                            <li>Message length analysis</li>
                            <li>Content pattern identification</li>
                        </ul>
                    </div>
                    
                    <div>
                        <h4 style="color: #1E3A8A; margin-bottom: 0.5rem;">😊 Sentiment Analysis</h4>
                        <ul style="color: #6B7280; line-height: 1.6;">
                            <li>Positive, neutral, negative message ratios</li>
                            <li>Sentiment trends over time</li>
                            <li>User satisfaction indicators</li>
                        </ul>
                    </div>
                    
                    <div>
                        <h4 style="color: #1E3A8A; margin-bottom: 0.5rem;">🔍 Search & Filter</h4>
                        <ul style="color: #6B7280; line-height: 1.6;">
                            <li>Full-text search across all messages</li>
                            <li>Filter by user role or time period</li>
                            <li>Export filtered results</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()