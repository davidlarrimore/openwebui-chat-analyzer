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


def inject_custom_css():
    """Inject modern CSS styling with dark mode support and animations"""
    st.markdown(
        """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Root variables for consistent theming */
        :root {
            --primary-blue: #3b82f6;
            --primary-purple: #8b5cf6;
            --success-green: #10b981;
            --warning-orange: #f59e0b;
            --danger-red: #ef4444;
            --neutral-gray: #6b7280;
            --dark-gray: #1f2937;
            --light-gray: #f9fafb;
            --white: #ffffff;
            --border-radius: 12px;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        }

        /* Global styles */
        .main {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        /* Hide Streamlit branding and menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container with glassmorphism effect */
        .block-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-xl);
            border: 1px solid rgba(255, 255, 255, 0.2);
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Enhanced header styling */
        h1 {
            background: linear-gradient(135deg, var(--primary-blue), var(--primary-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            font-size: 3rem;
            text-align: center;
            margin-bottom: 0.5rem;
            animation: fadeInUp 0.8s ease-out;
        }

        /* Subtitle styling */
        .main h3 {
            color: var(--neutral-gray);
            text-align: center;
            font-weight: 400;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }

        /* Section headers */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 2rem 0 1.5rem 0;
            padding: 1rem;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
            border-radius: var(--border-radius);
            border-left: 4px solid var(--primary-blue);
        }

        .section-header .icon {
            font-size: 1.5rem;
        }

        .section-header h2 {
            color: var(--dark-gray);
            margin: 0;
            font-weight: 600;
            font-size: 1.5rem;
        }

        /* Modern metric cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-md);
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary-blue), var(--primary-purple));
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .metric-card .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--dark-gray);
            margin: 0.5rem 0;
        }

        .metric-card .metric-label {
            font-size: 0.875rem;
            color: var(--neutral-gray);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* File uploader styling */
        .uploadedFile {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.1));
            border: 2px dashed var(--success-green);
            border-radius: var(--border-radius);
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .uploadedFile:hover {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(34, 197, 94, 0.15));
            transform: scale(1.02);
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.5);
            border-radius: var(--border-radius);
            padding: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: var(--neutral-gray);
            font-weight: 500;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary-blue), var(--primary-purple));
            color: white;
            box-shadow: var(--shadow-md);
        }

        /* Status messages */
        .status-success {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.1));
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #065f46;
            padding: 1rem 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            font-weight: 500;
            animation: slideInRight 0.5s ease-out;
        }

        .status-warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(251, 191, 36, 0.1));
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #92400e;
            padding: 1rem 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            font-weight: 500;
            animation: slideInRight 0.5s ease-out;
        }

        .status-error {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(248, 113, 113, 0.1));
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #991b1b;
            padding: 1rem 1.5rem;
            border-radius: var(--border-radius);
            margin: 1rem 0;
            font-weight: 500;
            animation: slideInRight 0.5s ease-out;
        }

        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(139, 92, 246, 0.05));
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: var(--border-radius);
            padding: 2rem;
            margin: 1.5rem 0;
            position: relative;
        }

        .info-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary-blue), var(--primary-purple));
            border-radius: var(--border-radius) var(--border-radius) 0 0;
        }

        .info-box h3 {
            color: var(--dark-gray);
            margin-top: 0;
            font-weight: 600;
        }

        .info-box h4 {
            color: var(--primary-blue);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .info-box ul {
            color: var(--neutral-gray);
            line-height: 1.6;
        }

        /* Search container */
        .search-container {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-md);
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* Word cloud container */
        .wordcloud-container {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7));
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-md);
            margin: 1rem 0;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-blue), var(--primary-purple));
            color: white;
            border: none;
            border-radius: var(--border-radius);
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 0.875rem;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-lg);
        }

        /* Download button styling */
        .stDownloadButton > button {
            background: linear-gradient(135deg, var(--success-green), #059669);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 0.875rem;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
            width: 100%;
        }

        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-lg);
        }

        /* Input field styling */
        .stTextInput > div > div > input {
            border-radius: var(--border-radius);
            border: 2px solid rgba(59, 130, 246, 0.2);
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            transition: all 0.3s ease;
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--primary-blue);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        /* Select box styling */
        .stSelectbox > div > div {
            border-radius: var(--border-radius);
            border: 2px solid rgba(59, 130, 246, 0.2);
        }

        /* Progress bars */
        .progress-bar {
            background: #e5e7eb;
            border-radius: 1rem;
            height: 8px;
            overflow: hidden;
        }

        .progress-fill {
            background: linear-gradient(90deg, var(--primary-blue), var(--primary-purple));
            height: 100%;
            border-radius: 1rem;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Chart containers */
        .js-plotly-plot {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-md);
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border-radius: var(--border-radius);
            border: 1px solid rgba(255, 255, 255, 0.3);
            font-weight: 500;
        }

        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.7;
            }
        }

        .fade-in {
            animation: fadeInUp 0.6s ease-out;
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        /* Loading spinner */
        .stSpinner > div {
            border-color: var(--primary-blue) transparent var(--primary-blue) transparent;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.85));
            backdrop-filter: blur(20px);
        }

        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem;
                margin: 1rem;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .metric-card {
                padding: 1rem;
            }
            
            .section-header {
                flex-direction: column;
                text-align: center;
                gap: 0.5rem;
            }
        }

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            :root {
                --light-gray: #1f2937;
                --white: #111827;
                --dark-gray: #f9fafb;
                --neutral-gray: #d1d5db;
            }
            
            .main {
                background: linear-gradient(135deg, #1e3a8a 0%, #581c87 100%);
            }
            
            .block-container {
                background: rgba(17, 24, 39, 0.95);
                color: #f9fafb;
            }
            
            .metric-card {
                background: linear-gradient(135deg, rgba(31, 41, 55, 0.9), rgba(17, 24, 39, 0.7));
            }
        }

        /* Accessibility improvements */
        .metric-card:focus,
        .stButton > button:focus,
        .stTextInput > div > div > input:focus {
            outline: 2px solid var(--primary-blue);
            outline-offset: 2px;
        }

        /* High contrast mode */
        @media (prefers-contrast: high) {
            .metric-card {
                border: 2px solid var(--dark-gray);
            }
            
            .status-success {
                border: 2px solid var(--success-green);
            }
            
            .status-warning {
                border: 2px solid var(--warning-orange);
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()


def create_header():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1>💬 Open WebUI Chat Analyzer</h1>
        <h3>Transform your conversation data into actionable insights with beautiful visualizations</h3>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, icon="📊", change=None):
    """Create a modern metric card with enhanced styling"""
    change_html = ""
    if change is not None:
        change_color = "#10b981" if change >= 0 else "#ef4444"
        change_icon = "↗️" if change >= 0 else "↘️"
        change_html = f"""
        <div style="font-size: 0.875rem; color: {change_color}; font-weight: 500; margin-top: 0.5rem; display: flex; align-items: center; gap: 0.25rem;">
            <span>{change_icon}</span>
            <span>{change:+.1f}%</span>
        </div>
        """
    
    return f"""
    <div class="metric-card fade-in">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <span class="metric-label">{title}</span>
        </div>
        <div class="metric-value">{value}</div>
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
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.1)); 
                        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid rgba(16, 185, 129, 0.2);">
                <h3 style="color: #065f46; margin-top: 0;">📊 Chat Data</h3>
                <p style="color: #047857; margin-bottom: 1rem;">Export {len(chats_df)} conversations with metadata</p>
            </div>
            """, unsafe_allow_html=True)
            
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
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1)); 
                        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; border: 1px solid rgba(59, 130, 246, 0.2);">
                <h3 style="color: #1e40af; margin-top: 0;">💬 Message Data</h3>
                <p style="color: #2563eb; margin-bottom: 1rem;">Export {len(messages_df)} individual messages</p>
            </div>
            """, unsafe_allow_html=True)
            
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
        <p style="margin-top: 1rem; font-style: italic; color: #6b7280;">The exported file will contain all your conversations, messages, and metadata.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 What you'll get from this analysis</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;">
            <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(139, 92, 246, 0.05)); 
                        border-radius: 8px; padding: 1.5rem; border: 1px solid rgba(59, 130, 246, 0.1);">
                <h4 style="color: #1d4ed8; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                    📊 Overview Metrics
                </h4>
                <ul style="margin: 0; color: #4b5563; line-height: 1.6;">
                    <li>Total chats, messages, and unique users</li>
                    <li>Average messages per conversation</li>
                    <li>User vs assistant message ratios</li>
                </ul>
            </div>
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(34, 197, 94, 0.05)); 
                        border-radius: 8px; padding: 1.5rem; border: 1px solid rgba(16, 185, 129, 0.1);">
                <h4 style="color: #047857; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                    📈 Time Analysis
                </h4>
                <ul style="margin: 0; color: #4b5563; line-height: 1.6;">
                    <li>Daily message activity trends</li>
                    <li>Activity heatmaps by hour and day</li>
                    <li>Conversation length distributions</li>
                </ul>
            </div>
            <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(168, 85, 247, 0.05)); 
                        border-radius: 8px; padding: 1.5rem; border: 1px solid rgba(139, 92, 246, 0.1);">
                <h4 style="color: #7c3aed; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                    🤖 Model Usage
                </h4>
                <ul style="margin: 0; color: #4b5563; line-height: 1.6;">
                    <li>Distribution of different AI models used</li>
                    <li>Usage statistics and percentages</li>
                    <li>Model performance insights</li>
                </ul>
            </div>
            <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.05), rgba(251, 191, 36, 0.05)); 
                        border-radius: 8px; padding: 1.5rem; border: 1px solid rgba(245, 158, 11, 0.1);">
                <h4 style="color: #d97706; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                    💭 Content Analysis
                </h4>
                <ul style="margin: 0; color: #4b5563; line-height: 1.6;">
                    <li>Word clouds from user messages</li>
                    <li>Message length analysis</li>
                    <li>Content pattern identification</li>
                </ul>
            </div>
            <div style="background: linear-gradient(135deg, rgba(236, 72, 153, 0.05), rgba(244, 114, 182, 0.05)); 
                        border-radius: 8px; padding: 1.5rem; border: 1px solid rgba(236, 72, 153, 0.1);">
                <h4 style="color: #be185d; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                    😊 Sentiment Analysis
                </h4>
                <ul style="margin: 0; color: #4b5563; line-height: 1.6;">
                    <li>Positive, neutral, negative message ratios</li>
                    <li>Sentiment trends over time</li>
                    <li>User satisfaction indicators</li>
                </ul>
            </div>
            <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(124, 58, 237, 0.05)); 
                        border-radius: 8px; padding: 1.5rem; border: 1px solid rgba(99, 102, 241, 0.1);">
                <h4 style="color: #4338ca; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                    🔍 Search & Filter
                </h4>
                <ul style="margin: 0; color: #4b5563; line-height: 1.6;">
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
            # Success message with modern styling
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
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.7)); 
                                        backdrop-filter: blur(10px); border-radius: 12px; padding: 1.5rem; 
                                        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); border: 1px solid rgba(255, 255, 255, 0.3);">
                                <h3 style="color: #1f2937; margin-top: 0;">📊 Model Statistics</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for i, (model, count) in enumerate(model_stats.head(5).items()):
                                percentage = (count / len(messages_df)) * 100
                                # Create modern progress bar
                                st.markdown(f"""
                                <div style="margin: 1rem 0; background: rgba(255, 255, 255, 0.5); 
                                            border-radius: 8px; padding: 1rem;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                        <span style="font-weight: 600; color: #1f2937;">{model}</span>
                                        <span style="color: #6b7280; font-weight: 500;">{count:,} ({percentage:.1f}%)</span>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width: {percentage}%;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="status-warning">
                                <strong>ℹ️ No model information available</strong> in the uploaded data
                            </div>
                            """, unsafe_allow_html=True)
            
            with tab3:
                st.markdown("""
                <div class="section-header">
                    <span class="icon">💭</span>
                    <h2>Content Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Word cloud with enhanced container
                wordcloud = generate_word_cloud(messages_df)
                if wordcloud:
                    st.markdown('<div class="wordcloud-container">', unsafe_allow_html=True)
                    st.markdown("### ☁️ Word Cloud (User Messages)")
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
                        with st.container():
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