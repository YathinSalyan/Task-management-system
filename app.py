import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import datetime
import uuid
import os

# Download necessary NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Set page configuration
st.set_page_config(
    page_title="AI Task Manager",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("AI-Powered Task Management System")
st.markdown("""
This application helps you manage your tasks with AI-powered sentiment analysis. 
The system analyzes the sentiment of your task descriptions and comments to help you 
understand the emotional context of your work.
""")

# Try to load the sentiment model
@st.cache_resource
def load_sentiment_model():
    try:
        # First try to load from file
        if os.path.exists("tweet_sentiment_model.pkl"):
            with open("tweet_sentiment_model.pkl", "rb") as f:
                return pickle.load(f)
        else:
            # If file doesn't exist, create a placeholder message
            st.warning("Sentiment model file not found. Some AI features won't be available.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

sentiment_model = load_sentiment_model()

# Create a class to handle tasks if no model is available
class SimpleSentimentAnalyzer:
    def predict(self, text):
        if not text:
            return "Neutral"
        
        positive_words = ["good", "great", "excellent", "happy", "excited", "important", "success", "achieve"]
        negative_words = ["bad", "terrible", "sad", "angry", "difficult", "challenging", "issue", "problem"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "Positive"
        elif neg_count > pos_count:
            return "Negative"
        else:
            return "Neutral"

# Initialize simple sentiment analyzer if model isn't available
if not sentiment_model:
    sentiment_model = SimpleSentimentAnalyzer()

# Function to get sentiment class for styling
def get_sentiment_class(sentiment):
    if "positive" in sentiment.lower():
        return "positive"
    elif "negative" in sentiment.lower():
        return "negative"
    else:
        return "neutral"

# Initialize session state for tasks if it doesn't exist
if 'tasks' not in st.session_state:
    st.session_state.tasks = []

if 'filter' not in st.session_state:
    st.session_state.filter = "All"

if 'sort_by' not in st.session_state:
    st.session_state.sort_by = "Due Date"

# Sidebar for app controls
with st.sidebar:
    st.header("Controls")
    
    # Filter options
    st.subheader("Filter Tasks")
    filter_option = st.radio(
        "Filter by status:",
        ("All", "Pending", "Completed")
    )
    st.session_state.filter = filter_option
    
    # Sort options
    st.subheader("Sort Tasks")
    sort_option = st.radio(
        "Sort by:",
        ("Due Date", "Priority", "Sentiment")
    )
    st.session_state.sort_by = sort_option
    
    # Add sample tasks button
    if st.button("Load Sample Tasks"):
        st.session_state.tasks = [
            {
                "id": str(uuid.uuid4()),
                "title": "Complete project proposal",
                "description": "I'm excited to work on this new innovative project!",
                "priority": "High",
                "status": "Pending",
                "due_date": datetime.datetime.now() + datetime.timedelta(days=2),
                "created_at": datetime.datetime.now(),
                "sentiment": sentiment_model.predict("I'm excited to work on this new innovative project!"),
                "comments": []
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Fix server bug",
                "description": "This is a frustrating issue causing system failures.",
                "priority": "Critical",
                "status": "Pending",
                "due_date": datetime.datetime.now() + datetime.timedelta(days=1),
                "created_at": datetime.datetime.now(),
                "sentiment": sentiment_model.predict("This is a frustrating issue causing system failures."),
                "comments": []
            },
            {
                "id": str(uuid.uuid4()),
                "title": "Weekly team meeting",
                "description": "Regular sync-up with the team on project progress.",
                "priority": "Medium",
                "status": "Pending",
                "due_date": datetime.datetime.now() + datetime.timedelta(days=3),
                "created_at": datetime.datetime.now(),
                "sentiment": sentiment_model.predict("Regular sync-up with the team on project progress."),
                "comments": []
            }
        ]
        st.success("Sample tasks loaded!")

# Add CSS for task cards with sentiment colors
st.markdown("""
<style>
    .task-card {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .positive {
        border-left: 5px solid #28a745;
    }
    .negative {
        border-left: 5px solid #dc3545;
    }
    .neutral {
        border-left: 5px solid #17a2b8;
    }
    .completed {
        opacity: 0.7;
        background-color: #f8f9fa;
    }
    .badge {
        padding: 5px 10px;
        border-radius: 10px;
        font-size: 0.8em;
        color: white;
        display: inline-block;
        margin-right: 5px;
    }
    .high {
        background-color: #dc3545;
    }
    .medium {
        background-color: #ffc107;
        color: black;
    }
    .low {
        background-color: #28a745;
    }
    .critical {
        background-color: #6610f2;
    }
    .comment-box {
        background-color: #f8f9fa;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        border-left: 3px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Create a two-column layout
col1, col2 = st.columns([2, 3])

# Form to add new tasks
with col1:
    with st.container():
        st.subheader("Add New Task")
        with st.form(key="add_task_form"):
            task_title = st.text_input("Task Title")
            task_description = st.text_area("Task Description")
            
            col_a, col_b = st.columns(2)
            with col_a:
                task_priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
            with col_b:
                task_due_date = st.date_input("Due Date", min_value=datetime.datetime.now().date())
            
            submit_button = st.form_submit_button(label="Add Task")
            
            if submit_button:
                if task_title:
                    # Analyze sentiment of task description
                    sentiment = sentiment_model.predict(task_description)
                    
                    # Create new task
                    new_task = {
                        "id": str(uuid.uuid4()),
                        "title": task_title,
                        "description": task_description,
                        "priority": task_priority,
                        "status": "Pending",
                        "due_date": datetime.datetime.combine(task_due_date, datetime.datetime.min.time()),
                        "created_at": datetime.datetime.now(),
                        "sentiment": sentiment,
                        "comments": []
                    }
                    
                    st.session_state.tasks.append(new_task)
                    st.success(f"Task '{task_title}' added successfully!")
                else:
                    st.error("Task title is required!")
    
    # Task statistics
    st.subheader("Task Statistics")
    with st.container():
        if st.session_state.tasks:
            total_tasks = len(st.session_state.tasks)
            completed_tasks = sum(1 for task in st.session_state.tasks if task["status"] == "Completed")
            pending_tasks = total_tasks - completed_tasks
            
            # Sentiment stats
            sentiments = [task["sentiment"].lower() for task in st.session_state.tasks]
            positive_tasks = sum("positive" in s for s in sentiments)
            negative_tasks = sum("negative" in s for s in sentiments)
            neutral_tasks = total_tasks - positive_tasks - negative_tasks
            
            # Create metrics display
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            with col_metrics1:
                st.metric("Total Tasks", total_tasks)
            with col_metrics2:
                st.metric("Completed", completed_tasks)
            with col_metrics3:
                st.metric("Pending", pending_tasks)
            
            # Create a simple chart for sentiment distribution
            sentiment_data = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [positive_tasks, neutral_tasks, negative_tasks]
            })
            st.bar_chart(sentiment_data.set_index("Sentiment"))
            
            # Priority distribution
            priorities = [task["priority"] for task in st.session_state.tasks]
            priority_counts = pd.DataFrame({
                "Priority": ["Critical", "High", "Medium", "Low"],
                "Count": [
                    sum(1 for p in priorities if p == "Critical"),
                    sum(1 for p in priorities if p == "High"),
                    sum(1 for p in priorities if p == "Medium"),
                    sum(1 for p in priorities if p == "Low")
                ]
            })
            st.bar_chart(priority_counts.set_index("Priority"))
            
        else:
            st.info("No tasks added yet. Add a task to see statistics.")

# Task List
with col2:
    st.subheader("Task List")
    
    # Filter tasks
    filtered_tasks = st.session_state.tasks
    if st.session_state.filter != "All":
        filtered_tasks = [task for task in filtered_tasks if task["status"] == st.session_state.filter]
    
    # Sort tasks
    if st.session_state.sort_by == "Due Date":
        filtered_tasks = sorted(filtered_tasks, key=lambda x: x["due_date"])
    elif st.session_state.sort_by == "Priority":
        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        filtered_tasks = sorted(filtered_tasks, key=lambda x: priority_order[x["priority"]])
    elif st.session_state.sort_by == "Sentiment":
        sentiment_order = {"negative": 0, "neutral": 1, "positive": 2}
        filtered_tasks = sorted(filtered_tasks, key=lambda x: sentiment_order.get(get_sentiment_class(x["sentiment"]), 1))
    
    if not filtered_tasks:
        st.info(f"No {st.session_state.filter.lower()} tasks found.")
    else:
        for index, task in enumerate(filtered_tasks):
            sentiment_class = get_sentiment_class(task["sentiment"])
            completed_class = " completed" if task["status"] == "Completed" else ""
            
            # Create task card
            with st.container():
                st.markdown(f"""
                <div class="task-card {sentiment_class}{completed_class}">
                    <h3>{task["title"]}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col_task1, col_task2 = st.columns([3, 1])
                
                with col_task1:
                    st.markdown(f"**Description:** {task['description']}")
                    st.markdown(f"**Due Date:** {task['due_date'].strftime('%Y-%m-%d')}")
                    st.markdown(f"""
                    <span class="badge {task['priority'].lower()}">{task['priority']}</span>
                    <span class="badge {sentiment_class}">{task['sentiment']}</span>
                    """, unsafe_allow_html=True)
                
                with col_task2:
                    # Task actions
                    task_status = "Mark Pending" if task["status"] == "Completed" else "Mark Complete"
                    if st.button(task_status, key=f"status_{task['id']}"):
                        for t in st.session_state.tasks:
                            if t["id"] == task["id"]:
                                t["status"] = "Pending" if t["status"] == "Completed" else "Completed"
                        st.experimental_rerun()
                    
                    if st.button("Delete", key=f"delete_{task['id']}"):
                        st.session_state.tasks = [t for t in st.session_state.tasks if t["id"] != task["id"]]
                        st.experimental_rerun()
                
                # Comment section
                with st.expander("Comments"):
                    # Display existing comments
                    for comment in task["comments"]:
                        sentiment_class = get_sentiment_class(comment["sentiment"])
                        st.markdown(f"""
                        <div class="comment-box {sentiment_class}">
                            <p>{comment["text"]}</p>
                            <small>{comment["timestamp"].strftime('%Y-%m-%d %H:%M')} · {comment["sentiment"]}</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add new comment
                    new_comment = st.text_area("Add a comment", key=f"comment_input_{task['id']}")
                    if st.button("Post Comment", key=f"add_comment_{task['id']}"):
                        if new_comment:
                            # Analyze sentiment of comment
                            comment_sentiment = sentiment_model.predict(new_comment)
                            
                            # Add comment to task
                            for t in st.session_state.tasks:
                                if t["id"] == task["id"]:
                                    t["comments"].append({
                                        "text": new_comment,
                                        "timestamp": datetime.datetime.now(),
                                        "sentiment": comment_sentiment
                                    })
                            st.success("Comment added!")
                            st.experimental_rerun()
                
                st.markdown("---")

# Download tasks as CSV
st.sidebar.subheader("Export Data")
if st.sidebar.button("Export Tasks as CSV"):
    if st.session_state.tasks:
        # Convert tasks to DataFrame
        tasks_data = []
        for task in st.session_state.tasks:
            task_data = {
                "title": task["title"],
                "description": task["description"],
                "priority": task["priority"],
                "status": task["status"],
                "due_date": task["due_date"].strftime('%Y-%m-%d'),
                "created_at": task["created_at"].strftime('%Y-%m-%d %H:%M'),
                "sentiment": task["sentiment"],
                "comment_count": len(task["comments"])
            }
            tasks_data.append(task_data)
        
        df = pd.DataFrame(tasks_data)
        
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="tasks_export.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.info("No tasks to export.")

# About section in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This AI-powered task management system uses sentiment analysis to help you understand the emotional context of your tasks and comments.
    
    **Features:**
    - Organize tasks with priorities and due dates
    - AI-powered sentiment analysis
    - Track task status and comments
    - Visual statistics and data export
    
    Built with Streamlit and NLTK.
    """)