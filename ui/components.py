import streamlit as st
import pandas as pd
from typing import Dict, List


def priority_badge(priority: str, size: str = "medium"):
    """
    Display priority badge with color coding
    
    Args:
        priority: P1, P2, or P3
        size: small, medium, large
    """
    colors = {
        'P1': '#FF4B4B',  # Red
        'P2': '#FFA500',  # Orange
        'P3': '#4CAF50'   # Green
    }
    
    labels = {
        'P1': 'HIGH',
        'P2': 'MEDIUM',
        'P3': 'LOW'
    }
    
    sizes = {
        'small': '0.8em',
        'medium': '1em',
        'large': '1.2em'
    }
    
    color = colors.get(priority, '#999')
    label = labels.get(priority, priority)
    font_size = sizes.get(size, '1em')
    
    st.markdown(
        f"""
        <span style="
            background-color: {color};
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: bold;
            font-size: {font_size};
        ">{label}</span>
        """,
        unsafe_allow_html=True
    )


def confidence_meter(confidence: float, label: str = "Confidence"):
    """
    Display confidence score with visual meter
    
    Args:
        confidence: 0.0 to 1.0
        label: Label text
    """
    percentage = confidence * 100
    
    # Color based on confidence
    if confidence >= 0.85:
        color = '#4CAF50'  # Green
        status = 'High'
    elif confidence >= 0.65:
        color = '#FFA500'  # Orange
        status = 'Medium'
    else:
        color = '#FF4B4B'  # Red
        status = 'Low'
    
    st.markdown(f"**{label}:** {status} ({percentage:.1f}%)")
    st.progress(confidence)


def metric_card(title: str, value: str, delta: str = None, icon: str = "📊"):
    """
    Display metric card
    
    Args:
        title: Metric title
        value: Main value
        delta: Change indicator (optional)
        icon: Emoji icon
    """
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"<div style='font-size: 2em;'>{icon}</div>", unsafe_allow_html=True)
    with col2:
        st.metric(label=title, value=value, delta=delta)


def explanation_panel(explanation: str, title: str = "Explanation"):
    """
    Display explanation in expandable panel
    
    Args:
        explanation: Markdown-formatted explanation
        title: Panel title
    """
    with st.expander(f"ℹ️ {title}", expanded=False):
        st.markdown(explanation)


def issue_card(issue: Dict, show_details: bool = True):
    """
    Display issue in card format
    
    Args:
        issue: Issue dictionary
        show_details: Show full details vs summary
    """
    with st.container():
        st.markdown("---")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{issue.get('issue_id', 'N/A')}**")
            st.caption(issue.get('issue_text', 'No description')[:100] + "...")
        
        with col2:
            priority_badge(issue.get('priority_label', 'P3'), size='small')
        
        with col3:
            category = issue.get('category_label', 'Unknown')
            st.markdown(f" {category}")
        
        if show_details:
            with st.expander("View Details"):
                st.write(f"**Status:** {issue.get('issue_status', 'Unknown')}")
                st.write(f"**Timestamp:** {issue.get('timestamp', 'N/A')}")
                st.write(f"**Full Text:** {issue.get('issue_text', 'N/A')}")


def comparison_table(ml_result: Dict, rule_result: Dict):
    """
    Display ML vs Rules comparison table
    
    Args:
        ml_result: ML prediction results
        rule_result: Rule engine results
    """
    comparison_data = {
        'Aspect': ['Category', 'Priority', 'Confidence/Score'],
        'ML Prediction': [
            ml_result.get('category', 'N/A'),
            ml_result.get('priority', 'N/A'),
            f"{ml_result.get('priority_confidence', 0):.1%}"
        ],
        'Rule-Based': [
            ml_result.get('category', 'N/A'),  # Rules don't predict category
            rule_result.get('rule_priority', 'N/A'),
            f"{rule_result.get('combined_score', 0):.2f}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def action_recommendations(priority: str, category: str, load_info: Dict = None):
    """
    Display recommended actions based on priority
    
    Args:
        priority: P1, P2, or P3
        category: Issue category
        load_info: Department load information
    """
    st.markdown("### Recommended Actions")
    
    if priority == 'P1':
        st.error("**URGENT - Immediate Action Required**")
        st.markdown("""
        -  **Assign to specialist immediately**
        -  **Contact user for details**
        -  **Escalate to department head if needed**
        -  **Target resolution: Within 2 hours**
        """)
    elif priority == 'P2':
        st.warning("**Medium Priority - Address Soon**")
        st.markdown("""
        -  **Add to today's work queue**
        -  **Assign to available team member**
        -  **Target resolution: Within 24 hours**
        """)
    else:
        st.info("**Low Priority - Standard Queue**")
        st.markdown("""
        -  **Add to backlog**
        -  **Process in normal workflow**
        -  **Target resolution: Within 48 hours**
        """)
    
    if load_info and category in load_info:
        load = load_info[category]
        if load['load_percentage'] > 80:
            st.warning(f"⚠️ {category} team at {load['load_percentage']:.0f}% capacity - consider reassignment")
