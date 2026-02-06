import streamlit as st
import plotly.graph_objects as go

def metric_card(title, value, delta=None, delta_color="normal", help_text=None):
    """Renders a styled metric card."""
    with st.container():
        st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #666;">{title}</div>
                <div style="font-size: 1.8rem; font-weight: bold; margin: 10px 0;">{value}</div>
            </div>
        """, unsafe_allow_html=True)
        if delta:
            # Note: Custom delta display since we are inside a DIV
            color = "#27ae60" if (delta_color == "normal" and "+" in delta) or (delta_color == "inverse" and "-" in delta) else "#e74c3c"
            st.markdown(f"<div style='text-align: center; color: {color}; margin-top: -15px;'>{delta} vs any ant.</div>", unsafe_allow_html=True)

def comparison_chart(labels, current_values, prev_values, current_label, prev_label):
    """Creates a standard comparison bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=prev_values, name=prev_label, marker_color='#BDC3C7'))
    fig.add_trace(go.Bar(x=labels, y=current_values, name=current_label, marker_color='#3498DB'))
    
    fig.update_layout(
        barmode='group', 
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
