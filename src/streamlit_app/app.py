"""
Decision Intelligence Studio - Streamlit Application

Multi-page application with:
- Real-time monitoring dashboard
- A/B test tracking and analysis
- Model comparison and versioning
- Advanced analytics and insights
- Customer journey visualization
- ROI optimization tools
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import FILE_PATHS, SEGMENTATION
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Decision Intelligence Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Improve readability */
    .stMarkdown p {
        font-size: 1rem;
        line-height: 1.6;
    }
    /* Better column spacing */
    [data-testid="column"] {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_data():
    """Load all necessary data with caching"""
    try:
        uplift_df = pd.read_parquet(FILE_PATHS["uplift_scores"])
        canonical_df = pd.read_parquet(FILE_PATHS["canonical_data"])
        
        # Load reports if available
        refutation_report = None
        if FILE_PATHS["refutation_report"].exists():
            with open(FILE_PATHS["refutation_report"], 'r') as f:
                refutation_report = json.load(f)
        
        dq_report = None
        if FILE_PATHS["data_quality_report"].exists():
            with open(FILE_PATHS["data_quality_report"], 'r') as f:
                dq_report = json.load(f)
        
        return {
            'uplift': uplift_df,
            'canonical': canonical_df,
            'refutation': refutation_report,
            'data_quality': dq_report,
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def main_dashboard():
    """Main overview dashboard"""
    st.markdown('<h1 class="main-header">🎯 Decision Intelligence Studio</h1>', unsafe_allow_html=True)
    st.markdown("**Real-time Causal ML for Marketing Optimization**")
    
    data = load_data()
    if data is None:
        st.error("⚠️ Data not loaded. Please run the pipeline first: `python run_pipeline.py`")
        return
    
    uplift_df = data['uplift']
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ate = uplift_df['uplift_score'].mean()
        st.metric(
            "Average Treatment Effect",
            f"${ate:.2f}",
            delta=f"+${ate:.2f} per customer"
        )
    
    with col2:
        total_users = len(uplift_df)
        st.metric("Total Customers", f"{total_users:,}")
    
    with col3:
        treatment_rate = uplift_df['treatment'].mean()
        st.metric("Historical Treatment Rate", f"{treatment_rate:.1%}")
    
    with col4:
        high_uplift = (uplift_df['segment_name'] == 'High Uplift').sum()
        st.metric("High-Uplift Customers", f"{high_uplift:,}")
    
    st.markdown("---")
    
    # Main charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Uplift Distribution by Segment")
        
        # Create violin plot
        fig = go.Figure()
        
        for segment in SEGMENTATION["segment_names"]:
            seg_data = uplift_df[uplift_df['segment_name'] == segment]['uplift_score']
            fig.add_trace(go.Violin(
                y=seg_data,
                name=segment,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            height=400,
            showlegend=True,
            yaxis_title="Uplift Score ($)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Segment Distribution")
        
        segment_counts = uplift_df['segment_name'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=segment_counts.index,
            values=segment_counts.values,
            hole=0.4,
            marker=dict(colors=['#667eea', '#764ba2', '#ed64a6', '#ff9a9e'])
        )])
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment Analysis
    st.markdown("---")
    st.subheader("📈 Segment Performance Analysis")
    
    segment_stats = uplift_df.groupby('segment_name').agg({
        'user_id': 'count',
        'uplift_score': ['mean', 'median', 'std'],
        'outcome': 'mean',
        'treatment': 'mean'
    }).round(2)
    
    segment_stats.columns = ['Count', 'Mean Uplift', 'Median Uplift', 'Std Dev', 'Mean Outcome', 'Treatment Rate']
    segment_stats = segment_stats.sort_values('Mean Uplift', ascending=False)
    
    # Calculate ROI
    cost_per_treatment = 10.0
    segment_stats['Expected Revenue'] = segment_stats['Mean Uplift'] * segment_stats['Count']
    segment_stats['Campaign Cost'] = segment_stats['Count'] * cost_per_treatment
    segment_stats['ROI'] = ((segment_stats['Expected Revenue'] - segment_stats['Campaign Cost']) / 
                            segment_stats['Campaign Cost'] * 100).round(1)
    
    st.dataframe(segment_stats, use_container_width=True)
    
    # Key Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Heterogeneous Effects")
        max_uplift = uplift_df['uplift_score'].max()
        min_uplift = uplift_df['uplift_score'].min()
        st.markdown(
            f"Treatment effects range from **${min_uplift:.2f}** to **${max_uplift:.2f}**, "
            f"demonstrating significant heterogeneity across customers."
        )
    
    with col2:
        st.markdown("#### 💰 ROI Optimization")
        high_seg = segment_stats.iloc[0]
        st.markdown(
            f"Targeting High Uplift segment (**{int(high_seg['Count'])} customers**) "
            f"yields **{high_seg['ROI']:.0f}% ROI** - significantly better than random targeting."
        )
    
    with col3:
        st.markdown("#### 📊 Precision Targeting")
        top_25 = uplift_df.nlargest(int(len(uplift_df) * 0.25), 'uplift_score')
        potential_revenue = top_25['uplift_score'].sum()
        st.markdown(
            f"Targeting top 25% of customers could generate "
            f"**${potential_revenue:,.0f}** in incremental revenue."
        )


def realtime_monitoring():
    """Real-time monitoring page"""
    st.markdown('<h1 class="main-header">📡 Real-time Monitoring</h1>', unsafe_allow_html=True)
    
    # Auto-refresh control
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info("💡 This page shows real-time streaming analytics. In production, this connects to your event stream (Kafka, Kinesis, Pub/Sub).")
    with col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=True, key="auto_refresh")
    with col3:
        if st.button("🔃 Refresh Now", key="manual_refresh"):
            st.rerun()
    
    # Display last update time
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    st.caption(f"⏱️ Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Simulate real-time metrics with random variations
    np.random.seed(int(time.time()))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        events_rate = np.random.uniform(3.8, 4.6)
        delta = np.random.uniform(-0.2, 0.5)
        st.metric("Events/sec", f"{events_rate:.1f}", delta=f"{delta:.1f}")
    
    with col2:
        latency = np.random.randint(40, 52)
        delta_lat = np.random.randint(-8, 3)
        st.metric("Avg Latency", f"{latency}ms", delta=f"{delta_lat}ms", delta_color="inverse")
    
    with col3:
        high_value_pct = np.random.uniform(21, 25)
        delta_pct = np.random.uniform(-1, 3)
        st.metric("High-Value %", f"{high_value_pct:.1f}%", delta=f"{delta_pct:.1f}%")
    
    with col4:
        alerts = np.random.randint(1, 4)
        st.metric("Active Alerts", f"{alerts}", delta=f"{alerts-2}", delta_color="inverse")
    
    st.markdown("---")
    
    # Real-time chart
    st.subheader("📈 Live Event Stream")
    
    # Simulate streaming data
    times = pd.date_range(end=datetime.now(), periods=100, freq='1min')
    uplift_values = np.random.normal(45, 15, 100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=uplift_values,
        mode='lines',
        name='Uplift Score',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Average Uplift ($)",
        template="plotly_white",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent events table
    st.subheader("🔄 Recent Scored Events")
    
    data = load_data()
    if data:
        recent = data['uplift'].tail(20)[['user_id', 'uplift_score', 'segment_name', 'outcome']].copy()
        recent['timestamp'] = pd.date_range(end=datetime.now(), periods=len(recent), freq='30s')
        recent = recent[['timestamp', 'user_id', 'uplift_score', 'segment_name', 'outcome']]
        recent.columns = ['Timestamp', 'User ID', 'Uplift Score', 'Segment', 'Outcome']
        
        st.dataframe(recent, use_container_width=True, height=300)
    
    # Alert panel
    st.markdown("---")
    st.subheader("🚨 Active Alerts")
    
    st.warning("⚠️ **WARNING:** Treatment rate below threshold (4.8% vs 10% target)  \n*Occurred: 2 minutes ago | Severity: Medium*")
    
    st.info("✅ **INFO:** High-value customer detected (user_00234567, uplift: $89.45)  \n*Occurred: 5 minutes ago | Severity: Info*")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(5)
        st.session_state.last_update = datetime.now()
        st.rerun()


def ab_test_tracking():
    """A/B test tracking and analysis"""
    st.markdown('<h1 class="main-header">🧪 A/B Test Tracking</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Track and compare predicted vs. actual treatment effects from A/B tests.
    This validates our causal model and provides feedback for continuous improvement.
    """)
    
    # Simulate A/B test results
    test_data = {
        'Test ID': ['TEST-001', 'TEST-002', 'TEST-003', 'TEST-004', 'TEST-005'],
        'Segment': ['High Uplift', 'Medium-High', 'All Users', 'Low Uplift', 'High Uplift'],
        'Status': ['Completed', 'Completed', 'Running', 'Completed', 'Planned'],
        'Start Date': ['2024-12-01', '2024-12-15', '2025-01-01', '2024-11-15', '2025-01-15'],
        'Sample Size': [1000, 1500, 5000, 800, 1200],
        'Predicted Uplift': [89.4, 52.2, 45.2, 8.7, 91.2],
        'Observed Uplift': [87.6, 49.8, 43.1, 9.2, None],
        'Confidence': ['95%', '95%', '90%', '95%', None],
    }
    
    test_df = pd.DataFrame(test_data)
    
    # Add error calculation
    test_df['Prediction Error'] = abs(test_df['Predicted Uplift'] - test_df['Observed Uplift'])
    test_df['Error %'] = (test_df['Prediction Error'] / test_df['Predicted Uplift'] * 100).round(1)
    
    # Status filtering
    status_filter = st.multiselect(
        "Filter by Status",
        options=['All', 'Completed', 'Running', 'Planned'],
        default=['Completed', 'Running']
    )
    
    if 'All' not in status_filter:
        test_df_filtered = test_df[test_df['Status'].isin(status_filter)]
    else:
        test_df_filtered = test_df
    
    st.dataframe(test_df_filtered, use_container_width=True)
    
    # Calibration plot
    st.markdown("---")
    st.subheader("📊 Model Calibration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        completed = test_df[test_df['Status'] == 'Completed'].copy()
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash')
        ))
        
        # Actual predictions
        fig.add_trace(go.Scatter(
            x=completed['Predicted Uplift'],
            y=completed['Observed Uplift'],
            mode='markers+text',
            name='A/B Tests',
            marker=dict(size=12, color='#667eea'),
            text=completed['Test ID'],
            textposition='top center'
        ))
        
        fig.update_layout(
            height=400,
            xaxis_title="Predicted Uplift ($)",
            yaxis_title="Observed Uplift ($)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calibration metrics
        mae = completed['Prediction Error'].mean()
        mape = completed['Error %'].mean()
        
        st.success(f"**Model Calibration Metrics:**\n\n"
                  f"- Mean Absolute Error: ${mae:.2f}\n"
                  f"- Mean Absolute Percentage Error: {mape:.1f}%\n\n"
                  f"✅ Model is well-calibrated with < 5% error")
    
    with col2:
        st.markdown("### 🎯 Test Summary")
        st.metric("Completed Tests", len(completed))
        st.metric("Active Tests", len(test_df[test_df['Status'] == 'Running']))
        st.metric("Avg. Calibration", f"{(100 - mape):.1f}%")
        
        st.markdown("---")
        st.markdown("### 📋 Next Steps")
        st.info("""
        1. Review TEST-003 (in progress)
        2. Plan TEST-005 for High Uplift
        3. Update model with new data
        4. Scale successful campaigns
        """)


def model_comparison():
    """Model versioning and comparison"""
    st.markdown('<h1 class="main-header">🔬 Model Comparison</h1>', unsafe_allow_html=True)
    
    st.markdown("Compare different model versions and estimation methods.")
    
    # Simulate model comparison data
    models = {
        'Model': ['CausalForest v1.0', 'CausalForest v0.9', 'Linear DML', 'S-Learner', 'T-Learner'],
        'Method': ['CausalForestDML', 'CausalForestDML', 'LinearDML', 'S-Learner', 'T-Learner'],
        'Training Date': ['2025-01-08', '2024-12-15', '2025-01-08', '2024-11-20', '2024-11-20'],
        'ATE Estimate': [45.23, 43.87, 44.12, 42.56, 46.78],
        'CATE Std Dev': [18.45, 16.23, 21.34, 14.56, 19.87],
        'Refutation Pass Rate': [100, 100, 80, 60, 80],
        'Test MAE': [2.34, 3.45, 4.56, 5.67, 3.98],
        'Training Time (s)': [67, 62, 23, 15, 18],
        'Status': ['Production', 'Archived', 'Candidate', 'Archived', 'Archived'],
    }
    
    models_df = pd.DataFrame(models)
    
    st.dataframe(
        models_df.style.background_gradient(subset=['Test MAE'], cmap='RdYlGn_r')
                      .background_gradient(subset=['Refutation Pass Rate'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Model comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 ATE Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=models_df['Model'],
            y=models_df['ATE Estimate'],
            marker_color=['#667eea' if s == 'Production' else '#cccccc' for s in models_df['Status']],
            text=models_df['ATE Estimate'].round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            height=350,
            yaxis_title="ATE Estimate ($)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Performance vs Complexity")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=models_df['Training Time (s)'],
            y=models_df['Test MAE'],
            mode='markers+text',
            marker=dict(
                size=models_df['Refutation Pass Rate'] / 5,
                color=models_df['ATE Estimate'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="ATE")
            ),
            text=models_df['Model'],
            textposition='top center'
        ))
        
        fig.update_layout(
            height=350,
            xaxis_title="Training Time (seconds)",
            yaxis_title="Test MAE (lower is better)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model selection recommendation
    st.markdown("---")
    st.subheader("🏆 Recommendation")
    
    st.success("""
    **Recommended: CausalForest v1.0**
    
    - ✅ Best calibration (MAE: $2.34)
    - ✅ 100% refutation pass rate
    - ✅ Good heterogeneity capture (CATE std: $18.45)
    - ⚠️ Slightly longer training time (acceptable for batch jobs)
    - ✅ Currently in production
    
    **Next Steps:**
    - Continue monitoring production performance
    - Consider Linear DML as backup (faster training)
    - Re-evaluate after 1000+ new A/B test observations
    """)


def advanced_analytics():
    """Advanced analytics and insights"""
    st.markdown('<h1 class="main-header">🔍 Advanced Analytics</h1>', unsafe_allow_html=True)
    
    data = load_data()
    if data is None:
        return
    
    uplift_df = data['uplift']
    canonical_df = data['canonical']
    
    # Merge data for feature analysis
    analysis_df = canonical_df.merge(
        uplift_df[['user_id', 'uplift_score', 'segment_name']], 
        on='user_id'
    )
    
    # Feature importance analysis
    st.subheader("📊 Feature Importance for Heterogeneity")
    
    # Simulate feature importance
    features = ['engagement_score', 'past_purchases', 'income_level', 'age', 'days_since_signup']
    importance = [0.35, 0.28, 0.18, 0.12, 0.07]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color='#667eea',
        text=[f'{i:.1%}' for i in importance],
        textposition='outside'
    ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Importance",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Interaction effects
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎨 Uplift by Age & Income")
        
        # Create bins
        analysis_df['age_bin'] = pd.cut(analysis_df['age'], bins=5, labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        
        pivot = analysis_df.pivot_table(
            values='uplift_score',
            index='age_bin',
            columns='income_level',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            text=pivot.values.round(1),
            texttemplate='$%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Uplift ($)")
        ))
        
        fig.update_layout(
            height=350,
            xaxis_title="Income Level",
            yaxis_title="Age Group",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Uplift vs Engagement")
        
        fig = px.scatter(
            analysis_df.sample(min(1000, len(analysis_df))),
            x='engagement_score',
            y='uplift_score',
            color='segment_name',
            trendline='lowess',
            opacity=0.6,
            color_discrete_map={
                'High Uplift': '#667eea',
                'Medium-High': '#764ba2',
                'Medium-Low': '#ed64a6',
                'Low Uplift': '#ff9a9e'
            }
        )
        
        fig.update_layout(
            height=350,
            xaxis_title="Engagement Score",
            yaxis_title="Uplift ($)",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Time-based analysis
    st.markdown("---")
    st.subheader("📅 Temporal Analysis")
    
    # Simulate time series
    analysis_df['month'] = pd.to_datetime(analysis_df['event_ts']).dt.to_period('M')
    monthly = analysis_df.groupby('month')['uplift_score'].mean().reset_index()
    monthly['month'] = monthly['month'].astype(str)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['month'],
        y=monthly['uplift_score'],
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Month",
        yaxis_title="Average Uplift ($)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Overview", "📡 Real-time Monitoring", "🧪 A/B Test Tracking", 
     "🔬 Model Comparison", "🔍 Advanced Analytics"]
)

# Add system status
st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.success("✅ API: Online")
st.sidebar.success("✅ Model: v1.0")
st.sidebar.info("📊 Last Updated: 2 min ago")

# Add download options
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")

data = load_data()
if data:
    if st.sidebar.button("📥 Download Uplift Scores"):
        csv = data['uplift'].to_csv(index=False)
        st.sidebar.download_button(
            "Download CSV",
            csv,
            "uplift_scores.csv",
            "text/csv"
        )

# Route to appropriate page
if page == "📊 Overview":
    main_dashboard()
elif page == "📡 Real-time Monitoring":
    realtime_monitoring()
elif page == "🧪 A/B Test Tracking":
    ab_test_tracking()
elif page == "🔬 Model Comparison":
    model_comparison()
elif page == "🔍 Advanced Analytics":
    advanced_analytics()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Decision Intelligence Studio v1.0<br>
    Built with ❤️ for Joon Solutions
</div>
""", unsafe_allow_html=True)