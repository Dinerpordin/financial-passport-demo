import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="💳 Financial Passport Demo",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .score-excellent { color: #10b981; }
    .score-good { color: #3b82f6; }
    .score-fair { color: #f59e0b; }
    .score-poor { color: #ef4444; }
    .info-box {
        padding: 1rem;
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Function 1: Generate Sample Transactions ---
def generate_sample_transactions(phone_number):
    seed = abs(hash(phone_number)) % (2**32)
    np.random.seed(seed)
    
    tx_types = ['Cash-In', 'Cash-Out', 'Send Money', 'Payment', 'Mobile Recharge']
    tx_weights = [0.35, 0.35, 0.12, 0.10, 0.08]

    n_tx = 500
    today = datetime.datetime.today()
    start_date = today - datetime.timedelta(days=180)
    dates = np.random.choice(pd.date_range(start_date, today), n_tx)
    dates.sort()
    
    tx_type = np.random.choice(tx_types, size=n_tx, p=tx_weights)
    amounts = []
    balance = []
    bal = np.random.randint(500, 10000)

    for tt in tx_type:
        if tt == 'Cash-In':
            amt = np.random.randint(500, 20000)
            bal += amt
        elif tt == 'Cash-Out':
            amt = np.random.randint(100, 15000)
            bal -= amt
        elif tt == 'Send Money':
            amt = np.random.randint(200, 7000)
            bal -= amt
        elif tt == 'Payment':
            amt = np.random.randint(50, 5000)
            bal -= amt
        else:
            amt = np.random.randint(10, 200)
            bal -= amt
        amounts.append(amt)
        balance.append(max(bal, 0))

    df = pd.DataFrame({
        'date': dates,
        'type': tx_type,
        'amount': amounts,
        'balance': balance,
    })
    return df.sort_values('date')

# --- Function 2: Calculate Credit Score ---
def calculate_credit_score(transaction_df):
    df = transaction_df.copy()
    df['month'] = df['date'].dt.to_period('M')

    avg_balance = df.groupby('month')['balance'].mean().mean()
    income_stability = (
        df[df['type'].isin(['Cash-In', 'Send Money'])]['amount'].sum()
        / df['amount'].sum()
    )
    tx_frequency = len(df) / 6
    savings_ratio = avg_balance / df['amount'].mean()

    score = 0
    score += (avg_balance / 10000) * 30
    score += income_stability * 30
    score += min(tx_frequency / 10, 20)
    score += min(savings_ratio * 10, 20)
    final_score = max(0, min(int(score), 100))
    
    breakdown_dict = {
        'Average Balance': f"৳{int(avg_balance):,}",
        'Income Stability': f"{income_stability*100:.0f}%",
        'Monthly Transactions': f"{int(tx_frequency)}",
        'Savings Ratio': f"{savings_ratio:.2f}",
    }
    return final_score, breakdown_dict, avg_balance, income_stability, tx_frequency, savings_ratio

# --- Function 3: Create Interactive Gauge Chart ---
def create_gauge_chart(score):
    if score >= 70:
        color = "green"
        grade = "Excellent"
    elif score >= 50:
        color = "blue"
        grade = "Good"
    elif score >= 30:
        color = "orange"
        grade = "Fair"
    else:
        color = "red"
        grade = "Poor"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': grade, 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 30], 'color': '#fee2e2'},
                {'range': [30, 50], 'color': '#fed7aa'},
                {'range': [50, 70], 'color': '#dbeafe'},
                {'range': [70, 100], 'color': '#d1fae5'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- Function 4: Create Balance Trend Chart ---
def create_balance_trend(df):
    monthly_data = df.groupby(df['date'].dt.to_period('M')).agg({
        'balance': 'mean'
    }).reset_index()
    monthly_data['date'] = monthly_data['date'].astype(str)
    
    fig = px.line(monthly_data, x='date', y='balance',
                  title='Average Balance Trend (Last 6 Months)',
                  labels={'date': 'Month', 'balance': 'Balance (৳)'})
    fig.update_traces(line_color='#667eea', line_width=3)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- Function 5: Create Transaction Type Distribution ---
def create_transaction_pie(df):
    tx_counts = df['type'].value_counts()
    fig = px.pie(values=tx_counts.values, names=tx_counts.index,
                 title='Transaction Type Distribution',
                 color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- Main App ---
st.markdown('<div class="main-header"><h1>🇧🇩 💳 Financial Passport Demo</h1><p>AI-Powered Credit Scoring for Bangladesh Mobile Money</p></div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>ℹ️ How it works:</strong> Enter any Bangladesh mobile number to generate a synthetic credit score based on 
simulated mobile money transaction patterns. Perfect for demonstrating financial inclusion technology!
</div>
""", unsafe_allow_html=True)

# Sample numbers section
with st.expander("📱 Try These Sample Numbers"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**01712345678**\nExcellent Score (90+)")
    with col2:
        st.warning("**01812345678**\nGood Score (60-80)")
    with col3:
        st.error("**01912345678**\nFair Score (30-60)")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    phone = st.text_input(
        "📞 Enter Bangladesh Mobile Number:",
        value="01712345678",
        placeholder="e.g., 01712XXXXXX",
        help="Try different numbers to see varied credit scores!"
    )
with col2:
    st.write("")
    st.write("")
    generate_btn = st.button("🚀 Generate Passport", type="primary", use_container_width=True)

if generate_btn:
    with st.spinner('🔄 Analyzing transaction data...'):
        df = generate_sample_transactions(phone)
        score, breakdown, avg_bal, income_stab, tx_freq, sav_ratio = calculate_credit_score(df)

    # Score Display
    st.markdown("---")
    st.subheader("📊 Credit Score Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gauge chart
        fig_gauge = create_gauge_chart(score)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Score interpretation
        if score >= 70:
            st.success(f"🌟 **{score}/100** - Excellent creditworthiness!")
        elif score >= 50:
            st.info(f"✅ **{score}/100** - Good credit standing!")
        elif score >= 30:
            st.warning(f"⚠️ **{score}/100** - Fair credit score.")
        else:
            st.error(f"❌ **{score}/100** - Needs improvement.")
    
    with col2:
        # Metrics in 2x2 grid
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("💰 Average Balance", breakdown['Average Balance'], 
                     delta="Good" if avg_bal > 20000 else "Low")
            st.metric("📈 Income Stability", breakdown['Income Stability'],
                     delta="Stable" if income_stab > 0.5 else "Variable")
        with metric_col2:
            st.metric("🔄 Monthly Transactions", breakdown['Monthly Transactions'],
                     delta="Active" if tx_freq > 70 else "Moderate")
            st.metric("💎 Savings Ratio", breakdown['Savings Ratio'],
                     delta="High" if sav_ratio > 3 else "Low")

    # Charts Section
    st.markdown("---")
    st.subheader("📈 Transaction Analysis")
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_trend = create_balance_trend(df)
        st.plotly_chart(fig_trend, use_container_width=True)
    with chart_col2:
        fig_pie = create_transaction_pie(df)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Transaction Table
    st.markdown("---")
    st.subheader("📋 Recent Transactions Preview")
    
    preview = df.tail(15).copy()
    preview['amount'] = preview['amount'].apply(lambda x: f"৳{x:,}")
    preview['balance'] = preview['balance'].apply(lambda x: f"৳{x:,}")
    preview['date'] = preview['date'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(
        preview[['date', 'type', 'amount', 'balance']].reset_index(drop=True),
        use_container_width=True,
        height=400
    )

    # Explanation
    st.markdown("---")
    st.subheader("🧠 How This Score Was Calculated")
    
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.markdown("""
        **💰 Average Balance (30 points)**
        - Measures typical mobile wallet balance
        - Higher balances indicate better financial stability
        
        **📊 Income Stability (30 points)**
        - Ratio of incoming money vs total transactions
        - Consistent income improves creditworthiness
        """)
    with exp_col2:
        st.markdown("""
        **🔄 Monthly Transactions (20 points)**
        - Transaction frequency shows financial engagement
        - Regular activity indicates active financial life
        
        **💎 Savings Ratio (20 points)**
        - Balance relative to transaction amounts
        - Higher savings demonstrate financial discipline
        """)

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("**Built for Bangladesh 🇧🇩**")
with footer_col2:
    st.markdown("[GitHub](https://github.com/Dinerpordin/financial-passport-demo) | [DinerPordin.com](https://dinerpordin.com)")
with footer_col3:
    st.markdown("*Demo uses synthetic data*")
