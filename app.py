import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
import json
import re
from io import BytesIO

# Page config
st.set_page_config(
    page_title="💳 Financial Passport Demo",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Language translations
TRANSLATIONS = {
    'en': {
        'title': '💳 Financial Passport Demo',
        'subtitle': 'AI-Powered Credit Scoring for Bangladesh Mobile Money',
        'how_it_works': 'How it works',
        'description': 'Enter any Bangladesh mobile number to generate a synthetic credit score based on simulated mobile money transaction patterns. Perfect for demonstrating financial inclusion technology!',
        'sample_numbers': 'Try These Sample Numbers',
        'excellent_score': 'Excellent Score (90+)',
        'good_score': 'Good Score (60-80)',
        'fair_score': 'Fair Score (30-60)',
        'enter_mobile': 'Enter Bangladesh Mobile Number',
        'generate_btn': '🚀 Generate Passport',
        'analyzing': '🔄 Analyzing transaction data...',
        'credit_score': 'Credit Score Analysis',
        'excellent': 'Excellent',
        'good': 'Good',
        'fair': 'Fair',
        'poor': 'Poor',
        'avg_balance': 'Average Balance',
        'income_stability': 'Income Stability',
        'monthly_tx': 'Monthly Transactions',
        'savings_ratio': 'Savings Ratio',
        'transaction_analysis': 'Transaction Analysis',
        'recent_tx': 'Recent Transactions Preview',
        'how_calculated': 'How This Score Was Calculated',
        'demo_mode': 'Demo Mode',
        'privacy_notice': 'Privacy & Data Notice',
        'export_json': '📥 Export JSON',
        'export_pdf': '📄 Generate PDF Report',
        'invalid_phone': '⚠️ Please enter a valid Bangladesh mobile number (11 digits starting with 01)',
        'language': 'Language',
    },
    'bn': {
        'title': '💳 ফিন্যান্সিয়াল পাসপোর্ট ডেমো',
        'subtitle': 'বাংলাদেশ মোবাইল মানির জন্য এআই-চালিত ক্রেডিট স্কোরিং',
        'how_it_works': 'এটি কিভাবে কাজ করে',
        'description': 'সিমুলেটেড মোবাইল মানি লেনদেন প্যাটার্নের উপর ভিত্তি করে একটি সিন্থেটিক ক্রেডিট স্কোর তৈরি করতে যেকোনো বাংলাদেশ মোবাইল নম্বর লিখুন। আর্থিক অন্তর্ভুক্তি প্রযুক্তি প্রদর্শনের জন্য উপযুক্ত!',
        'sample_numbers': 'এই নমুনা নম্বরগুলি চেষ্টা করুন',
        'excellent_score': 'চমৎকার স্কোর (৯০+)',
        'good_score': 'ভাল স্কোর (৬০-৮০)',
        'fair_score': 'মধ্যম স্কোর (৩০-৬০)',
        'enter_mobile': 'বাংলাদেশ মোবাইল নম্বর লিখুন',
        'generate_btn': '🚀 পাসপোর্ট তৈরি করুন',
        'analyzing': '🔄 লেনদেন ডেটা বিশ্লেষণ করা হচ্ছে...',
        'credit_score': 'ক্রেডিট স্কোর বিশ্লেষণ',
        'excellent': 'চমৎকার',
        'good': 'ভাল',
        'fair': 'মধ্যম',
        'poor': 'দুর্বল',
        'avg_balance': 'গড় ব্যালেন্স',
        'income_stability': 'আয়ের স্থিতিশীলতা',
        'monthly_tx': 'মাসিক লেনদেন',
        'savings_ratio': 'সঞ্চয় অনুপাত',
        'transaction_analysis': 'লেনদেন বিশ্লেষণ',
        'recent_tx': 'সাম্প্রতিক লেনদেন পূর্বরূপ',
        'how_calculated': 'এই স্কোর কিভাবে গণনা করা হয়েছিল',
        'demo_mode': 'ডেমো মোড',
        'privacy_notice': 'গোপনীয়তা এবং ডেটা নোটিশ',
        'export_json': '📥 JSON এক্সপোর্ট করুন',
        'export_pdf': '📄 PDF রিপোর্ট তৈরি করুন',
        'invalid_phone': '⚠️ অনুগ্রহ করে একটি বৈধ বাংলাদেশ মোবাইল নম্বর লিখুন (০১ দিয়ে শুরু হওয়া ১১ সংখ্যা)',
        'language': 'ভাষা',
    }
}

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True

def t(key):
    """Translation helper function"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

# Validation function
def validate_bd_phone(phone):
    """Validate Bangladesh mobile number format"""
    phone = str(phone).strip()
    pattern = r'^01[3-9]\d{8}$'
    return bool(re.match(pattern, phone))

# Custom CSS for better styling
st.markdown("""<style>
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
.privacy-box {
    padding: 1rem;
    background-color: #fef3c7;
    border-left: 4px solid #f59e0b;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>""", unsafe_allow_html=True)

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
    
    score_components = {
        'Balance Score': min((avg_balance / 10000) * 30, 30),
        'Income Score': income_stability * 30,
        'Activity Score': min(tx_frequency / 10, 20),
        'Savings Score': min(savings_ratio * 10, 20)
    }
    
    return final_score, breakdown_dict, avg_balance, income_stability, tx_frequency, savings_ratio, score_components

# --- Function 3: Create Interactive Gauge Chart ---
def create_gauge_chart(score):
    if score >= 70:
        color = "green"
        grade = t('excellent')
    elif score >= 50:
        color = "blue"
        grade = t('good')
    elif score >= 30:
        color = "orange"
        grade = t('fair')
    else:
        color = "red"
        grade = t('poor')
    
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

# --- Function 4: Create Score Breakdown Chart ---
def create_score_breakdown(score_components):
    fig = go.Figure(data=[
        go.Bar(
            x=list(score_components.values()),
            y=list(score_components.keys()),
            orientation='h',
            marker=dict(
                color=['#667eea', '#764ba2', '#f59e0b', '#10b981'],
                line=dict(color='rgb(248, 248, 249)', width=1)
            )
        )
    ])
    fig.update_layout(
        title='Score Component Breakdown',
        xaxis_title='Points',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- Function 5: Create Balance Trend Chart ---
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

# --- Function 6: Create Transaction Type Distribution ---
def create_transaction_pie(df):
    tx_counts = df['type'].value_counts()
    fig = px.pie(values=tx_counts.values, names=tx_counts.index,
                title='Transaction Type Distribution',
                color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- Function 7: Export to JSON ---
def export_to_json(score, breakdown, phone, df):
    data = {
        'phone_number': phone,
        'credit_score': int(score),
        'generated_date': datetime.datetime.now().isoformat(),
        'breakdown': breakdown,
        'transaction_summary': {
            'total_transactions': len(df),
            'date_range': {
                'from': df['date'].min().isoformat(),
                'to': df['date'].max().isoformat()
            },
            'transaction_types': df['type'].value_counts().to_dict()
        },
        'demo_mode': st.session_state.demo_mode
    }
    return json.dumps(data, indent=2)

# --- Main App ---
# Top bar with language toggle
col1, col2, col3 = st.columns([2, 2, 1])
with col3:
    lang_option = st.selectbox(
        t('language'),
        options=['en', 'bn'],
        index=0 if st.session_state.language == 'en' else 1,
        format_func=lambda x: 'English' if x == 'en' else 'বাংলা',
        key='lang_select'
    )
    if lang_option != st.session_state.language:
        st.session_state.language = lang_option
        st.rerun()

st.markdown(f'<div class="main-header"><h1>{t("title")}</h1><p>{t("subtitle")}</p></div>', unsafe_allow_html=True)

# Demo mode and privacy notice
with st.expander(f"🛡️ {t('privacy_notice')}"):
    st.markdown(f"""
    **{t('demo_mode')}:** This application uses synthetic data for demonstration purposes only.
    
    - **No Real Data**: All transaction patterns and scores are algorithmically generated
    - **Privacy**: No actual financial data is collected, stored, or transmitted
    - **For Testing**: This demo showcases the technology for financial inclusion
    - **Methodology**: Credit scores are calculated using simulated mobile money patterns
    
    *Built for Bangladesh 🇧🇩 | Powered by AI | Open Source on [GitHub](https://github.com/Dinerpordin/financial-passport-demo)*
    """)

st.markdown(f"""<div class="info-box"><strong>ℹ️ {t('how_it_works')}:</strong> {t('description')}</div>""", unsafe_allow_html=True)

# Sample numbers section
with st.expander(f"📱 {t('sample_numbers')}"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**01712345678**\n{t('excellent_score')}")
    with col2:
        st.success(f"**01812345678**\n{t('good_score')}")
    with col3:
        st.warning(f"**01912345678**\n{t('fair_score')}")
    with col4:
        st.error(f"**01612345678**\nPoor Score (10-30)")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    phone = st.text_input(
        f"📞 {t('enter_mobile')}:",
        value="01712345678",
        placeholder="e.g., 01712XXXXXX",
        help="Try different numbers to see varied credit scores!"
    )
    
    # Validation feedback
    if phone and not validate_bd_phone(phone):
        st.error(t('invalid_phone'))

with col2:
    st.write("")
    st.write("")
    generate_btn = st.button(t('generate_btn'), type="primary", use_container_width=True)

if generate_btn and validate_bd_phone(phone):
    with st.spinner(t('analyzing')):
        df = generate_sample_transactions(phone)
        score, breakdown, avg_bal, income_stab, tx_freq, sav_ratio, score_components = calculate_credit_score(df)
        
        # Store in session state for export
        st.session_state.current_data = {
            'score': score,
            'breakdown': breakdown,
            'phone': phone,
            'df': df
        }
        
        # Score Display
        st.markdown("---")
        st.subheader(f"📊 {t('credit_score')}")
        
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
            # Score breakdown chart
            fig_breakdown = create_score_breakdown(score_components)
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Metrics in grid
        st.markdown("---")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric(f"💰 {t('avg_balance')}", breakdown['Average Balance'], 
                     delta="Good" if avg_bal > 20000 else "Low")
        with metric_col2:
            st.metric(f"📈 {t('income_stability')}", breakdown['Income Stability'],
                     delta="Stable" if income_stab > 0.5 else "Variable")
        with metric_col3:
            st.metric(f"🔄 {t('monthly_tx')}", breakdown['Monthly Transactions'],
                     delta="Active" if tx_freq > 70 else "Moderate")
        with metric_col4:
            st.metric(f"💎 {t('savings_ratio')}", breakdown['Savings Ratio'],
                     delta="High" if sav_ratio > 3 else "Low")
        
        # Export buttons
        st.markdown("---")
        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
        with exp_col1:
            json_data = export_to_json(score, breakdown, phone, df)
            st.download_button(
                label=t('export_json'),
                data=json_data,
                file_name=f"financial_passport_{phone}.json",
                mime="application/json",
                use_container_width=True
            )
        with exp_col2:
            st.button(t('export_pdf'), use_container_width=True, help="PDF generation coming soon!")
        
        # Charts Section
        st.markdown("---")
        st.subheader(f"📈 {t('transaction_analysis')}")
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_trend = create_balance_trend(df)
            st.plotly_chart(fig_trend, use_container_width=True)
        with chart_col2:
            fig_pie = create_transaction_pie(df)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Transaction Table
        st.markdown("---")
        st.subheader(f"📋 {t('recent_tx')}")
        
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
        st.subheader(f"🧠 {t('how_calculated')}")
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("""
            **💰 Average Balance (30 points)**
            - Measures typical mobile wallet balance
            - Higher balances indicate better financial stability
            
            **📈 Income Stability (30 points)**
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

# API Documentation Section
with st.expander("📚 API Integration Guide"):
    st.markdown("""
    ### For Developers & DaaS Partners
    
    **Integration Options:**
    
    1. **REST API** (Coming Soon)
       ```python
       POST /api/v1/credit-score
       {
         "phone_number": "01712345678",
         "transaction_data": [...]
       }
       ```
    
    2. **Python SDK**
       ```python
       from financial_passport import CreditScorer
       scorer = CreditScorer(api_key="your_key")
       result = scorer.calculate(phone="01712345678")
       ```
    
    3. **Webhook Notifications**
       - Real-time score updates
       - Batch processing support
       - Secure data transmission
    
    **Features:**
    - ✅ Synthetic data generation for testing
    - ✅ Customizable scoring models
    - ✅ Multi-language support (EN/BN)
    - ✅ Export to JSON/PDF
    - ✅ Privacy-first design
    
    **Contact:** e.dinerpordin@gmail.com for API access
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
