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
        'description': 'Enter any Bangladesh mobile number to generate a synthetic credit score based on simulated mobile money transaction patterns with advanced alternative data analysis.',
        'sample_numbers': 'Try These Sample Numbers',
        'excellent_score': 'Prime Tier (90+)',
        'good_score': 'Near-Prime (70-89)',
        'fair_score': 'Subprime (50-69)',
        'poor_score': 'Deep Subprime (<50)',
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
        'risk_tier': 'Risk Tier',
        'loan_eligibility': 'Your Loan Eligibility',
        'score_journey': 'Your Score Journey',
        'peer_comparison': 'How You Compare',
    },
    'bn': {
        'title': '💳 ফিন্যান্সিয়াল পাসপোর্ট ডেমো',
        'subtitle': 'বাংলাদেশ মোবাইল মানির জন্য এআই-চালিত ক্রেডিট স্কোরিং',
        'how_it_works': 'এটি কিভাবে কাজ করে',
        'description': 'উন্নত বিকল্প ডেটা বিশ্লেষণ সহ সিমুলেটেড মোবাইল মানি লেনদেন প্যাটার্নের উপর ভিত্তি করে একটি সিন্থেটিক ক্রেডিট স্কোর তৈরি করতে যেকোনো বাংলাদেশ মোবাইল নম্বর লিখুন।',
        'sample_numbers': 'এই নমুনা নম্বরগুলি চেষ্টা করুন',
        'excellent_score': 'প্রাইম স্তর (৯০+)',
        'good_score': 'নিয়ার-প্রাইম (৭০-৮৯)',
        'fair_score': 'সাবপ্রাইম (৫০-৬৯)',
        'poor_score': 'ডিপ সাবপ্রাইম (<৫০)',
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
        'recent_tx': 'সাম্প্রতিক লেনদেন',
        'how_calculated': 'এই স্কোর কিভাবে গণনা করা হয়েছিল',
        'demo_mode': 'ডেমো মোড',
        'privacy_notice': 'গোপনীয়তা নোটিশ',
        'export_json': '📥 JSON এক্সপোর্ট',
        'export_pdf': '📄 PDF রিপোর্ট',
        'invalid_phone': '⚠️ বৈধ নম্বর লিখুন',
        'language': 'ভাষা',
        'risk_tier': 'ঝুঁকি স্তর',
        'loan_eligibility': 'ঋণের যোগ্যতা',
        'score_journey': 'স্কোর যাত্রা',
        'peer_comparison': 'তুলনা',
    }
}

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True

def t(key):
    """Translation helper"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

def validate_bd_phone(phone):
    """Validate Bangladesh mobile number"""
    phone = str(phone).strip()
    pattern = r'^01[3-9]\d{8}$'
    return bool(re.match(pattern, phone))

# Custom CSS
st.markdown("""<style>
.main-header {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.info-box {
    padding: 1rem;
    background-color: #f0f9ff;
    border-left: 4px solid #3b82f6;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>""", unsafe_allow_html=True)

# ENHANCED Transaction Generation
def generate_sample_transactions(phone_number):
    seed = abs(hash(phone_number)) % (2**32)
    np.random.seed(seed)
    
    tx_data = {
        'Cash-In': {'weight': 0.28, 'category': 'income'},
        'Cash-Out': {'weight': 0.25, 'category': 'withdrawal'},
        'Send Money': {'weight': 0.12, 'category': 'transfer'},
        'Payment': {'weight': 0.10, 'category': 'expense'},
        'Mobile Recharge': {'weight': 0.05, 'category': 'utility'},
        'Utility Bill': {'weight': 0.08, 'category': 'utility'},
        'E-commerce': {'weight': 0.07, 'category': 'shopping'},
        'Transport': {'weight': 0.05, 'category': 'transport'}
    }
    
    tx_types = list(tx_data.keys())
    tx_weights = [tx_data[t]['weight'] for t in tx_types]
    
    merchant_categories = ['Groceries', 'Restaurant', 'Pharmacy', 'Electronics', 'Clothing',
                          'Fuel', 'Entertainment', 'Education', 'Healthcare', 'Other']
    time_patterns = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_weights = [0.25, 0.40, 0.30, 0.05]
    location_types = ['Urban', 'Suburban', 'Rural']
    location_weights = [0.60, 0.25, 0.15]
    
    n_tx = 500
    today = datetime.datetime.today()
    start_date = today - datetime.timedelta(days=180)
    dates = np.random.choice(pd.date_range(start_date, today), n_tx)
    dates.sort()
    
    tx_type = np.random.choice(tx_types, size=n_tx, p=tx_weights)
    times_of_day = np.random.choice(time_patterns, size=n_tx, p=time_weights)
    locations = np.random.choice(location_types, size=n_tx, p=location_weights)
    merchants = [np.random.choice(merchant_categories) if t in ['Payment', 'E-commerce'] else 'N/A' for t in tx_type]
    
    amounts = []
    balance = []
    categories = []
    bal = np.random.randint(500, 10000)
    
    for tt in tx_type:
        category = tx_data[tt]['category']
        categories.append(category)
        
        if tt == 'Cash-In':
            amt = np.random.randint(500, 20000)
            bal += amt
        elif tt == 'Cash-Out':
            amt = np.random.randint(100, 15000)
            bal -= amt
        elif tt == 'Send Money':
            amt = np.random.randint(200, 7000)
            bal -= amt
        elif tt in ['Payment', 'E-commerce']:
            amt = np.random.randint(50, 5000)
            bal -= amt
        elif tt == 'Utility Bill':
            amt = np.random.randint(500, 3000)
            bal -= amt
        elif tt == 'Transport':
            amt = np.random.randint(20, 500)
            bal -= amt
        else:
            amt = np.random.randint(10, 200)
            bal -= amt
        
        amounts.append(amt)
        balance.append(max(bal, 0))
    
    df = pd.DataFrame({
        'date': dates,
        'type': tx_type,
        'category': categories,
        'amount': amounts,
        'balance': balance,
        'time_of_day': times_of_day,
        'location_type': locations,
        'merchant_category': merchants
    })
    
    return df.sort_values('date')
# PART 2: State, UI, Score, and Visual Components

# Scoring function (transparent methodology)
def calculate_score(df):
    # Mean balance (0–30 pts)
    avg_balance = df['balance'].mean()
    balance_score = min(30, int(avg_balance / 7000 * 30))
    # Income stability (0–30 pts)
    income_tx = df[df['type'] == 'Cash-In']['amount']
    income_std = income_tx.std() if len(income_tx) > 1 else 0
    income_score = 30 - min(20, int((income_std / (income_tx.mean() + 1)) * 20))
    income_score = max(10, income_score)
    # Monthly activity (0–20 pts)
    months = (df['date'].max() - df['date'].min()).days // 30
    monthly_tx = len(df) // max(months, 1)
    activity_score = min(20, int(monthly_tx / 40 * 20))
    # Savings ratio (0–20 pts)
    end_balance = df['balance'].iloc[-1]
    income_total = income_tx.sum()
    savings_ratio = end_balance / (income_total + 1)
    savings_score = min(20, int(savings_ratio * 20))
    # Total
    total = balance_score + income_score + activity_score + savings_score
    detail = {
        'Balance Score': balance_score,
        'Income Score': income_score,
        'Activity Score': activity_score,
        'Savings Score': savings_score,
        'Total': total,
    }
    return total, detail

# UI Helper: Score tier translation
def score_tier(score, lang='en'):
    if lang == 'bn':
        if score >= 90: return 'প্রাইম স্তর'
        if score >= 70: return 'নিয়ার-প্রাইম'
        if score >= 50: return 'সাবপ্রাইম'
        return 'ডিপ সাবপ্রাইম'
    if score >= 90: return 'Prime Tier'
    if score >= 70: return 'Near-Prime'
    if score >= 50: return 'Subprime'
    return 'Deep Subprime'

# Export data as JSON
def export_json(results):
    # Convert all datetime in transaction records to str (ISO format)
    clean_results = results.copy()
    tx_list = clean_results.get('tx', [])

    for tx in tx_list:
        # If 'date' is a pandas Timestamp or datetime object, convert to ISO string
        if hasattr(tx['date'], 'isoformat'):
            tx['date'] = tx['date'].isoformat()
        else:
            tx['date'] = str(tx['date'])

    return BytesIO(json.dumps(clean_results, ensure_ascii=False, indent=2).encode('utf-8'))
# Export PDF (placeholder)
def export_pdf():
    pdf = BytesIO()
    pdf.write(b'%PDF-1.4\n% Demo PDF Placeholder\n')
    pdf.seek(0)
    return pdf

# Sidebar: Language & Demo mode toggle
with st.sidebar:
    st.selectbox(t('language'), options=['en', 'bn'], index=0 if st.session_state.language == 'en' else 1, key='language')
    st.checkbox(t('demo_mode') + ' 🧪', value=st.session_state.demo_mode, key='demo_mode')

# Main UI
st.markdown(f"<div class='main-header'><h2>{t('title')}</h2><h4>{t('subtitle')}</h4></div>", unsafe_allow_html=True)
st.markdown(f"<div class='info-box'>{t('description')}</div>", unsafe_allow_html=True)

# Input
phone = st.text_input(t('enter_mobile'), value=st.session_state.get('phone', ''), max_chars=11, key='phone')
valid_phone = validate_bd_phone(phone)
if not valid_phone and phone:
    st.warning(t('invalid_phone'))

if st.button(t('generate_btn'), disabled=not valid_phone):
    st.session_state['run'] = True

run = st.session_state.get('run', False) and valid_phone

if run:
    st.info(t('analyzing'))
    df = generate_sample_transactions(phone)
    score, detail = calculate_score(df)
    st.session_state['results'] = {
        'phone': phone,
        'score': score,
        'detail': detail,
        'tx': df.to_dict(orient='records')
    }
    st.session_state['run'] = False

results = st.session_state.get('results')
if results:
    score = results['score']
    detail = results['detail']
    df = pd.DataFrame(results['tx'])

    # Show Credit Score
    st.subheader(t('credit_score'))
    color = 'green' if score >= 90 else 'orange' if score >= 70 else 'yellow' if score >= 50 else 'red'
    st.markdown(f"<h2 style='color:{color};'>{score}</h2>", unsafe_allow_html=True)
    st.caption(f"{t('risk_tier')}: {score_tier(score, st.session_state.language)}")

        # Half-circle gauge chart for score visualization
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': t('credit_score'), 'font': {'size': 20}},
        number={'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 70], 'color': '#fff9c4'},
                {'range': [70, 90], 'color': '#ffe0b2'},
                {'range': [90, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score,
            }
        }
    ))
    gauge_fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=0),
        height=300
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # PART 3 starts here...
    # Score Breakdown Chart
    st.markdown("#### " + t('how_calculated'))
    fig = go.Figure(go.Bar(
        x=list(detail.keys())[:-1], y=list(detail.values())[:-1], marker_color=['#636EFA', '#00CC96', '#FFA15A', '#19D3F3']))
    fig.update_layout(yaxis=dict(range=[0, 30]), title=None, margin=dict(l=0, r=0, t=0, b=0), height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Metrics Grid
    metrics = [
        (t('avg_balance'), f"৳{df['balance'].mean():,.0f}"),
        (t('income_stability'), f"৳{df[df['type']=='Cash-In']['amount'].mean():,.0f}"),
        (t('monthly_tx'), f"{len(df)//6}"),
        (t('savings_ratio'), f"{df['balance'].iloc[-1]/(df[df['type']=='Cash-In']['amount'].sum()+1):.0%}")
    ]
    cols = st.columns(4)
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, val)

    # Transaction Analysis
    st.markdown("### " + t('transaction_analysis'))
    bal_fig = px.line(df.sort_values('date'), x='date', y='balance', title=None, markers=True)
    bal_fig.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=15, b=0))
    st.plotly_chart(bal_fig, use_container_width=True)

    pie = px.pie(df, names='type', title=None, hole=0.4)
    pie.update_layout(showlegend=True, margin=dict(l=0, r=0, t=0, b=0), height=200)
    st.plotly_chart(pie, use_container_width=True)

    # Recent transactions
    st.markdown("#### " + t('recent_tx'))
    st.dataframe(df.sort_values('date', ascending=False).head(15)[['date', 'type', 'amount', 'balance']].reset_index(drop=True), height=330)

    # Export Options
    btn_col, pdf_col = st.columns(2)
    with btn_col:
        st.download_button(t('export_json'), data=export_json(results), file_name=f"financial_passport_{results['phone']}.json")
    with pdf_col:
        st.download_button(t('export_pdf'), data=export_pdf(), file_name="financial-passport-report.pdf")

    # Privacy and Methodology
    with st.expander(t('privacy_notice')):
        st.write(
            "This demo only uses synthetic/mobile money data for simulation purposes. No real personal data is processed. See [GitHub](https://github.com/Dinerpordin/financial-passport-demo) for open source code."
        )
    with st.expander("API Documentation / Integration Guide"):
        st.markdown(
            "- **Endpoint**: `/predict`\n- **Method**: POST\n- **Payload**: `{ \"phone\": \"01XXXXXXXXX\" }`\n- **Response**: Returns score and component details\n- **Python SDK & webhook examples included in GitHub**\n- **Contact**: e.dinerpordin@gmail.com"
        )

    # Reset
    if st.button("🔄 New Score / Clear"):
        st.session_state['results'] = None
        st.experimental_rerun()

