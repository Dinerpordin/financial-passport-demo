import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
import random
from io import BytesIO
import base64

# =========== Multilingual and Font Settings ==============
LANGS = {'English':'en','বাংলা':'bn'}
translations = {
    'en': {
        'title': '💳 Financial Passport - Comprehensive Credit Rating Report',
        'subtitle': 'Advanced Credit Assessment System for Bangladesh Mobile-First Population',
        'description': 'Comprehensive credit scoring based on MFS transaction behavior, alternative data, and financial indicators.',
        'kyc_header': '📋 Personal Information (KYC)',
        'credit_score': 'Credit Score & Rating',
        'recommendations': '💡 Personalized Recommendations',
        'percentile': 'Percentile Rank',
        'security_notice': '🔒 Privacy & Security Notice',
        'security_text': 'Your data is encrypted and secure. This is a synthetic demo.',
        'export_pdf': 'Export PDF',
        'export_excel': 'Export Excel',
        'print': 'Print Report',
        'language': 'Language',
    },
    'bn': {
        'title': '💳 আর্থিক পাসপোর্ট - বিস্তৃত ক্রেডিট রেটিং রিপোর্ট',
        'subtitle': 'বাংলাদেশ মোবাইল-প্রথম জনসংখ্যার জন্য উন্নত ক্রেডিট মূল্যায়ন সিস্টেম',
        'description': 'এমএফএস লেনদেন আচরণ, বিকল্প ডেটা এবং আর্থিক সূচকের উপর ভিত্তি করে ব্যাপক ক্রেডিট স্কোরিং।',
        'kyc_header': '📋 ব্যক্তিগত তথ্য (কেওয়াইসি)',
        'credit_score': 'ক্রেডিট স্কোর ও রেটিং',
        'recommendations': '💡 ব্যক্তিগত সুপারিশ',
        'percentile': 'শতাংশিক র‍্যাঙ্ক',
        'security_notice': '🔒 গোপনীয়তা ও নিরাপত্তা নোটিশ',
        'security_text': 'আপনার ডেটা এনক্রিপ্ট করা এবং নিরাপদ। এটি একটি সিন্থেটিক ডেমো।',
        'export_pdf': 'পিডিএফ এক্সপোর্ট',
        'export_excel': 'এক্সেল এক্সপোর্ট',
        'print': 'প্রিন্ট রিপোর্ট',
        'language':'ভাষা',
    }
}

def T(key):
    return translations[st.session_state.get('lang', 'en')][key]

# =========== Sidebar Language Option =============
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'
with st.sidebar:
    st.session_state['lang'] = LANGS[st.selectbox('Language | ভাষা', list(LANGS.keys()))]

st.set_page_config(page_title=T('title'), layout="wide")
# Standardize fonts/styles in headers
st.markdown(f"<h2 style='font-size:2rem;'>{T('title')}</h2>",unsafe_allow_html=True)
st.markdown(f"<h4 style='font-size:1.3rem; color:#555;'>{T('subtitle')}</h4>",unsafe_allow_html=True)
st.markdown(f"<p style='font-size:1.1rem;'>{T('description')}</p>",unsafe_allow_html=True)

# ================= Input & Data Generation ==================
with st.sidebar:
    st.header("DaaS Assessment Inputs")
    phone_number = st.text_input("Mobile Number (Bangladesh)", value="01710000001", max_chars=11)
    profile_type = st.selectbox("Profile Type",["Urban High Income","Urban Low Income","Rural"])
def kyc_info_for_number(phone_number):
    # ... unchanged ...
    pass
def detect_provider(phone):
    # ... unchanged ...
    pass
def generate_sample_transactions(phone_number, provider, profile_type):
    # FIX: Recent dates within last 12 months
    months = 12
    days = months * 30
    start_date = datetime.now() - timedelta(days=days)
    # ... unchanged except start_date ...
    pass

provider = detect_provider(phone_number)
user_kyc = kyc_info_for_number(phone_number)
df = generate_sample_transactions(phone_number, provider, profile_type)

# ============= Core Calculations ================
def calculate_credit_score(df, user_kyc, registration_year):
    # ... unchanged ...
    pass
def get_credit_rating_tier(score): pass
# etc. (all unchanged logic: recommendations, percentile, metrics)

credit_score = calculate_credit_score(df, user_kyc, user_kyc['Registration Year'])
rating_tier, tier_color = get_credit_rating_tier(credit_score)

# ========== Visualization Functions ============
def create_credit_gauge(score): pass # unchanged
def create_profile_radar(df, user_kyc): pass # unchanged

def create_transaction_heatmap(df):
    # FIX: Recent, accurate years
    daily_counts = df.groupby('Date').size().reset_index(name='count')
    daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
    fig = go.Figure(data=go.Heatmap(
        z=daily_counts['count'],
        x=daily_counts['Date'].dt.strftime('%Y-%m-%d'),
        colorscale='YlGnBu',
        name='Transactions',
        colorbar=dict(title="Txns")
    ))
    fig.update_layout(title="Transaction Activity Heatmap", xaxis_title="Date", height=250, font=dict(size=13))
    return fig

def create_transaction_sankey(df):
    # FIX: Larger fonts/labels, show amount in link label
    transaction_flow = df[df['Type'] != 'Cash-In'].groupby(['Type', 'Merchant'])['Amount'].sum().reset_index()
    types = transaction_flow['Type'].unique()
    merchants = transaction_flow['Merchant'].unique()
    node_labels = list(types) + list(merchants)
    values = transaction_flow['Amount'].tolist()
    sources = [list(types).index(row['Type']) for idx, row in transaction_flow.iterrows()]
    targets = [len(types)+list(merchants).index(row['Merchant']) for idx, row in transaction_flow.iterrows()]
    custom_labels = [f"{row['Type']}→{row['Merchant']}: ৳{int(row['Amount']):,}" for _,row in transaction_flow.iterrows()]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=node_labels, pad=20, thickness=24, font=dict(size=16)),
        link=dict(source=sources, target=targets, value=values, label=custom_labels)
    )])
    fig.update_layout(title="Transaction Flow Sankey", height=400, font=dict(size=15), margin=dict(t=40))
    return fig

# ============ Export/Print ===============
def create_download(df, filename, method='excel'):
    if method == 'excel':
        output = BytesIO()
        df.to_excel(output, index=False)
        data = output.getvalue()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{T("export_excel")}</a>'
    else:
        csv = df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/pdf;base64,{b64}" download="{filename}">{T("export_pdf")}</a>'
    st.markdown(href, unsafe_allow_html=True)

def print_button():
    st.markdown(f"""<button onclick="window.print()" style="margin:6px; padding:7px 17px; 
                font-size:1.1em;">{T("print")}</button>""", unsafe_allow_html=True)

# ============ UI Layout ==============
with st.expander(T('kyc_header'), expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Name", user_kyc['Name'])
    col2.metric("Phone", phone_number)
    col3.metric("Provider", provider)
    col4.metric("Registered Since", user_kyc['Registration Date'])
    st.info(f"{user_kyc['Address']} | {user_kyc['Profession']}")

st.divider()

colA, colB, colC = st.columns([2,1,1])
colA.write(f"### {T('credit_score')}")
colB.metric("Score", credit_score)
colC.markdown(f"<h4 style='text-align:center;color:{tier_color}'>{rating_tier}</h4>", unsafe_allow_html=True)

# **Improvements: Larger consistent fonts, responsive columns**
st.plotly_chart(create_credit_gauge(credit_score), use_container_width=True)
st.plotly_chart(create_profile_radar(df, user_kyc), use_container_width=True)
st.plotly_chart(create_transaction_heatmap(df), use_container_width=True)
st.plotly_chart(create_transaction_sankey(df), use_container_width=True)

st.divider()

# Export/print controls
colE1, colE2, colE3 = st.columns(3)
with colE1: create_download(df, 'credit_report.xlsx', method='excel')
with colE2: create_download(df, 'credit_report.csv', method='csv')
with colE3: print_button()

# Recommendations, Percentile, etc.
# ... rest of app logic unchanged, but UI labels use T()

st.divider()
st.caption(f"Assessment for {phone_number} ({provider}) - Profile: {profile_type}. Data is synthetic for DaaS demo purposes.")

# Security notice
with st.expander(T('security_notice')):
    st.success(T('security_text'))

# QUICK DEPLOY NOTES:
# - Replace unchanged logic and helper functions with the improved ones as per your business requirements.
# - Add large comments next to improvements for maintainability.
# - Test in repo (Streamlit’s live reload will show all changes instantly).

