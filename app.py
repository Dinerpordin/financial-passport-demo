import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
import random

# ===== CREDIT SCORE CALCULATION FUNCTIONS =====
def calculate_credit_score(df, user_kyc, registration_year):
    base_score = 300
    payment_history_score = calculate_payment_history_score(df)
    utilization_score = calculate_utilization_score(df)
    length_score = calculate_length_score(registration_year)
    mix_score = calculate_credit_mix_score(df)
    inquiries_score = calculate_inquiries_score()
    final_score = base_score + (
        payment_history_score * 0.35 +
        utilization_score * 0.30 +
        length_score * 0.15 +
        mix_score * 0.10 +
        inquiries_score * 0.10
    )
    return min(850, max(300, final_score))

def calculate_payment_history_score(df):
    daily_txns = df.groupby('Date').size()
    consistency = min(100, len(daily_txns) * 5)
    timeliness_bonus = 100
    return consistency + timeliness_bonus

def calculate_utilization_score(df):
    avg_balance = df['Balance'].mean()
    total_monthly_spend = df[df['Type'] != 'Cash-In']['Amount'].sum() / 12
    utilization_ratio = min(1, total_monthly_spend / (avg_balance + 1))
    util_score = 150 * (1 - utilization_ratio * 0.5)
    return util_score

def calculate_length_score(registration_year):
    years_active = 2024 - registration_year
    length_score = min(100, years_active * 10)
    return length_score

def calculate_credit_mix_score(df):
    unique_types = df['Type'].nunique()
    unique_merchants = df['Merchant'].nunique()
    mix_score = (unique_types / 6) * 40 + (min(unique_merchants, 20) / 20) * 40
    return mix_score

def calculate_inquiries_score():
    return random.randint(60, 75)

def get_credit_rating_tier(score):
    if score >= 750:
        return "EXCELLENT", "#27ae60"
    elif score >= 700:
        return "VERY GOOD", "#2ecc71"
    elif score >= 650:
        return "GOOD", "#f39c12"
    elif score >= 600:
        return "FAIR", "#e67e22"
    else:
        return "POOR", "#e74c3c"

# ===== KYC INFO AND DATA GENERATION =====
def kyc_info_for_number(phone_number):
    random.seed(hash(phone_number))
    first_names = ["Rahim", "Karim", "Sumaiya", "Alamgir", "Nazmul", "Jannat", "Rifat", "Sabina", "Mizan", "Shamim"]
    last_names = ["Hossain", "Islam", "Chowdhury", "Ahmed", "Rahman", "Begum", "Akter", "Khan", "Sarker", "Miah"]
    cities = ["Dhaka", "Chattogram", "Sylhet", "Khulna", "Rajshahi", "Barisal", "Comilla", "Rangpur"]
    professions = ["Student", "Service", "Business", "Freelancer", "Teacher", "Engineer", "Doctor", "Driver"]
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    address = f"{random.randint(1,140)} {random.choice(['Main Rd', 'Lane', 'Ave','Goli'])}, {random.choice(cities)}"
    profession = random.choice(professions)
    reg_year = random.randint(2011, 2023)
    reg_date = datetime(reg_year, random.randint(1,12), random.randint(1,28)).strftime("%Y-%m-%d")
    alt_numbers = []
    for i in range(random.randint(0,2)):
        alt_pref = random.choice(['017','018','019','015','016'])
        alt_number = alt_pref + f"{random.randint(10000000,99999999)}"
        alt_numbers.append(alt_number)
    return {"Name": name, "Address": address, "Profession": profession, "Registration Date": reg_date, "Registration Year": reg_year, "Other Numbers": alt_numbers}

PROVIDER_PREFIX = {'Grameenphone': ['017'], 'Robi': ['018'], 'Banglalink': ['019'], 'Teletalk': ['015'], 'Airtel': ['016']}
PROVIDERS = list(PROVIDER_PREFIX.keys())

def detect_provider(phone):
    phone = str(phone)
    for provider, prefixes in PROVIDER_PREFIX.items():
        if any(phone.startswith(pref) for pref in prefixes):
            return provider
    return "Unknown"

def generate_sample_transactions(phone_number, provider, profile_type):
    np.random.seed(abs(hash(phone_number)) % 10**7)
    random.seed(abs(hash(phone_number)) % 10**7)
    months = 12
    days = months * 30
    base_profile = {"Urban High Income": {"base_balance": 150000, "txn_multiplier": 1.7}, "Urban Low Income": {"base_balance": 25000, "txn_multiplier": 1.0}, "Rural": {"base_balance": 6000, "txn_multiplier": 0.7}}
    profile = base_profile.get(profile_type, base_profile["Urban High Income"])
    txns = []
    current_balance = profile["base_balance"]
    date = datetime.now() - timedelta(days=days)
    for i in range(days):
        num_txns_today = int(np.random.poisson(2 * profile["txn_multiplier"]))
        for _ in range(num_txns_today):
            t_type, t_amt, t_merchant = simulate_transaction(provider)
            if t_type == "Cash-In":
                current_balance += t_amt
            else:
                current_balance = max(0, current_balance - t_amt)
            txns.append({"Date": date.strftime("%Y-%m-%d"), "Type": t_type, "Amount": round(t_amt, 2), "Merchant": t_merchant, "Balance": round(current_balance, 2), "Provider": provider})
        date += timedelta(days=1)
    df = pd.DataFrame(txns)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def simulate_transaction(provider):
    weights = {'Grameenphone': [0.13, 0.13, 0.15, 0.20, 0.14, 0.25], 'Robi': [0.20, 0.10, 0.18, 0.18, 0.14, 0.20], 'Banglalink': [0.10, 0.11, 0.25, 0.25, 0.11, 0.18], 'Teletalk': [0.20, 0.13, 0.13, 0.18, 0.18, 0.18], 'Airtel': [0.17, 0.19, 0.16, 0.18, 0.13, 0.17], 'Unknown': [0.15,0.15,0.15,0.15,0.20,0.20]}
    types = ['Cash-In','Cash-Out','Mobile Recharge','E-commerce','Bill Payment','P2P Payment']
    merchants_pool = {'Mobile Recharge': ['GP Topup', 'Robi Topup', 'Banglalink Topup'], 'E-commerce': ['Daraz', 'Evaly', 'Pickaboo'], 'Bill Payment': ['DESCO','WASA','ISP'], 'Cash-In': ['bKash Agent', 'Nagad Agent'], 'Cash-Out': ['ATM','Agent'], 'P2P Payment':['Friend/Family','Rent','Tuition']}
    t_type = random.choices(types, weights=weights.get(provider,"Unknown"))[0]
    if t_type == 'Cash-In':
        amt = random.randint(1000,5000)
    elif t_type == 'Mobile Recharge':
        amt = random.randint(30,300)
    elif t_type == 'E-commerce':
        amt = random.randint(300,5000)
    elif t_type == 'Bill Payment':
        amt = random.randint(250,2500)
    elif t_type == 'P2P Payment':
        amt = random.randint(100,2000)
    else:
        amt = random.randint(1000,5000)
    t_merchant = random.choice(merchants_pool[t_type])
    return t_type, amt, t_merchant

# ===== STREAMLIT UI =====
st.set_page_config(page_title="💳 Financial Passport - Credit Rating DaaS Bangladesh", layout="wide")
st.title("💳 Financial Passport - Comprehensive Credit Rating Report")
st.markdown("""**Advanced Credit Assessment System for Bangladesh Mobile-First Population**  
Comprehensive credit scoring based on MFS transaction behavior, alternative data, and financial indicators.""")

with st.sidebar:
    st.header("DaaS Assessment Inputs")
    phone_number = st.text_input("Mobile Number (Bangladesh)", value="01710000001", max_chars=11)
    profile_type = st.selectbox("Profile Type",["Urban High Income","Urban Low Income","Rural"])

provider = detect_provider(phone_number)
user_kyc = kyc_info_for_number(phone_number)
df = generate_sample_transactions(phone_number, provider, profile_type)

# Calculate credit score and metrics
credit_score = calculate_credit_score(df, user_kyc, user_kyc['Registration Year'])
rating_tier, tier_color = get_credit_rating_tier(credit_score)

with st.expander("📋 Personal Information (KYC)", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Name", user_kyc['Name'])
    with col2:
        st.metric("Phone", phone_number)
    with col3:
        st.metric("Provider", provider)
    with col4:
        st.metric("Registered Since", user_kyc['Registration Date'])
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Address:** {user_kyc['Address']}")
    with col_b:
        st.write(f"**Profession:** {user_kyc['Profession']}")
    if user_kyc['Other Numbers']:
        st.info(f"Associated Numbers: {', '.join(user_kyc['Other Numbers'])}")

# Credit Score Section
st.divider()
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.write("## Credit Score & Rating")
with col2:
    st.metric("Score", credit_score, delta=None)
with col3:
    st.markdown(f"<h3 style='text-align: center; color: {tier_color};'>{rating_tier}</h3>", unsafe_allow_html=True)

# Financial Metrics & Key Indicators
with st.expander("📊 Financial Metrics & Indicators", expanded=True):
    avg_balance = int(df['Balance'].mean())
    total_txn = int(df['Amount'].sum())
    monthly_spend = int(df[df['Type']!='Cash-In']['Amount'].sum() / 12)
    avg_deposits_per_month = int(df[df['Type']=="Cash-In"].groupby(df['Date'].dt.to_period('M')).size().mean())
    utilization = min(100, int(100 * (df[df['Type']!='Cash-In']['Amount'].sum()) / (avg_balance * 12 + 1)))
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Avg Balance", f"৳{avg_balance:,}")
    with m2:
        st.metric("Total Volume", f"৳{total_txn:,}")
    with m3:
        st.metric("Monthly Spend", f"৳{monthly_spend:,}")
    with m4:
        st.metric("Utilization", f"{utilization}%")
    with m5:
        st.metric("Deposits/Month", avg_deposits_per_month)
    with m6:
        st.metric("Active Days", len(df.groupby('Date')))

# Credit Score Breakdown
with st.expander("📊 Credit Score Components (Weighted)", expanded=True):
    c_col1, c_col2, c_col3, c_col4, c_col5 = st.columns(5)
    payment_score = calculate_payment_history_score(df)
    util_score = calculate_utilization_score(df)
    length_score = calculate_length_score(user_kyc['Registration Year'])
    mix_score = calculate_credit_mix_score(df)
    inq_score = calculate_inquiries_score()
    with c_col1:
        st.progress(min(1.0, payment_score/200), text=f"Payment History (35%): {int(payment_score)}")
    with c_col2:
        st.progress(min(1.0, util_score/150), text=f"Credit Utilization (30%): {int(util_score)}")
    with c_col3:
        st.progress(min(1.0, length_score/100), text=f"Length of History (15%): {int(length_score)}")
    with c_col4:
        st.progress(min(1.0, mix_score/80), text=f"Credit Mix (10%): {int(mix_score)}")
    with c_col5:
        st.progress(min(1.0, inq_score/75), text=f"Inquiries (10%): {int(inq_score)}")

# Transaction Analysis
with st.expander("💳 Transaction Analysis", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Balance Trend", "Spend by Category", "Top Merchants"])
    with tab1:
        monthly_bal = df.groupby(df['Date'].dt.to_period('M'))['Balance'].mean().reset_index()
        trend = go.Figure()
        trend.add_trace(go.Scatter(x=monthly_bal['Date'].astype(str), y=monthly_bal['Balance'], mode='lines+markers', name='Avg Balance', line=dict(color='#1064ea', width=3), marker=dict(size=8)))
        trend.update_layout(margin=dict(t=20, b=15, l=0, r=0), xaxis_title="Month", yaxis_title="Balance", height=340)
        st.plotly_chart(trend, use_container_width=True)
    with tab2:
        cat_sums = df[df.Type!='Cash-In'].groupby('Type')['Amount'].sum().sort_values(ascending=False)
        piefig = go.Figure()
        piefig.add_trace(go.Pie(labels=cat_sums.index, values=cat_sums.values, hole=0.42))
        piefig.update_layout(margin=dict(l=5, r=5, t=10, b=5), height=340)
        st.plotly_chart(piefig, use_container_width=True)
    with tab3:
        top_merchants = df['Merchant'].value_counts().head(8)
        barfig = go.Figure()
        barfig.add_trace(go.Bar(x=top_merchants.values, y=top_merchants.index, orientation='h', marker=dict(color='#a6d8f8')))
        barfig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=300)
        st.plotly_chart(barfig, use_container_width=True)

# Public Records Section
with st.expander("⚠️ Public Records & Risk Indicators", expanded=False):
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Defaults on Record", "0")
    with col_p2:
        st.metric("Collections", "None")
    with col_p3:
        st.metric("Delinquency Rate", "0%")
    st.success("No negative public records detected for this profile.")

# Transaction History
with st.expander("📋 Complete Transaction History", expanded=False):
    st.dataframe(df.sort_values('Date', ascending=False).reset_index(drop=True), use_container_width=True, height=420)

st.divider()
st.caption(f"Assessment for {phone_number} ({provider}) - Profile: {profile_type}. Data is synthetic for DaaS demo purposes. All metrics derived from mobile transaction patterns and alternative data sources.")
