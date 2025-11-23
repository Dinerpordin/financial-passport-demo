import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
import random

# ----- Synthetic KYC Profile Generator -----
def kyc_info_for_number(phone_number):
    random.seed(hash(phone_number))
    first_names = ["Rahim", "Karim", "Sumaiya", "Alamgir", "Nazmul", "Jannat", "Rifat", "Sabina", "Mizan", "Shamim"]
    last_names = ["Hossain", "Islam", "Chowdhury", "Ahmed", "Rahman", "Begum", "Akter", "Khan", "Sarker", "Miah"]
    cities = ["Dhaka", "Chattogram", "Sylhet", "Khulna", "Rajshahi", "Barisal", "Comilla", "Rangpur", "Mymensingh", "Jessore"]
    professions = ["Student", "Service", "Business", "Freelancer", "Teacher", "Engineer", "Doctor", "Driver", "Housewife"]
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
    return {
        "Name": name,
        "Address": address,
        "Profession": profession,
        "Registration Date": reg_date,
        "Other Numbers": alt_numbers
    }

# ----- Financial Data Simulation -----
PROVIDER_PREFIX = {
    'Grameenphone': ['017'],
    'Robi': ['018'],
    'Banglalink': ['019'],
    'Teletalk': ['015'],
    'Airtel': ['016'],
}
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

    base_profile = {
        "Urban High Income": {"base_balance": 150000, "txn_multiplier": 1.7},
        "Urban Low Income": {"base_balance": 25000, "txn_multiplier": 1.0},
        "Rural": {"base_balance": 6000, "txn_multiplier": 0.7}
    }
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
                current_balance -= t_amt
            txns.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Type": t_type,
                "Amount": round(t_amt, 2),
                "Merchant": t_merchant,
                "Balance": round(max(current_balance, 0), 2),
                "Provider": provider
            })
        date += timedelta(days=1)
    df = pd.DataFrame(txns)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def simulate_transaction(provider):
    weights = {
        'Grameenphone': [0.13, 0.13, 0.15, 0.20, 0.14, 0.25],
        'Robi': [0.20, 0.10, 0.18, 0.18, 0.14, 0.20],
        'Banglalink': [0.10, 0.11, 0.25, 0.25, 0.11, 0.18],
        'Teletalk': [0.20, 0.13, 0.13, 0.18, 0.18, 0.18],
        'Airtel': [0.17, 0.19, 0.16, 0.18, 0.13, 0.17],
        'Unknown': [0.15,0.15,0.15,0.15,0.20,0.20],
    }
    types = ['Cash-In','Cash-Out','Mobile Recharge','E-commerce','Bill Payment','P2P Payment']
    merchants_pool = {
        'Mobile Recharge': ['GP Topup', 'Robi Topup', 'Banglalink Topup', 'Airtel Topup', 'Teletalk Topup'],
        'E-commerce': ['Daraz', 'Evaly', 'Pickaboo', 'Ajkerdeal'],
        'Bill Payment': ['DESCO','Dhaka WASA','Banglalion','ISP Vision'],
        'Cash-In': ['bKash Agent', 'Nagad Agent'],
        'Cash-Out': ['ATM','Agent'],
        'P2P Payment':['Friend/Family','Landlord/Rent','Tuition Fee'],
    }
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

# ----- Streamlit UI & Dashboard -----
st.set_page_config(page_title="💳 Financial Passport DaaS Bangladesh", layout="wide")

st.title("💳 Financial Passport - DaaS Financial Assessment (Bangladesh)")
st.markdown("""
Input any Bangladeshi mobile number for instant synthetic financial and KYC assessment as a professional DaaS demo.
""")

# User input
with st.sidebar:
    st.header("DaaS Inputs")
    phone_number = st.text_input("Mobile Number (Bangladesh)", value="01710000001", max_chars=11, help="Enter a Bangladeshi mobile number")
    profile_type = st.selectbox("Profile Type",["Urban High Income","Urban Low Income","Rural"])

provider = detect_provider(phone_number)
user_kyc = kyc_info_for_number(phone_number)

with st.expander("🔎 Registered User KYC Information", expanded=True):
    k1, k2, k3, k4 = st.columns([1,2,1,1])
    with k1:
        st.markdown(f"<b>User Name:</b> {user_kyc['Name']}", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<b>Address:</b> {user_kyc['Address']}", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<b>Profession:</b> {user_kyc['Profession']}", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<b>Registered:</b> {user_kyc['Registration Date']}", unsafe_allow_html=True)
    if len(user_kyc["Other Numbers"]) > 0:
        st.info(f"Other mobile numbers on record: {', '.join(user_kyc['Other Numbers'])}")

# ----- Financial Assessment -----
df = generate_sample_transactions(phone_number, provider, profile_type)
monthly_bal = df.groupby(df['Date'].dt.to_period('M'))['Balance'].mean().reset_index()
avg_balance = int(df['Balance'].mean())
total_txn = int(df['Amount'].sum())
monthly_spend = int(df[df['Type']!='Cash-In']['Amount'].sum() / 12)
utilization = min(100, int(100 * (df[df['Type']!='Cash-In']['Amount'].sum()) / (avg_balance * 12)))
avg_deposits_per_month = int(df[df['Type']=="Cash-In"].groupby(df['Date'].dt.to_period('M')).size().mean())
payment_timeliness = f"{random.randint(93, 100)}%"  # Synthetic KPI for demo

st.write(f"#### Financial Assessment for: {phone_number} ({provider}), Profile: {profile_type}")

# KPIs
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    st.metric('AVG. BALANCE', f"৳{avg_balance:,}")
with c2:
    st.metric('TOTAL VOLUME', f"৳{total_txn:,}")
with c3:
    st.metric('MONTHLY SPEND', f"৳{monthly_spend:,}")
with c4:
    st.metric('UTILIZATION RATE', f"{utilization}%")
with c5:
    st.metric('DEPOSITS/MONTH', f"{avg_deposits_per_month:,}")

st.metric('PAYMENT TIMELINESS', payment_timeliness)

# ---- Main Chart ----
st.subheader("Balance Trend (12 months)")
trend = go.Figure()
trend.add_trace(go.Scatter(
    x=monthly_bal['Date'].astype(str),
    y=monthly_bal['Balance'],
    mode='lines+markers',
    name='Avg Balance',
    line=dict(color='#1064ea', width=3),
    marker=dict(size=8, color='#12b886')
))
trend.update_layout(
    margin=dict(t=20, b=15, l=0, r=0),
    xaxis_title="Month",
    yaxis_title="Average Balance",
    height=340,
    template='plotly_white',
    plot_bgcolor='#f5f7fa'
)
st.plotly_chart(trend, use_container_width=True)

col1, col2 = st.columns([2,1])
with col1:
    st.markdown("#### Spend by Category")
    piefig = go.Figure()
    cat_sums = df[df.Type!='Cash-In'].groupby('Type')['Amount'].sum().sort_values(ascending=False)
    piefig.add_trace(go.Pie(
        labels=cat_sums.index,
        values=cat_sums.values,
        hole=0.42,
        marker=dict(line=dict(color='#fff',width=2)),
        pull=[0.02]*len(cat_sums),
        sort=False
    ))
    piefig.update_traces(textinfo='percent+label')
    piefig.update_layout(showlegend=False, margin=dict(l=5, r=5, t=10, b=5))
    st.plotly_chart(piefig, use_container_width=True)
with col2:
    st.markdown("#### Top Vendors")
    top_merchants = df['Merchant'].value_counts().head(5)
    barfig = go.Figure()
    barfig.add_trace(go.Bar(
        x=top_merchants.values,
        y=top_merchants.index,
        orientation='h',
        marker=dict(color='#a6d8f8')
    ))
    barfig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10), height=225)
    st.plotly_chart(barfig, use_container_width=True)

st.divider()
st.subheader("Full Transaction History (Latest at top)")
st.dataframe(df.sort_values('Date', ascending=False).reset_index(drop=True), use_container_width=True, height=420)

st.caption("All data above is synthetic and for DaaS demonstration purposes. Change the mobile number for instant, unique assessment.")
