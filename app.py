import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime, timedelta
import random

# Provider Prefix Map
PROVIDER_PREFIX = {
    'Grameenphone': ['017'],
    'Robi': ['018'],
    'Banglalink': ['019'],
    'Teletalk': ['015'],
    'Airtel': ['016'],
}
PROVIDERS = list(PROVIDER_PREFIX.keys())

def detect_provider(phone):
    for provider, prefixes in PROVIDER_PREFIX.items():
        if any(phone.startswith(pref) for pref in prefixes):
            return provider
    return "Unknown"

def generate_sample_transactions(phone_number, provider, profile_type):
    np.random.seed(abs(hash(phone_number))%10**7)
    random.seed(abs(hash(phone_number))%10**7)
    months = 12
    days = months * 30

    base_profile = {
        "Urban High Income": {"base_balance": 150000, "txn_multiplier": 1.7},
        "Urban Low Income": {"base_balance": 25000, "txn_multiplier": 1.0},
        "Rural": {"base_balance": 6000, "txn_multiplier": 0.7}
    }
    profile = base_profile[profile_type]
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
    if t_type in ['Cash-In']:
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

# ---- UI and Dashboard ----

# Streamlit Page config
st.set_page_config(page_title="💳 Financial Passport - Professional DaaS Demo", layout="wide")
st.markdown("<style>div.block-container{padding-top:2rem;} .metric-label{font-weight:700; text-transform:uppercase; letter-spacing:0.04em;} .kpi{background:#f5f7fa;border-radius:12px;padding:1.2em 1em;text-align:center;margin-bottom:1em;box-shadow: 0 1px 8px #cfd8dc40;} .kpi .number{font-size:2rem; font-weight: bold; color:#17355a;}</style>", unsafe_allow_html=True)

# Auto-generation controls (hidden from user, but you can expose profile as needed)
phone_number = '01710000001'
provider = detect_provider(phone_number)
profile_type = "Urban High Income"

# ---- Data Generation (12 months, background) ----
df = generate_sample_transactions(phone_number, provider, profile_type)

# ---- Metrics Calculation ----
today = datetime.now()
twelve_months_ago = today - pd.DateOffset(months=12)
monthly_bal = df.groupby(df['Date'].dt.to_period('M'))['Balance'].mean().reset_index()
monthly_tx = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()
monthly_deposits = df[df.Type=='Cash-In'].groupby(df['Date'].dt.to_period('M'))['Amount'].sum().reset_index()

# KPIs
avg_balance = int(df['Balance'].mean())
total_txn = int(df['Amount'].sum())
monthly_spend = int(df[df['Type']!='Cash-In']['Amount'].sum() / 12)
utilization = min(100, int(100 * (df[df['Type']!='Cash-In']['Amount'].sum()) / (avg_balance * 12)))
avg_deposits_per_month = int(df[df['Type']=="Cash-In"].groupby(df['Date'].dt.to_period('M')).size().mean())
payment_timeliness = f"{random.randint(95, 100)}%"  # Synthetic for demo

# --- Dashboard Layout ---
st.title("💳 Financial Passport")
st.markdown("A *professional* DaaS demonstration with 12 months of synthetic transaction analytics. For demonstration use only.")

# ---- KPI Cards ----
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    st.markdown('<div class="kpi"><div class="metric-label">AVG. BALANCE</div><div class="number">৳{:,.0f}</div></div>'.format(avg_balance), unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi"><div class="metric-label">TOTAL TRANSACTIONS</div><div class="number">{:,}</div></div>'.format(total_txn), unsafe_allow_html=True)
with c3:
    st.markdown('<div class="kpi"><div class="metric-label">AVG. MONTHLY SPEND</div><div class="number">৳{:,.0f}</div></div>'.format(monthly_spend), unsafe_allow_html=True)
with c4:
    st.markdown('<div class="kpi"><div class="metric-label">CREDIT UTILIZATION</div><div class="number">{:d}%</div></div>'.format(utilization), unsafe_allow_html=True)
with c5:
    st.markdown('<div class="kpi"><div class="metric-label">DEPOSITS/MONTH</div><div class="number">{:,.0f}</div></div>'.format(avg_deposits_per_month), unsafe_allow_html=True)

# Timeliness (synthetic international metric)
st.markdown(f'<div class="kpi"><div class="metric-label">PAYMENT TIMELINESS</div><div class="number">{payment_timeliness}</div></div>', unsafe_allow_html=True)

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
    margin=dict(t=10, b=10, l=0, r=0),
    xaxis_title="Month",
    yaxis_title="Average Balance",
    height=340,
    template='plotly_white',
    plot_bgcolor='#f5f7fa'
)
st.plotly_chart(trend, use_container_width=True)

# --- Analytics Breakdown ---
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

st.caption("All data above is synthetic and for illustrative DaaS demo use only. Dashboard visual design inspired by leading international reporting matrices.")

