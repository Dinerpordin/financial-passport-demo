import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    np.random.seed(int(phone_number[-6:]))
    random.seed(int(phone_number[-6:]))
    days = 90

    base_profile = {
        "Urban High Income": {"base_balance": 100000, "txn_multiplier": 1.5},
        "Urban Low Income": {"base_balance": 20000, "txn_multiplier": 1.0},
        "Rural": {"base_balance": 5000, "txn_multiplier": 0.7}
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
                "Balance": round(max(current_balance,0), 2),
                "Provider": provider
            })
        date += timedelta(days=1)
    return pd.DataFrame(txns)

def simulate_transaction(provider):
    # Different weights for providers, can expand logic further
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
    else: # Cash-Out (withdraw)
        amt = random.randint(1000,5000)
    t_merchant = random.choice(merchants_pool[t_type])
    return t_type, amt, t_merchant

st.set_page_config(page_title="💳 Financial Passport Demo (Bangladesh)",layout="wide")

st.title("💳 Financial Passport Demo")
st.write("Simulated transaction history and analytics across Bangladeshi mobile providers. For DaaS Evaluation & Demo Purposes.")

st.sidebar.header("Demo Controls")
phone_number = st.sidebar.text_input("Enter Mobile Number (e.g. 017XXXXXXXX)", "01710000001",max_chars=11)
provider_select = st.sidebar.selectbox("Mobile Provider",["Auto detect"] + PROVIDERS)
profile_select = st.sidebar.selectbox("Profile Type",["Urban High Income","Urban Low Income","Rural"])

if provider_select=="Auto detect":
    provider = detect_provider(phone_number)
else:
    provider = provider_select

if st.sidebar.button("Generate Synthetic Transactions"):
    df = generate_sample_transactions(phone_number,provider,profile_select)
    st.session_state["transactions"] = df
elif "transactions" in st.session_state:
    df = st.session_state["transactions"]
else:
    df = None

if df is not None:
    st.subheader(f"Provider: {provider} | Profile: {profile_select}")
    st.write(df.tail(20))
    # Main Analysis Chart
    fig = px.line(df, x="Date", y="Balance", title="Balance Over Time")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Transactions by Category (Pie):**")
    piefig = px.pie(df, names='Type', values='Amount')
    st.plotly_chart(piefig, use_container_width=True)
    
    st.markdown("**Top Merchants (Bar):**")
    merchfig = px.bar(df.groupby('Merchant').Amount.sum().sort_values(ascending=False).head(8), labels={'y':'Total (Tk)'}, title="Top Merchants by Spend")
    st.plotly_chart(merchfig, use_container_width=True)

    st.info("Try switching your provider/profile in the sidebar and regenerating to simulate different scenarios.")

    st.download_button("Export as CSV", df.to_csv(index=False), file_name="synthetic_transactions.csv")
else:
    st.warning("Press 'Generate Synthetic Transactions' to view analytics.")

st.sidebar.info("All data is synthetic and randomly generated for demo purposes only.")

