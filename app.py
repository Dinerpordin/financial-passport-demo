import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# --- Function 1: Generate Sample Transactions ---
def generate_sample_transactions(phone_number):
    # Deterministic seed from phone number string
    seed = abs(hash(phone_number)) % (2**32)
    np.random.seed(seed)
    
    # Transaction types and their weights
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
    bal = np.random.randint(500, 10000)  # starting balance

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
        else:  # Mobile Recharge
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
    df = df.sort_values('date')
    return df

# --- Function 2: Calculate Credit Score ---
def calculate_credit_score(transaction_df):
    df = transaction_df.copy()
    df['month'] = df['date'].dt.to_period('M')

    # Feature Engineering
    avg_balance = df.groupby('month')['balance'].mean().mean()
    income_stability = (
        df[df['type'].isin(['Cash-In', 'Send Money'])]['amount'].sum()
        / df['amount'].sum()
    )
    tx_frequency = len(df) / 6  # last 6 months
    savings_ratio = avg_balance / df['amount'].mean()

    # Scoring Formula
    score = 0
    score += (avg_balance / 10000) * 30  # Max 30 points
    score += income_stability * 30        # Max 30 points
    score += min(tx_frequency / 10, 20)   # Max 20 points
    score += min(savings_ratio * 10, 20)  # Max 20 points
    final_score = max(0, min(int(score), 100))  # Ensure 0-100
    
    breakdown_dict = {
        'Average Balance': f"৳{int(avg_balance):,}",
        'Income Stability': f"{income_stability*100:.0f}%",
        'Monthly Transactions': f"{int(tx_frequency)}",
        'Savings Ratio': f"{savings_ratio:.2f}",
    }
    return final_score, breakdown_dict

# --- Helper: Gauge Rendering ---
def gauge(score):
    fig, ax = plt.subplots(figsize=(2, 1))
    cmap = plt.get_cmap("RdYlGn")
    normed = (score-0)/100
    color = cmap(normed)
    ax.barh([0], [score], color=color, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(score, 0, f'{score}', va='center', ha='left', fontsize=20, color='black')
    plt.close(fig)
    return fig

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="💳 Financial Passport Demo", layout="centered")
    st.title("💳 Financial Passport Demo")
    st.write("This demo generates a credit score from synthetic mobile money transactions. Enter your phone to see how it works.")

    phone = st.text_input(
        "Enter a sample phone number (e.g., 01712XXXXXX):", value="01712345678"
    )
    if st.button("Generate Financial Passport"):
        with st.spinner('Analyzing transaction data...'):
            df = generate_sample_transactions(phone)
            score, breakdown = calculate_credit_score(df)

        # Score Display
        st.subheader("Credit Score")
        score_color = "red" if score < 40 else "orange" if score < 70 else "green"
        st.markdown(
            f'<h1 style="color:{score_color};">{score}/100</h1>',
            unsafe_allow_html=True,
        )
        st.progress(score)
        st.pyplot(gauge(score))

        # Metrics
        st.subheader("Score Breakdown")
        cols = st.columns(len(breakdown))
        for i, (label, val) in enumerate(breakdown.items()):
            cols[i].metric(label, val)

        # Data Preview
        st.subheader("Recent Transactions Preview")
        preview = df.tail(10).copy()
        preview['amount'] = preview['amount'].apply(lambda x: f"৳{x:,}")
        preview['balance'] = preview['balance'].apply(lambda x: f"৳{x:,}")
        st.dataframe(preview[['date', 'type', 'amount', 'balance']].reset_index(drop=True))

        # Explanation
        st.subheader("🧠 How This Score Was Calculated")
        st.markdown("""
- **Average Balance**: The typical balance left in your mobile wallet each month.
- **Income Stability**: The share of money 'coming in' (Cash-In/Send Money) compared to all transactions. More stable income means a higher score.
- **Monthly Transactions**: The number of transactions each month. Higher activity shows financial engagement.
- **Savings Ratio**: How your average wallet balance compares to your typical transaction amount. Saving more boosts this ratio.
        """)

if __name__ == "__main__":
    main()
