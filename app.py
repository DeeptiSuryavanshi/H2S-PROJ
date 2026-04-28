import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- CONFIGURATION ---
# Apni API Key yahan daalein
genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE") 
model = genai.GenerativeModel('gemini-1.5-pro')

# --- UI DESIGN ---
st.set_page_config(page_title="FairScan AI", layout="wide")

st.title("⚖️ FairScan AI: Unbiased Decision Dashboard")
st.markdown("### Detect and Mitigate AI Bias using Google Gemini")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("Settings")
target_column = st.sidebar.text_input("Column to Audit (e.g., gender)", "gender")
outcome_column = st.sidebar.text_input("Outcome Column (e.g., hired)", "hired")

# --- MAIN SECTION ---
# FIXED: file_file ko file_uploader kar diya hai
uploaded_file = st.file_uploader("Upload your Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # Calculation logic
    if target_column in df.columns and outcome_column in df.columns:
        stats = df.groupby(target_column)[outcome_column].mean()
        # Bias calculation (ratio of min mean to max mean)
        bias_score = stats.min() / stats.max() if stats.max() != 0 else 0

        with col2:
            st.subheader("Bias Scorecard")
            st.metric(label="Fairness Score (Disparate Impact)", value=f"{bias_score:.2f}")
            st.bar_chart(stats)
            
            if bias_score < 0.8:
                st.warning("⚠️ High Bias Detected! (Score below 0.8 is generally considered biased)")
            else:
                st.success("✅ Model appears to be Fair.")

        # --- GEMINI INSIGHTS ---
        st.divider()
        st.subheader("🤖 Gemini AI Insights & Mitigation Steps")
        
        if st.button("Generate AI Insights"):
            with st.spinner("Gemini is analyzing your data..."):
                prompt = f"""
                The dataset shows a fairness score of {bias_score:.2f} for '{target_column}' against '{outcome_column}'.
                Group Statistics: {stats.to_dict()}
                1. Explain in simple English why this result is biased.
                2. Provide 3 technical steps to fix this bias in the training data.
                3. Suggest one Google Cloud tool to help manage this.
                """
                response = model.generate_content(prompt)
                st.write(response.text)
            
    else:
        st.error(f"Columns '{target_column}' or '{outcome_column}' not found. Please check spelling in your CSV.")

else:
    st.info("Waiting for CSV upload... Use your 'test_data.csv' to see it in action.")

