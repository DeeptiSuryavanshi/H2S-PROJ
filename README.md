# FairScan AI: Unbiased Decision Dashboard ⚖️

**FairScan AI** is an interactive dashboard designed to detect and mitigate machine learning bias in datasets. By leveraging **Streamlit** for the interface and **Google Gemini 1.5 Pro** for intelligent analysis, it helps developers and data scientists build fairer AI systems.

## ✨ Features
- **Data Auditing:** Calculate Disparate Impact scores to identify bias in outcomes.
- **Interactive Visualizations:** Real-time bar charts showing group distributions.
- **AI-Powered Insights:** Detailed bias patterns explained by Google Gemini.
- **Mitigation Strategies:** Actionable steps to fix training data bias.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **LLM:** Google Gemini 1.5 Pro
- **Data Processing:** Pandas

## 🚀 Quick Start
1. Install dependencies: `pip install streamlit pandas google-generativeai`
2. Add your Gemini API Key in `app.py`.
3. Run the app: `streamlit run app.py`