import os
import pandas as pd
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File

# 1. Setup Gemini API
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-pro')

app = FastAPI()

def calculate_disparate_impact(df, group_col, outcome_col):
    """Calculates basic bias metric: Disparate Impact"""
    # Probability of positive outcome for group A vs group B
    stats = df.groupby(group_col)[outcome_col].mean()
    # Simple ratio of min mean vs max mean
    bias_score = stats.min() / stats.max()
    return stats, bias_score

@app.post("/audit")
async def audit_dataset(file: UploadFile = File(...)):
    # Read the uploaded CSV
    df = pd.read_csv(file.file)
    
    # Example: Auditing 'Gender' against 'Hired' status
    # Note: In a real app, user would select these columns
    stats, score = calculate_disparate_impact(df, 'gender', 'hired')
    
    # 2. Use Gemini to interpret results
    prompt = f"""
    The AI model has a Disparate Impact score of {score:.2f}.
    Group stats: {stats.to_dict()}
    1. Explain if this is biased (ideal score is 1.0).
    2. Provide 3 steps to fix this bias in the training data.
    3. Suggest how to use Google Cloud Vertex AI to improve this.
    Keep the response professional and concise.
    """
    
    response = model.generate_content(prompt)
    
    return {
        "fairness_score": score,
        "raw_stats": stats.to_dict(),
        "gemini_insights": response.text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
