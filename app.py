import pandas as pd
import streamlit as st
from openai import OpenAI
import os 
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.title("AI Data Insights App")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())
    user_query = st.text_input("Enter your query about the data:")
    if user_query:
        prompt = f"Given the following data:\n{df.to_string()}\n\nAnswer the following question: {user_query}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        answer = response.choices[0].message.content
        st.write("AI Response:")
        st.write(answer)
    st.subheader("Data Summary:")
    st.write(df.describe())
