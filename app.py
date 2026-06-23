import pandas as pd
import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Streamlit UI
st.title("AI Data Insights App")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Preview data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Data summary
    st.subheader("Data Summary")
    st.write(df.describe())

    # User query
    user_query = st.text_input(
        "Enter your query about the data:"
    )

    if st.button("Generate"):

        if user_query:

            with st.spinner("Generating insights..."):

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a data analyst. "
                                "Analyze the dataset information provided "
                                "and answer the user's question."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"""
Dataset Columns:
{list(df.columns)}

Dataset Shape:
{df.shape}

Dataset Preview:
{df.head(10).to_string()}

Statistical Summary:
{df.describe().to_string()}

User Question:
{user_query}
"""
                        }
                    ]
                )

                answer = response.choices[0].message.content

                st.subheader("AI Response")
                st.write(answer)

        else:
            st.warning("Please enter a query.")