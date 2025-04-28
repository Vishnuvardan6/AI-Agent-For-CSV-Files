# agent/basic_data_agent.py

import pandas as pd
from transformers import pipeline
from google.colab import files

def main():

    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

    # Upload CSV
    uploaded = files.upload()

    for filename in uploaded.keys():
        df = pd.read_csv(filename)
        print(f"✅ Loaded {filename}")

    print("\nPreview:")
    print(df.head())

    question = input("\nAsk a question about your CSV: ")

    prompt = f"""Understand the user's intention based on the question.

Question: {question}

Tell if the user wants one of these:
- AVERAGE
- SUM
- COUNT
- FILTER (with condition)
- MAXIMUM
- MINIMUM
- LIST
- UNKNOWN
Only reply with the keyword in CAPITALS.
"""

    operation = qa_pipeline(prompt, max_length=10)[0]['generated_text'].strip().upper()

    print(f" Detected operation: {operation}")

    answer = "Sorry, I couldn't understand."

    try:
        if operation == "AVERAGE":
            col = input("Which column to average? ")
            answer = df[col].mean()

        elif operation == "SUM":
            col = input("Which column to sum? ")
            answer = df[col].sum()

        elif operation == "COUNT":
            answer = df.shape[0]

        elif operation == "FILTER":
            col = input("Which column to filter on? ")
            value = input("What value to filter by? ")
            filtered_df = df[df[col] == value]
            answer = filtered_df

        elif operation == "MAXIMUM":
            col = input("Which column to find maximum? ")
            answer = df[col].max()

        elif operation == "MINIMUM":
            col = input("Which column to find minimum? ")
            answer = df[col].min()

        elif operation == "LIST":
            col = input("Which column to list? ")
            answer = df[col].unique()

    except Exception as e:
        answer = f"Error: {e}"

    print("\nAnswer:")
    print(answer)

if __name__ == "__main__":
    main()
