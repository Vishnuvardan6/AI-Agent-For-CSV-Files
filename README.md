Building a lightweight AI agent that can intelligently answer basic questions from a simple CSV file containing customer purchase data or any client data.

The agent should be able to:

  -Parse the CSV file.

  -Understand simple user queries in natural language.

  -Perform appropriate data lookups and respond with correct answers.

The Model uses Pandas to parse the CSV files into DataFrames.

The Model uses HuggingFace Model - "google/flan-t5-small" and The pipeline - "text2text-generation" For the Question and Answering.
