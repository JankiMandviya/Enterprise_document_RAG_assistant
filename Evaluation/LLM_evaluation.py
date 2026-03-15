import os
import re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset, Features, Value, Sequence
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

from ragas.embeddings import HuggingfaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

print("Starting evaluation...")

# -------------------------
# Load dataset
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the evaluation dataset from Excel
# This file contains queries, model answers, retrieved contexts, and ground truth answers
df = pd.read_excel(str(BASE_DIR/"Evaluation/LLM_evaluation.xlsx"))

# Rename columns to match the schema expected by RAG evaluation frameworks
# RAGAS expects: question, answer, contexts, ground_truth
df.rename(columns={
    "queries": "question",
    "Answers": "ground_truth"
}, inplace=True)

# Convert contexts column to string to avoid datatype issues while parsing
# Sometimes Excel loads cells as lists or NaN, which breaks downstream processing
df["contexts"] = df["contexts"].astype(str)

# Remove newline characters from context text to keep each context chunk clean
df["contexts"] = df["contexts"].str.replace("\n", " ", regex=False)

# Function to convert context string representation into a list of strings
# Example input:
# '["chunk1 text", "chunk2 text"]'
# Output:
# ["chunk1 text", "chunk2 text"]
def parse_contexts(x):
    x = x.strip("[]")    # Remove surrounding square brackets
    parts = re.split(r'","', x)    # Split contexts using delimiter between quoted strings
    parts = [p.strip().strip('"') for p in parts if p.strip()]   # Clean each chunk by removing extra quotes and whitespace
    return parts

# Apply parsing function so each row contains a list of context chunks
df["contexts"] = df["contexts"].apply(parse_contexts)

# Keep only the columns required for RAG evaluation
df = df[["question", "answer", "contexts", "ground_truth"]]

# Ensure all text fields are explicitly strings
# This avoids schema validation errors when creating a HuggingFace Dataset
df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)
df["ground_truth"] = df["ground_truth"].astype(str)

# Define dataset schema explicitly
# This prevents automatic type inference issues in the dataset loader
# Required format for RAG evaluation:
# question -> string
# answer -> string
# contexts -> list of strings
# ground_truth -> string
features = Features({
    "question": Value("string"),
    "answer": Value("string"),
    "contexts": Sequence(Value("string")),
    "ground_truth": Value("string")
})

# Remove rows with missing values in essential columns
# Missing values can cause evaluation or dataset creation to fail
df = df.dropna(subset=["question", "answer", "ground_truth"])  

# print a list of context chunks
print("Sample context:")
print(df["contexts"].iloc[0])

# -------------------------
# Convert to dataset
# -------------------------

dataset = Dataset.from_pandas(df, features=features, preserve_index=False)

# -------------------------
# Load embeddings
# -------------------------

embeddings = HuggingfaceEmbeddings(model_name="intfloat/e5-base-v2")

# -------------------------
# Load API key
# -------------------------

load_dotenv()
MISTRAL_API_KEY = os.getenv("Mistral_API")
# -------------------------
# Initialize LLM
# -------------------------

llm = ChatOpenAI(
    model="mistral-small-latest",
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",
    temperature=0
)

# -------------------------
# Run configuration
# -------------------------

run_config = RunConfig(
    max_workers=1,
    max_retries=5,
    timeout=180
)

# -------------------------
# Run evaluation
# -------------------------

result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ],
    llm=llm,
    embeddings=embeddings,
    run_config=run_config
)

print("\nEvaluation Results:")
print(result)

print("\nDetailed results:")
# print(result.to_pandas().head())