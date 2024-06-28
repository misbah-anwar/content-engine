# Import required libraries
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import streamlit as st

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Paths to the PDF documents
pdf_paths = {
    "Alphabet Inc.": "alphabet_10K.pdf",
    "Tesla Inc.": "tesla_10K.pdf",
    "Uber Technologies Inc.": "uber_10K.pdf"
}

# Extract text from the PDFs
document_texts = {company: extract_text_from_pdf(path) for company, path in pdf_paths.items()}

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for document content
document_embeddings = {company: embedding_model.encode(text) for company, text in document_texts.items()}

# Combine all embeddings into a single matrix and create an index
all_embeddings = np.vstack(list(document_embeddings.values()))
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings)

# Store document names with their corresponding index ranges
document_ranges = {}
start_idx = 0
for company, embeddings in document_embeddings.items():
    end_idx = start_idx + len(embeddings)
    document_ranges[company] = (start_idx, end_idx)
    start_idx = end_idx

# Function to search for relevant document sections
def search(query):
    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    results = []
    for idx in I[0]:
        for company, (start, end) in document_ranges.items():
            if start <= idx < end:
                results.append((company, document_texts[company][idx-start:idx-start+500]))
                break
    return results

# Function to generate response using the Groq API
def generate_response(prompt, api_key):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        'prompt': prompt,
        'max_tokens': 150,
    }
    response = requests.post('https://api.groq.com/v1/engines/text-davinci-003/completions', headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['text']
    else:
        return "Error: Could not generate response."

# Streamlit interface
st.title("Content Engine Chatbot")

query = st.text_input("Enter your query:")

# Your Groq API key
api_key = os.getenv("GROQ_API_KEY")

if st.button("Submit"):
    search_results = search(query)
    st.write("Search Results:")
    for company, result in search_results:
        st.write(f"**{company}:** {result}")

    response = generate_response(query, api_key)
    st.write("Generated Response:")
    st.write(response)
