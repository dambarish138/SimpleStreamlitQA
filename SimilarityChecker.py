import streamlit as st
from langchain.embeddings import OpenAIEmbedding
from langchain.vectorstores import Chroma
import openai
import numpy as np

# Set up your Azure OpenAI credentials
openai.api_key = "your_azure_openai_api_key"
openai.api_base = "your_azure_openai_api_endpoint"
openai.api_type = "azure"
openai.api_version = "v1"

# Initialize LangChain embeddings using Azure OpenAI
embedding_function = OpenAIEmbedding(api_key=openai.api_key, api_base=openai.api_base)

# Initialize ChromaDB
chroma_db = Chroma(embedding_function=embedding_function, collection_name="semantic_similarity")

# Function to compute semantic similarity
def compute_similarity(text1, text2):
    # Embed the texts
    embedding1 = embedding_function.embed(text1)
    embedding2 = embedding_function.embed(text2)
    
    # Add embeddings to ChromaDB
    chroma_db.add_document({"text": text1}, embedding1)
    chroma_db.add_document({"text": text2}, embedding2)
    
    # Retrieve the embeddings back from ChromaDB
    embeddings = chroma_db.query({"texts": [text1, text2]})
    embedding1, embedding2 = embeddings[0]["embedding"], embeddings[1]["embedding"]
    
    # Calculate the cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

# Streamlit app
st.title("Semantic Similarity Checker")

# Input text areas
text1 = st.text_area("Text 1")
text2 = st.text_area("Text 2")

if st.button("Check Similarity"):
    if text1 and text2:
        similarity_score = compute_similarity(text1, text2)
        st.write(f"Semantic Similarity Score: {similarity_score:.2f}")
    else:
        st.write("Please enter both texts to check similarity.")
