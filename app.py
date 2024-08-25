from accelerate import Accelerator
import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import requests

# Function to clear memory
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Load the dataset directly from GitHub
csv_url = 'https://raw.githubusercontent.com/reemamemon/Quranic_Insights/main/The_Quran_Dataset.csv'
chunk_size = 1000  # Adjust based on your RAM constraints
df_iterator = pd.read_csv(csv_url, chunksize=chunk_size)

# Load a smaller model for generating embeddings
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Faiss index
dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
index = faiss.IndexFlatL2(dimension)

# Process data in chunks
for chunk in df_iterator:
    vectors = embedder.encode(chunk['ayah_en'].tolist(), convert_to_numpy=True, show_progress_bar=False)
    vectors = vectors.astype(np.float32)
    index.add(vectors)
    clear_memory()

# Load the Granite model and tokenizer for generating the final response
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "ibm-granite/granite-3b-code-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.float32 if device == "cuda" else torch.float32)
model.eval()

# Streamlit app layout
st.title("Islamic AI Assistant with Quranic Knowledge")
st.write("Ask a question about Islamic teachings, and the AI will provide a relevant Quranic verse and explanation.")

# User input
input_text = st.text_input("Enter your question here:")

if input_text:
    # Encode the input query using the smaller model
    input_vector = embedder.encode([input_text], convert_to_numpy=True, show_progress_bar=False)
    
    # Perform search to find the top relevant Ayahs
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = index.search(input_vector.astype(np.float32), k)
    
    # Gather the retrieved Ayahs with all columns
    retrieved_ayahs = []
    for idx in indices[0]:
        for chunk in pd.read_csv(csv_url, chunksize=chunk_size):
            if idx < len(chunk):
                retrieved_ayahs.append(chunk.iloc[idx])
                break
            idx -= len(chunk)

    # Prepare the text for the language model
    retrieved_texts = "\n".join([ayah['ayah_en'] for ayah in retrieved_ayahs])
    
    # Generate the final response using the Granite model
    input_tokens = tokenizer(retrieved_texts, return_tensors="pt", truncation=True, max_length=512)
    for i in input_tokens:
        input_tokens[i] = input_tokens[i].to(device)
    
    with torch.no_grad():
        output = model.generate(
            **input_tokens,
            max_length=512,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Display the results
    st.write("### Arabic Reference from the Quran:")
    for ayah in retrieved_ayahs:
        st.write(ayah['ayah_ar'])  # Display the Arabic Ayah

    st.write("### Translation and Explanation:")
    st.write(generated_text)  # Display the generated explanation

    st.write("### Para, Surah, Ayah Details:")
    for ayah in retrieved_ayahs:
        st.write(f"Para: {ayah['para']}, Surah: {ayah['surah']}, Ayah: {ayah['ayah']}")

# Clear memory one last time
clear_memory()

