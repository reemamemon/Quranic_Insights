import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import requests
from io import StringIO

# Function to clear memory
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Streamlit app
st.title("Quran Ayah Search and Response Generation")

# URL of the dataset on GitHub
dataset_url = "https://raw.githubusercontent.com/reemamemon/Quranic_Insights/main/The_Quran_Dataset.csv"

# Input for user query
input_text = st.text_input("Enter your query:")

# Define a placeholder for the generated response
generated_text_placeholder = st.empty()

# Add "Generate" button
if st.button("Generate"):
    if input_text:
        # Download the dataset from GitHub
        response = requests.get(dataset_url)
        response.raise_for_status()  # Ensure we notice bad responses

        # Load the dataset into a DataFrame
        df = pd.read_csv(StringIO(response.text))

        # Load a smaller model for generating embeddings
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize Faiss index
        dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        index = faiss.IndexFlatL2(dimension)

        # Encode the 'ayah_en' column into vectors using the smaller model
        vectors = embedder.encode(df['ayah_en'].tolist(), convert_to_numpy=True, show_progress_bar=False)
        
        # Convert to float32 to save memory
        vectors = vectors.astype(np.float32)

        # Add vectors to the index
        index.add(vectors)

        # Clear memory
        clear_memory()

        # Load the Granite model and tokenizer for generating the final response
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "ibm-granite/granite-3b-code-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=torch.float32)
        model.eval()

        # Encode the input query using the smaller model
        input_vector = embedder.encode([input_text], convert_to_numpy=True, show_progress_bar=False)

        # Perform search to find the top relevant Ayahs
        k = 5  # Number of nearest neighbors to retrieve
        distances, indices = index.search(input_vector.astype(np.float32), k)

        # Gather the retrieved Ayahs
        retrieved_ayahs = df.iloc[indices[0]]

        # Display the results with all columns
        st.write("Retrieved Ayahs:")
        for i, ayah in retrieved_ayahs.iterrows():
            st.write(f"\nAyah {i}:")
            for column, value in ayah.items():
                st.write(f"{column}: {value}")

        # Prepare the text for the language model
        retrieved_texts = "\n".join(retrieved_ayahs['ayah_en'])

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

        # Display the generated response
        generated_text_placeholder.text_area("Generated Response", generated_text)

        # Clear memory one last time
        clear_memory()
    else:
        st.warning("Please enter a query.")
