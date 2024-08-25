import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

# Function to clear memory
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

# Streamlit app
st.title("Quran Ayah Search and Response Generation")

# File uploader for CSV
csv_file = st.file_uploader("Upload the Quran Dataset CSV", type=["csv"])

# Input for user query
input_text = st.text_input("Enter your query:", value="what is the order about the believer?")

# Define a placeholder for the generated response
generated_text_placeholder = st.empty()

# Proceed if file is uploaded and query is provided
if csv_file is not None and input_text:

    # Load the dataset in chunks
    chunk_size = 1000  # Adjust based on your RAM constraints
    df_iterator = pd.read_csv(csv_file, chunksize=chunk_size)

    # Load a smaller model for generating embeddings
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Initialize Faiss index
    dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
    index = faiss.IndexFlatL2(dimension)

    # Process data in chunks
    for chunk in df_iterator:
        # Encode the 'ayah_en' column into vectors using the smaller model
        vectors = embedder.encode(chunk['ayah_en'].tolist(), convert_to_numpy=True, show_progress_bar=False)

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

    # Gather the retrieved Ayahs with all columns
    retrieved_ayahs = []
    for idx in indices[0]:
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            if idx < len(chunk):
                retrieved_ayahs.append(chunk.iloc[idx])
                break
            idx -= len(chunk)

    # Display the results with all columns
    st.write("Retrieved Ayahs:")
    for i, ayah in enumerate(retrieved_ayahs, 1):
        st.write(f"\nAyah {i}:")
        for column, value in ayah.items():
            st.write(f"{column}: {value}")

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

    # Display the generated response
    generated_text_placeholder.text_area("Generated Response", generated_text)

    # Clear memory one last time
    clear_memory()
