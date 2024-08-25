import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.credentials import Credentials
import requests
from io import StringIO
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get IBM Watsonx credentials from environment variables
credentials = Credentials(
    api_key=os.getenv("IBM_API_KEY"),
    url=os.getenv("IBM_URL")
)
project_id = os.getenv("PROJECT_ID")
model_id = "ibm/granite-13b-chat-v2"
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 600,
    "temperature": 0.2,
    "top_k": 1,
    "top_p": 1
}
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

def format_ayah_reference(ayah):
    return (
        f"**Arabic Reference from the Quran:**\n{ayah['ayah_ar']}\n\n"
        f"**Translation and Explanation:**\n{ayah['ayah_en']}\n\n"
        f"Para: {ayah['para']}, Surah: {ayah['surah']}, Ayah: {ayah['ayah_number']}"
    )

def generate_response(input_text, retrieved_ayahs):
    # Prepare the text for the IBM model
    retrieved_texts = "\n".join(retrieved_ayahs['ayah_en'])

    # Generate the response using the IBM model
    response = model.generate(
        prompt=input_text + "\n\n" + retrieved_texts
    )
    return response['text']

st.title("Islamic AI Assistant: Quran Ayah Search and Response Generation")

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
        response.raise_for_status()

        # Load the dataset into a DataFrame
        df = pd.read_csv(StringIO(response.text))

        # Load the sentence-transformers model
        embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # Initialize Faiss index
        dimension = 384
        index = faiss.IndexFlatL2(dimension)

        # Encode the 'ayah_en' column into vectors
        vectors = embedder.encode(df['ayah_en'].tolist(), convert_to_numpy=True)
        vectors = vectors.astype(np.float32)

        # Add vectors to the index
        index.add(vectors)

        # Encode the input query
        input_vector = embedder.encode([input_text], convert_to_numpy=True)

        # Perform search to find the top relevant Ayahs
        k = 5
        distances, indices = index.search(input_vector.astype(np.float32), k)

        # Gather the retrieved Ayahs
        retrieved_ayahs = df.iloc[indices[0]]

        # Display the results with all columns
        st.write("Retrieved Ayahs:")
        for i, ayah in retrieved_ayahs.iterrows():
            st.write(f"\nAyah {i}:")
            st.write(format_ayah_reference(ayah))

        # Generate the response
        generated_text = generate_response(input_text, retrieved_ayahs)

        # Display the generated response
        generated_text_placeholder.text_area("Generated Response", generated_text)
    else:
        st.warning("Please enter a query.")
