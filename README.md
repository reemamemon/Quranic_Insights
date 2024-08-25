
# Quranic Knowledge AI Chatbot

This is a Quranic Knowledge Chatbot designed to provide answers directly from the Quran in Arabic, followed by translations and explanations in the same language as the user's input. The chatbot also provides detailed references, including Para, Surah, and Ayah numbers, for each answer. The model is specifically designed to answer queries related to the Quran.

## Features

- **Quranic Verses in Arabic:** Provides the relevant Quranic verse in Arabic for each query.
- **Language-Specific Explanations:** Delivers explanations and translations in the user's input language.
- **Detailed References:** Includes Para, Surah, and Ayah numbers for each Quranic reference.

## Requirements

Before running the app, ensure that you have the following Python packages installed. You can install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Dataset

The app expects a CSV file containing Quranic Ayahs. Each row in the CSV file should represent an Ayah, and the columns should include:

- `ayah_en`: The English translation of the Ayah.

### 2. Running the App

To run the app, use the following command:

```bash
streamlit run app.py
```

### 3. Using the App

- **File Upload**: The app automatically loads the dataset named `The Quran Dataset.csv`.
- **Enter Query**: Type your query in the input box (e.g., "what is the order about the believer?").
- **View Results**: The app will display the top relevant Ayahs along with their details.
- **Generated Response**: The app will generate a response based on the retrieved Ayahs using the Granite model.

## Code Overview

### 1. Memory Management

The app uses garbage collection (`gc.collect()`) and clears CUDA memory (`torch.cuda.empty_cache()`) to manage memory usage efficiently.

### 2. Embedding Generation

The `SentenceTransformer` model `all-MiniLM-L6-v2` is used to generate embeddings for the Ayahs, which are then indexed using FAISS.

### 3. FAISS Search

The embeddings are indexed using FAISS, and the app searches for the most similar Ayahs to the user's query.

### 4. Response Generation

The Granite model from Hugging Face is used to generate the final response, which is then displayed in the app.

## Notes

- **Performance**: Depending on the size of your dataset, processing may take some time, especially on CPU. Consider using a GPU for faster performance.
- **Model Selection**: The Granite model is a large language model, so ensure that your hardware can support it.

## Acknowledgments

- **Streamlit**: For providing an easy-to-use interface for web apps.
- **FAISS**: For efficient similarity search.
- **Hugging Face**: For the Transformers library and pre-trained models.
- **SentenceTransformers**: For easy-to-use sentence embedding models.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

### Explanation:
- **Introduction and Features**: An overview of what the app does.
- **Requirements**: Provides instructions on dependencies.
- **Usage**: Step-by-step guide on how to run and use the app.
- **Code Overview**: Brief explanation of the main code components.
- **Acknowledgments and License**: Credits and licensing information.

This `README.md` file should give users a clear understanding of how to set up and use your app.
