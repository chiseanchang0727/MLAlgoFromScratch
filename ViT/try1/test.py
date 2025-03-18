from sentence_transformers import SentenceTransformer

# Choose a model from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"

import ssl
print(ssl.get_default_verify_paths())

import requests
requests.get("https://cdn-lfs.hf.co/sentence-transformers/all-MiniLM-L6-v2/", verify=False)


# Load and download the model
model = SentenceTransformer(model_name)

# Test the model with an example sentence
sentence = "This is an example sentence for embedding."
embedding = model.encode(sentence)

# Print the embedding shape
print(f"Model '{model_name}' downloaded successfully!")
print("Embedding shape:", embedding.shape)
