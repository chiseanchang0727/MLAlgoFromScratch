from sentence_transformers import SentenceTransformer

# Choose a model from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load and download the model
model = SentenceTransformer(model_name)

# Test the model with an example sentence
sentence = "This is an example sentence for embedding."
embedding = model.encode(sentence)

# Print the embedding shape
print(f"Model '{model_name}' downloaded successfully!")
print("Embedding shape:", embedding.shape)
