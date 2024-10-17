import streamlit as st
import pickle
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.docstore.in_memory import InMemoryDocstore
import os
from dotenv import load_dotenv

st.title("Toronto Neighbourhood Finder")
st.write("Find the best Toronto neighbourhood based on your preferences.")

load_dotenv(".env")
KEY = os.getenv('OPENAI_API_TOKEN')

# Load precomputed neighbourhood embeddings
# The pickle file contains a dictionary where the keys are neighbourhood names, and the values are their corresponding embedding vectors
with open('neighbourhood_embeddings.pkl', 'rb') as f:
    neighbourhood_embeddings = pickle.load(f)

# Prepare embeddings for FAISS
embedding_vectors = np.array(list(neighbourhood_embeddings.values()))  # Extract embedding vectors
neighbourhood_names = list(neighbourhood_embeddings.keys())  # Extract neighbourhood names

# Initialize FAISS index and embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=KEY)
dimension = embedding_vectors.shape[1]  # Get the length of the embedding vector (1536 dimensions in this case)
index = faiss.IndexFlatL2(dimension)  
index.add(embedding_vectors)

# Create an InMemoryDocstore to store metadata (neighbourhood names)
# The docstore stores a dictionary where each key is the index of a neighbourhood, and the value is its metadata (the neighbourhood name)
docstore = InMemoryDocstore({i: {"Neighbourhood": neighbourhood_names[i]} for i in range(len(neighbourhood_names))})

# Create FAISS vector store with InMemoryDocstore
faiss_store = FAISS(
    embedding_function=embedding_model.embed_query, 
    index=index,
    docstore=docstore,
    index_to_docstore_id={i: i for i in range(len(neighbourhood_names))}  # Map FAISS indices to docstore IDs
)

# Initialize OpenAI language model
llm = OpenAI(api_key=KEY)

# Get user input to describe their ideal neighbourhood
user_query = st.text_input("Describe the neighbourhood you're looking for:")

# Only search when the user submits the query
if st.button("Find Neighbourhoods"):
    # Generate embedding for the user's query
    user_embedding = embedding_model.embed_query(user_query)
    user_embedding = np.array(user_embedding).reshape(1, -1)  # Reshape to 2D array for FAISS

    # Perform a FAISS search to find the 3 closest neighbourhoods
    D, I = index.search(user_embedding, k=3)  # D: distances, I: indices of neighbors

    # Retrieve metadata for similar neighbourhoods based on FAISS search results
    similar_neighbourhoods = []
    for idx in I[0]:  # Iterate over indices of the nearest neighbors
        metadata = docstore.search(idx)  # Get metadata for each neighbourhood
        similar_neighbourhoods.append({"id": idx, "metadata": metadata})

    # Generate explanations for why each neighbourhood is a good match
    for neighbourhood_index in similar_neighbourhoods:
        neighbourhood_id = neighbourhood_index['id']  # Get the neighbourhood id
        metadata = neighbourhood_index['metadata']  # Get the metadata (neighbourhood name)
        
        # Create a prompt for OpenAI to generate an explanation based on the user's query
        explanation_prompt = (
            f"User's neighbourhood interests: {user_query}. "
            f"Explain why the neighbourhood {metadata['Neighbourhood']} is a good match based on its demographic data: {metadata}"
        )
        
        # Get response from OpenAI language model
        response = llm.generate([explanation_prompt])
        explanation = response.generations[0][0].text.strip()

        # Display the neighbourhood name and explanation
        st.subheader(f"Neighbourhood: {metadata['Neighbourhood']}")
        st.write(explanation)