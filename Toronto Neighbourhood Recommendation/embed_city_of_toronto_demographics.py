import pandas as pd
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import time
from tqdm import tqdm
import pickle
import openai

load_dotenv(".env")
KEY = os.getenv('OPENAI_API_TOKEN')

print("Reading Spreadsheet")
df = pd.read_excel('neighbourhood-profiles-2021-158-model.xlsx')

# Transpose the DataFrame to make neighbourhoods the rows and demographic features the columns
df_transposed = df.T
df_transposed.columns = df_transposed.iloc[0]
df_transposed = df_transposed.drop(df_transposed.index[0])  
df_transposed.index.name = 'Neighbourhood' 
df_cleaned = df_transposed.reset_index()  

# Fill missing values with 0 and ensure columns have correct data types
df_filled = df_cleaned.fillna(0).infer_objects()

# Select numeric columns for scaling
numeric_columns = df_filled.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
df_filled[numeric_columns] = scaler.fit_transform(df_filled[numeric_columns])

print("Processing Data...")
# Create a description for each neighborhood by combining its demographic data into a single string
descriptions = df_filled.apply(lambda row: ', '.join([f"{col}: {row[col]}" for col in df_filled.columns if col != 'Neighbourhood']), axis=1)
df_description = pd.DataFrame({'Description': descriptions})

# Combine the cleaned DataFrame with the descriptions
df_filtered = pd.concat([df_filled, df_description], axis=1).copy() 
print("Data Processing Complete")

print("Creating Embeddings...")
# Initialize OpenAI embeddings model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=KEY)

# Function to create embeddings in batches to avoid rate limit errors
def batch_embeddings(descriptions, neighbourhoods, batch_size=10, sleep_time=1):
    """
    Generate embeddings for neighborhood descriptions in batches to prevent rate-limit errors.
    Args:
        descriptions (list): List of neighborhood descriptions.
        neighbourhoods (list): List of neighborhood names.
        batch_size (int): Number of descriptions to process in each batch.
        sleep_time (int): Time to sleep between batches in case of rate-limit errors.
    Returns:
        dict: Dictionary of neighborhood names and their corresponding embeddings.
    """
    embeddings = {}
    for i in tqdm(range(0, len(descriptions), batch_size)):  # Process descriptions in batches
        batch_descriptions = descriptions[i:i+batch_size]  # Extract batch of descriptions
        batch_neighbourhoods = neighbourhoods[i:i+batch_size]  # Extract corresponding batch of neighborhood names
        try:
            # Generate embeddings for the current batch
            batch_embeddings = [embedding_model.embed_query(desc) for desc in batch_descriptions]
            embeddings.update(dict(zip(batch_neighbourhoods, batch_embeddings)))  # Map neighbourhoods to embeddings
        except openai.RateLimitError:
            # Handle rate-limit error by sleeping and retrying
            print("Rate limit reached. Sleeping for a while...")
            time.sleep(sleep_time)
            batch_embeddings = [embedding_model.embed_query(desc) for desc in batch_descriptions]
            embeddings.update(dict(zip(batch_neighbourhoods, batch_embeddings)))  # Map neighbourhoods to embeddings
    return embeddings

# Create embeddings for each neighborhood in batches
neighbourhood_names = df_filtered['Neighbourhood'].tolist()
neighbourhood_descriptions = df_filtered['Description'].tolist()
neighbourhood_embeddings = batch_embeddings(neighbourhood_descriptions, neighbourhood_names)
print("Embeddings Created")


# Save the generated embeddings to a pickle file for future use
with open('neighbourhood_embeddings.pkl', 'wb') as f:
    pickle.dump(neighbourhood_embeddings, f)

print("Embeddings saved to 'neighbourhood_embeddings.pkl'.")
