# Toronto Neighbourhood Recommendation System

https://github.com/user-attachments/assets/a97b4e6d-9e41-4b20-b084-e3a8f5e5615f

This project is a recommendation system that helps users find the best Toronto neighbourhoods based on their preferences. By leveraging demographic data from the City of Toronto, users can search for neighborhoods that match their interests using a language model and vector search.

## Features

- Uses OpenAI's `text-embedding-ada-002` model to generate neighbourhood embeddings from demographic data.
- Utilizes FAISS for fast similarity search between user queries and neighbourhood embeddings.
- Provides explanations of why specific neighbourhoods match user preferences.
- Interactive interface built using Streamlit.

## Project Structure

- **embed_city_of_toronto_demographics.py**: Generates embeddings for neighbourhoods based on demographic data.
- **gen_city_of_toronto_demographics.py**: Provides the recommendation system that matches user preferences with neighbourhood embeddings.
- **neighbourhood_embeddings.pkl**: Precomputed embeddings stored in this file.

## Technologies Used

- **Streamlit**: For the user interface.
- **OpenAI**: For generating neighborhood embeddings using the `text-embedding-ada-002` model.
- **FAISS**: For performing fast similarity searches between embeddings.
- **Pandas**: For data manipulation.
- **TQDM**: To display a progress bar during batch processing of embeddings.
- **scikit-learn**: For scaling and normalization of demographic data.
- **City of Toronto Open Data**: Data Source:https://open.toronto.ca/dataset/neighbourhood-profiles/

## Getting Started

### Prerequisites

- Python 3.8+
- API key for OpenAI (stored in `.env` file).

### Running the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/toronto-neighbourhood-recommendation.git
   cd toronto-neighbourhood-recommendation
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
3. Add your OpenAI API key to the .env file:
   ```bash
   OPENAI_API_TOKEN=your_openai_api_key
2. Launch the Streamlit app and find neighbourhoods based on your preferences, run:
   ```bash
   streamlit run gen_city_of_toronto_demographics.py
  
### Notes
- To satisfy rate limit requirements, some data from the original City of Toronto neighbourhood spreadsheet was removed
- The embeddings are precomputed and stored in neighbourhood_embeddings.pkl, so you can skip the embedding step if the embeddings are already generated.

