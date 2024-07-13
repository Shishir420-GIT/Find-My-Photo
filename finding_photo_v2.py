import os
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
# Setup
GOOGLE_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["image_database"]
collection = db["image_embeddings"]

# Gemini model setup
gemini_model = genai.GenerativeModel('models/gemini-1.5-pro')
embedding_model = 'models/embedding-001'

def get_image_details(image_path):
    """Make function API call for fetching details about the image."""
    image = Image.open(image_path)
    response = gemini_model.generate_content(["Describe this image in detail", image])
    return response.text

def create_text_embedding(text):
    """Convert the image data into text embeddings."""
    result = genai.embed_content(
        model=embedding_model,
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']

def process_images(image_list):
    """Process each image in the list."""
    for image_path in image_list:
        # Make function API call for fetching details about the image
        image_details = get_image_details(image_path)
        
        # Convert the image data into text embeddings
        embedding = create_text_embedding(image_details)
        
        # Create a DB record with Image Name, Path, and Abstract (Text Embedding)
        collection.insert_one({
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "abstract": image_details,
            "embedding": embedding
        })
        print(f"Processed Image {os.path.basename(image_path)}")
        # print({
        #     "image_name": os.path.basename(image_path),
        #     "image_path": image_path,
        #     "abstract": image_details,
        #     "embedding": embedding
        # })

def find_closest_image_old(query):
    """Find the closest vector (cosine distance) and fetch the record."""
    query_embedding = create_text_embedding(query)
    print("Query embed", query_embedding)
    # Find all documents in the collection
    all_docs = list(collection.find({}))
    print("Add docs", all_docs)
    # Calculate cosine similarities
    similarities = [cosine_similarity([query_embedding], [doc['embedding']])[0][0] for doc in all_docs]
    print("similarity cosine", similarities)
    # Find the index of the maximum similarity
    closest_index = np.argmax(similarities)
    
    return all_docs[closest_index]

def find_closest_image(query):
  query_embedding = create_text_embedding(query)
  print(f"Query embedding: {query_embedding[:5]}...") # Print first 5 elements
  all_docs = list(collection.find({}))
  print(f"Number of documents: {len(all_docs)}")
  similarities = []
  for i, doc in enumerate(all_docs):
    sim = cosine_similarity([query_embedding], [doc['embedding']])[0][0]
    similarities.append(sim)
    print(f"Document {i}: Similarity = {sim}, Embedding: {doc['embedding'][:5]}...")
  closest_index = np.argmax(similarities)
  print(f"Closest index: {closest_index}")
  print(f"Similarity scores: {similarities}")
  return all_docs[closest_index]

def main():
    # Import the list of Images
    image_list = [
        "/Users/shishir/Downloads/code_vipasana/Session 2 - Step 8 - Shishir.jpg",
        "/Users/shishir/Downloads/code_vipasana/Session 1 - Step 7 - Shishir.jpg",
        # Add more image paths as needed
    ]
    
    # Process all images
    process_images(image_list)
    
    # Create a prompt query option to fetch the Data
    query = input("Enter your image query: ")
    
    # Find the closest vector and fetch the record
    closest_image = find_closest_image(query)
    
    # Display the top image/images
    print(f"Closest matching image: {closest_image['image_name']}")
    print(f"Image path: {closest_image['image_path']}")
    print(f"Image description: {closest_image['abstract']}")

#if __name__ == "__main__":
    #main()
    # Import the list of Images
image_list = [
    "/Users/shishir/Pictures/Images/Zoro Roronoa HD Wallpapers.jpg",
    "/Users/shishir/Pictures/myImage.png",
    # Add more image paths as needed
]

# Process all images
process_images(image_list)

# Create a prompt query option to fetch the Data
query = input("Enter your image query: ")

# Find the closest vector and fetch the record
closest_image = find_closest_image(query)

# Display the top image/images
print(f"Closest matching image: {closest_image['image_name']}")
print(f"Image path: {closest_image['image_path']}")
print(f"Image description: {closest_image['abstract']}")