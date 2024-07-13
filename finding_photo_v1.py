import os
from dotenv import load_dotenv
import google.generativeai as genai
from pymongo import MongoClient
from bson.son import SON
from PIL import Image
import logging

logging.basicConfig(level=logging.ERROR)

load_dotenv()

# Setup
GOOGLE_API_KEY = os.getenv("API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API_KEY not found in environment variables")
genai.configure(api_key=GOOGLE_API_KEY)

# MongoDB setup with geospatial indexing
client = MongoClient("mongodb://localhost:27017/")
db = client["image_database"]
collection = db["image_embeddings"]
collection.create_index([("embedding", "2dsphere")])

# Gemini model setup
gemini_model = genai.GenerativeModel('models/gemini-1.5-pro')
embedding_model = 'models/embedding-001'

def get_image_details(image_path):
    """Fetch details about the image."""
    try:
        image = Image.open(image_path)
        response = gemini_model.generate_content(["Describe this image in detail", image])
        return response.text
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

def create_text_embedding(text):
    """Convert text into embeddings."""
    try:
        result = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        # Ensure the embedding is a GeoJSON object
        embedding = {
            "type": "Point",
            "coordinates": result['embedding']
        }
        return embedding
    except Exception as e:
        logging.error(f"Error creating embedding for text: {e}")
        return None

def process_images(image_list):
    """Process each image in the list."""
    for image_path in image_list:
        if not os.path.exists(image_path):
            logging.error(f"Image path does not exist: {image_path}")
            continue
        image_details = get_image_details(image_path)
        if not image_details:
            continue
        embedding = create_text_embedding(image_details)
        if not embedding:
            continue
        collection.insert_one({
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "abstract": image_details,
            "embedding": embedding
        })
        print(f"Processed Image {os.path.basename(image_path)}")

def find_closest_image(query):
    """Find the closest image using geospatial search."""
    query_embedding = create_text_embedding(query)
    if not query_embedding:
        return None

    # Use the $near operator for geospatial queries
    closest_image = collection.find_one({
        "embedding": {
            "$near": {
                "$geometry": query_embedding,
                "$maxDistance": 0.5  # Adjust threshold for similarity (in meters)
            }
        }
    })

    return closest_image

def main():
    image_list = [
        "/Users/shishir/Pictures/Images/Zoro Roronoa HD Wallpapers.jpg",
        "/Users/shishir/Pictures/myImage.png",
    ]
    process_images(image_list)
    query = input("Enter your image query: ")
    closest_image = find_closest_image(query)
    if closest_image:
        print(f"Closest matching image: {closest_image['image_name']}")
        print(f"Image path: {closest_image['image_path']}")
        print(f"Image description: {closest_image['abstract']}")
    else:
        print("No image found matching your query.")

if __name__ == "__main__":
    main()
