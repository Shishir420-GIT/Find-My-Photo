# Find-My-Photo
Remember when you suddenly are reminded that you had a cool picture in a blue shirt or a yellow summer dress, but just can't find the picture in the huge stack of images you have? This GenAI app is here to solve that problem.

# Pre-requisites
To run this app, you'll need to have the following prepared:
- MongoDB installed
- Google API Key
- Python
  
# How it Works
This app uses the Google Gemini API to generate a detailed description of an image, and then converts that description into an embedding using the models/embedding-001 model. The embedding is then stored in a MongoDB database with geospatial indexing.

When you query the app with a description of an image, it converts the query into an embedding and uses the $near operator to find the closest matching image in the database.

# Code
The code for this app is written in Python and uses the following libraries:
- google.generativeai for interacting with the Google Gemini API
- pymongo for interacting with the MongoDB database
- PIL for opening and processing images
- logging for error handling

# Usage
To use this app, simply run the main.py file and follow the prompts. You'll be asked to enter a query describing the image you're looking for, and the app will return the closest matching image from the database.

# Note
This app is still in development, and you may need to adjust the image_list variable in the main.py file to point to the images you want to process. Additionally, you'll need to replace the GOOGLE_API_KEY variable with your own Google API Key.
