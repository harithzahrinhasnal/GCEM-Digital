import streamlit as st
from PIL import Image
import os
from google.cloud import vision
from google.cloud.vision import types
import sqlite3

# Initialize database connection
def init_db():
    conn = sqlite3.connect('business_cards.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS BusinessCards (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            company TEXT,
            address TEXT
        )
    ''')
    conn.commit()
    return conn

# Set the path to the service account key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'path_to_your_service_account_key.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

# Extract data from image using Google Cloud Vision API
def extract_data(image):
    content = image.tobytes()
    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    text = texts[0].description if texts else ""

    # Basic parsing (customize as needed)
    name = "Unknown"  # Add logic to extract name
    email = "Unknown" if '@' not in text else [line for line in text.split('\n') if '@' in line][0]
    phone = "Unknown"  # Add logic to extract phone
    company = "Unknown"  # Add logic to extract company
    address = "Unknown"  # Add logic to extract address
    return name, email, phone, company, address


# Save data to database
def save_to_db(conn, data):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO BusinessCards (name, email, phone, company, address)
        VALUES (?, ?, ?, ?, ?)
    ''', data)
    conn.commit()

# App UI
st.title("Business Card Uploader")
conn = init_db()

# Upload business card
uploaded_file = st.file_uploader("Upload Business Card", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Business Card", use_column_width=True)
    if st.button("Extract and Save"):
        extracted_data = extract_data(image)
        save_to_db(conn, extracted_data)
        st.success("Business card saved successfully!")

# Display all cards
if st.checkbox("Show Saved Cards"):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM BusinessCards")
    rows = cursor.fetchall()
    for row in rows:
        st.write(f"Name: {row[1]}, Email: {row[2]}, Phone: {row[3]}, Company: {row[4]}, Address: {row[5]}")
