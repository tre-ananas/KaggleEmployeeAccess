# GovDocsOperational.py
# 10/10/23

# DESCRIPTION
# Process the data scraped by GovDocs_Step1_Selenium

# END OF TITLE AND DESCRIPTION

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 1: LIBRARIES AND SETTINGS

# Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import PyPDF2
from io import BytesIO
from tqdm import tqdm
import time
import csv
import sys
from random import randint
import csv

# Other Settings
# pd.set_option('display.max.colwidth', None) # max display width

# END OF STEP 1

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 2: LOAD DATA

# Load data from CSV file into a DataFrame
article_link_directory = pd.read_csv('article_link_directory.csv')

# END OF STEP 2

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 3: COLLECT PDF CONTENT

# Initialize an empty list to collect text data from PDFs
pdf_text_data = []

for pdf_url in tqdm(article_link_directory['Full Collected Links'], desc = "Step 2: Collecting PDF Text", unit = "link"):
    
    # Check if the URL contains the NO CONTENT alert
    if "NO CONTENT" in pdf_url:
        # If yes, append "NO CONTENT" to pdf_text_data and continue to the next URL
        pdf_text_data.append("NO CONTENT")

    else:
        try:
            # Send an HTTP GET request to the PDF URL
            response = requests.get(pdf_url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Wrap the response content in a BytesIO object
                pdf_bytes = BytesIO(response.content)

                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdf_bytes)

                # Initialize an empty string to store the text data
                text_data = ""

                # Use len(reader.pages) to determine the number of pages
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_data += page.extract_text()

                # Append the extracted text to the pdf_text_data list
                pdf_text_data.append(text_data)
            else:
                print(f"Failed to fetch PDF URL: {pdf_url}, Status code: {response.status_code}")
                pdf_text_data.append("NO CONTENT")

        except Exception as e:
            print(f"An error occurred while processing PDF URL: {pdf_url}, Error: {str(e)}")
            pdf_text_data.append("NO CONTENT")

    # Introduce a random delay time before the next request
    time.sleep(3)  # Adjust the delay time as needed

# END OF STEP 3

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 4: ADD INSIDE PDF TEXT TO article_link_directory AND EXPORT AS .csv

# Create a new column "Inside PDF Text" in article_link_directory and assign pdf_text_data to it
article_link_directory['Inside PDF Text'] = pdf_text_data

# Save the DataFrame to a CSV file
article_link_directory.to_csv('article_link_directory_with_pdf_content.csv', index=False)

# END OF STEP 4

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------