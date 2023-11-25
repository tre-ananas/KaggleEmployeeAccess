# SentimentAnalysis.py
# 11/20/23

# DESCRIPTION
# This code takes .csv data produced by the GovDocsOperational.py script and outputs a .csv with sentiment and keyword data.
# This data can be loaded into R or Python for further analysis, including keyword frequency/proportion, sentiment by keyword presence, and sentiment across time.

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 1: LIBRARIES AND SETTINGS

# Import Packages and Model Commands
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
import pandas as pd
import nltk
from tqdm import tqdm
import datetime
import sys
from nltk.corpus import stopwords

# Swedish stopwords
nltk.download('stopwords')
stop_words_swedish = set(stopwords.words('swedish'))

# Download the NLTK sentence tokenizer data
nltk.download("punkt")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 2: LOAD MODELS FROM HUGGING FACE - RECORDED FUTURE

# Load pre-trained model and tokenizer from Recorded Future
tokenizer = BertTokenizerFast.from_pretrained("RecordedFuture/Swedish-Sentiment-Fear")
classifier_fear= BertForSequenceClassification.from_pretrained("RecordedFuture/Swedish-Sentiment-Fear")
classifier_violence = BertForSequenceClassification.from_pretrained("RecordedFuture/Swedish-Sentiment-Violence")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 3: LOAD DATA

# Check if the correct number of command-line arguments are provided
if len(sys.argv) != 3:
    print("Usage: python script_name.py input_file_name.csv output_file_name.csv")
    sys.exit(1)

# Get input and output file names from command-line arguments
input_file_name = sys.argv[1]
output_file_name = sys.argv[2]

# Load Data
try:
    unprocessed_data = pd.read_csv(input_file_name)
    unprocessed_data = unprocessed_data.rename(columns={"Unnamed: 0": "Index"})
    unprocessed_data = unprocessed_data[unprocessed_data['Text'] != 'NO CONTENT']
except FileNotFoundError:
    print(f"Error: Input file '{input_file_name}' not found.")
    sys.exit(1)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# STEP 6: CLASSIFY ARTICLES AS A WHOLE WITH NO STOPWORDS

# Lists to store probabilities, texts, and IDs for both classifiers
probabilities_fear_list_2 = []
probabilities_violence_list_2 = []
texts_list_2 = []
ids_list_2 = []

text_entries_2 = unprocessed_data['Text'].tolist()
ids_2 = unprocessed_data['Index'].tolist()

for text, text_id in tqdm(zip(text_entries_2, ids_2), total=len(text_entries_2), desc="Processing Texts"):
    try:
        # Lowercase the entire text
        text_lower = text.lower()

        # Tokenize and remove stop words
        tokens = nltk.word_tokenize(text_lower)
        tokens_filtered = [word for word in tokens if word.isalnum() and word not in stop_words_swedish]

        # Reconstruct the text
        text_filtered = ' '.join(tokens_filtered)

        # Process the entire text at once
        inputs = tokenizer(text_filtered, return_tensors="pt", truncation=True, max_length=512)

        # Calculate probabilities for classifier_fear
        outputs_fear = classifier_fear(**inputs)
        probabilities_fear = torch.nn.functional.softmax(outputs_fear.logits, dim=1).tolist()[0]
        probabilities_fear_list_2.append(probabilities_fear)

        # Calculate probabilities for classifier_violence
        outputs_violence = classifier_violence(**inputs)
        probabilities_violence = torch.nn.functional.softmax(outputs_violence.logits, dim=1).tolist()[0]
        probabilities_violence_list_2.append(probabilities_violence)

        texts_list_2.append(text_filtered)
        ids_list_2.append(text_id)
    except Exception as e:
        # If an error occurs, print the error and append zeros to the probability lists
        print(f"Error processing text with ID {text_id}: {e}")
        probabilities_fear_list_2.append([0.0, 0.0, 0.0])
        probabilities_violence_list_2.append([0.0, 0.0, 0.0])
        texts_list_2.append("Error processing text")
        ids_list_2.append(text_id)

# Now, probabilities_fear_list_2 and probabilities_violence_list_2 contain probabilities
# for classifier_fear and classifier_violence respectively, for each text.

# Creating the classified_sentence_data DataFrame
processed_data = pd.DataFrame({
    'ID_2': ids_list_2,
    'Date': unprocessed_data['Date'],
    'Fear Class 0 Probability_2': [item[0] for item in probabilities_fear_list_2],
    'Fear Class 1 Probability_2': [item[1] for item in probabilities_fear_list_2],
    'Fear Class 2 Probability_2': [item[2] for item in probabilities_fear_list_2],
    'Violence Class 0 Probability_2': [item[0] for item in probabilities_violence_list_2],
    'Violence Class 1 Probability_2': [item[1] for item in probabilities_violence_list_2],
    'Violence Class 2 Probability_2': [item[2] for item in probabilities_violence_list_2],
    'Document Type': unprocessed_data['Document Type'],
    'Source': unprocessed_data['Source'],
    'Text_2': texts_list_2,
    'PDF Indicator': unprocessed_data['PDF Indicator'],
    'Content Page': unprocessed_data['Content Page'],
    'PDF Link': unprocessed_data['PDF Link']

})


# STEP 13: SAVE DATA TO .CSV

# Save processed data to CSV
processed_data.to_csv(output_file_name, index=True)

print(f"Processed data has been saved as {output_file_name}")