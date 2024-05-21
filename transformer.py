import os
import csv
import re
import string
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


txt_directory = "own_mails"

csv_save_directory = "own_mails\\prediction.csv"

def transformer():
    # Get a list of all txt files in the directory
    txt_files = [f for f in os.listdir(txt_directory) if f.endswith('.txt')]

    # Open the CSV file in write mode
    with open(csv_save_directory, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write the header
        writer.writerow(['text'])

        # Loop over the txt files
        for i, txt_file in enumerate(txt_files, start=1):
            # Open the txt file and read its contents
            with open(os.path.join(txt_directory, txt_file), 'r', encoding='utf-8', errors='replace') as tf:
                text = tf.read()

            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))

            # Remove special characters like '\r\n\t'
            text = re.sub(r'\s+', ' ', text)

            # Convert text to lowercase
            text = text.lower()

            # Write the data to the CSV file
            writer.writerow([text])

        df = pd.read_csv("own_mails\\prediction.csv")

        nltk.download("stopwords")

        stemmer = PorterStemmer()

        corpus = []

        stopwords_set = set(stopwords.words("english"))
        stopwords_set.add("subject")
        stopwords_set.add("hou")
        stopwords_set.add("ect")
        stopwords_set.add("enron")

        for i in range(len(df)):
            text = df.iloc[i]['text']
            text = text.split() # Add this line to split the text into words
            text = [word for word in text if word.isalpha() and len(word) > 1]
            text = [stemmer.stem(word) for word in text if word not in stopwords_set]
            text = " ".join(text)
            corpus.append(text)

        return corpus
