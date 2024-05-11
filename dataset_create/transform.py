import os
import csv
import re
import string
import pandas as pd

# Directory containing your txt files
txt_dir_ham = 'dataset_create\\raw_txt\\enron6\\ham'

# Directory containing your txt files
txt_dir_spam = 'dataset_create\\raw_txt\\enron6\\spam'

# Output CSV file
csv_file_ham = 'dataset_create\\dataset\\ham.csv'

# Output CSV file
csv_file_spam = 'dataset_create\\dataset\\spam.csv'



# Get a list of all txt files in the directory
txt_files = [f for f in os.listdir(txt_dir_ham) if f.endswith('.txt')]

# Open the CSV file in write mode
with open(csv_file_ham, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Write the header
    writer.writerow(['label', 'label_num', 'text'])

    # Loop over the txt files
    for i, txt_file in enumerate(txt_files, start=1):
        # Open the txt file and read its contents
        with open(os.path.join(txt_dir_ham, txt_file), 'r', encoding='utf-8', errors='replace') as tf:
            text = tf.read()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove special characters like '\r\n\t'
        text = re.sub(r'\s+', ' ', text)

        # Convert text to lowercase
        text = text.lower()

        # Determine the label and label_num
        label = 'ham'
        label_num = 0 if label == 'ham' else 1

        # Write the data to the CSV file
        writer.writerow([label, label_num, text])




# Get a list of all txt files in the directory
txt_files = [f for f in os.listdir(txt_dir_spam) if f.endswith('.txt')]

# Open the CSV file in write mode
with open(csv_file_spam, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Write the header
    writer.writerow(['label', 'label_num', 'text'])

    # Loop over the txt files
    for i, txt_file in enumerate(txt_files, start=1):
        # Open the txt file and read its contents
        with open(os.path.join(txt_dir_spam, txt_file), 'r', encoding='utf-8', errors='replace') as tf:
            text = tf.read()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove special characters like '\r\n\t'
        text = re.sub(r'\s+', ' ', text)

        # Convert text to lowercase
        text = text.lower()

        # Determine the label and label_num
        label = 'spam'
        label_num = 1 if label == 'spam' else 0

        # Write the data to the CSV file
        writer.writerow([label, label_num, text])


# Load the first CSV file
ham = pd.read_csv('dataset_create\\dataset\\ham.csv')

# Load the second CSV file
spam = pd.read_csv('dataset_create\\dataset\\spam.csv')

# Concatenate the two dataframes
df = pd.concat([ham, spam])

# Shuffle your DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled DataFrame to a new CSV file
df.to_csv('dataset_create\\dataset\\enron6.csv', index=False)

os.remove('dataset_create\\dataset\\ham.csv') 
os.remove('dataset_create\\dataset\\spam.csv')