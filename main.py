import os
import csv
import re
import string
import pandas as pd

# Directory containing your txt files
txt_dir = 'C:\\Users\\halil\\Desktop\\spam_dataset_test\\txt_files'

# Output CSV file
csv_file = 'C:\\Users\\halil\\Desktop\\spam_dataset_test\\csv_output\\output2.csv'

# Get a list of all txt files in the directory
txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    # Write the header
    writer.writerow(['text number', 'label', 'text', 'label_num'])

    # Loop over the txt files
    for i, txt_file in enumerate(txt_files, start=1):
        # Open the txt file and read its contents
        with open(os.path.join(txt_dir, txt_file), 'r') as tf:
            text = tf.read()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove special characters like '\r\n\t'
        text = re.sub('\s+', ' ', text)

        # Convert text to lowercase
        text = text.lower()

        # Determine the label and label_num
        label = 'ham'
        label_num = 0 if label == 'ham' else 1

        # Write the data to the CSV file
        writer.writerow([i, label, text, label_num])

        import pandas as pd

# Load your DataFrame
df = pd.read_csv('C:\\Users\\halil\\Desktop\\spam_dataset_test\\csv_output\\output2.csv')

# Shuffle your DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled DataFrame to a new CSV file
df.to_csv('C:\\Users\\halil\\Desktop\\spam_dataset_test\\csv_output\\shuffled_output.csv', index=False)