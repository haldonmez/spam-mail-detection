# Spam mail detection using IMAP and Machine Learning

<p align="center">
  <img src="images\\images_readme\\tumblr_na9uijUN3v1ru5h8co1_500.gif">
</p>

## :rocket: Quick Start:
[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-24292e?logo=github)](https://github.com/haldonmez/spam-mail-detection)
[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20beff?logo=kaggle)](https://www.kaggle.com/code/haldonmez/prediction-of-spam-mails-using-logictic-regression)


## Content
- [1. Introduction](#sparkles-1-introduction)
- [2.Creating the Dataset ](#sparkles-2-creating-the-dataset)
- [3. Connecting to the IMAP](#sparkles-3-connecting-to-the-imap)
- [4. Importing user mails with IMAP](#sparkles-4-importing-user-mails-with-imap)
- [5. Setting up the Model](#sparkles-5-setting-up-the-model)
- [6. Connecting to Flask](#sparkles-6-connecting-to-flask)
- [7. Evaluation & Final Words](#sparkles-7-evaluation-and-final-words)
- [8. Project Structure](#sparkles-8-project-structure)


## :sparkles: 1. Introduction
### 1.1 What is the Enron Dataset?
The Enron mail dataset was collected and prepared by the CALO Project (A Cognitive Assistant that Learns and Organizes). It contains data from about 150 users, mostly senior management of Enron, organized into folders. The corpus contains a total of about 0.5M messages. This data was originally made public, and posted to the web, by the Federal Energy Regulatory Commission during its investigation. Further information on the dataset contents and conversion process can be found in the paper available at [Enron Email Dataset](http://www.cs.cmu.edu/~enron/).

### 1.2 A short summary
There are six different versions provided in this dataset. Which includes the raw version an the preprocessed versions of the mails from the corpus. The mails are divided between as ham and spam meaning spam or not-spam class division. We will examine the preprocessed further.

**Enron 1: 3,672 mails for ham. 1,500 mails for spam.**       
Enron 2: 4,361 mails for ham. 1,496 mails for spam.         
Enron 3: 4,012 mails for ham. 1,500 mails for spam.        
Enron 4: 1,500 mails for ham. 4,500 mails for spam.         
Enron 5: 1,500 mails for ham. 3,675 mails for spam.         
Enron 2: 1,500 mails for ham. 4,500 mails for spam.       
**Note: The dataset used has been bolded.**

This entire dataset is also free to access via the original corpus website [here](https://www2.aueb.gr/users/ion/data/enron-spam/). Original website includes raw and preprocessed versions of the mails as txt files one by one. You can also access the post processed csv version from my kaggle [here](https://www.kaggle.com/datasets/haldonmez/spam-or-ham-a-dataset-for-email-classification). My directory also include the combined version of the entire dataset for usage on a larger scale.

## :sparkles: 2. Creating the Dataset
The original dataset presented from the AUEB website includes the mails as separate txt files for both ham and spam mails. In my process I needed it to be plain csv to analyze and create a functioning model. Which in the following section describes how i took the txt files and iterated one by one creating plain csv file for simpler usage.

### 2.1 Iterating through the texts
```python
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
        writer.writerow([label, label_num,text])
```
I have repeated the process above for both ham and spam.
### 2.2 Writing to csv
```python
# Load the first CSV file
ham = pd.read_c('dataset_create\\dataset\\ham.csv')
# Load the second CSV file
spam = pd.read_c('dataset_create\\dataset\\spam.csv')
# Concatenate the two dataframes
df = pd.concat([ham, spam])
# Shuffle your DataFrame
df = df.sample(frac=1).reset_ind(drop=True)
# Save the shuffled DataFrame to new CSV file
df.to_c('dataset_create\\dataset\\enroncsv',index=False)
```
We then connect the ham and spam csv files we create and shuffle them for better model generalization.

## :sparkles: 3. Connecting to the IMAP
To be able to extract our own mails and then be able to test those in the models we need to connect to our account via IMAP (Internet Message Access Protocol) [What is IMAP?](https://en.wikipedia.org/wiki/Internet_Message_Access_Protocol).        
In this section, I will connect to my gmail account with IMAP.

- Gmail: 

  The new versions for the IMAP connections with gmail, Google requires special type of password called app passwords. That can only be used in special cases like IMAP connection the external sources.     
  [Get the app password for your account.](https://myaccount.google.com/)

  - After logging into your account in the google website. You can enter the Security tab. From there in order to get an app password your account must be 2-step verification activated which you can complete from this tab.

<p align="center">
  <img src="images\images_readme\Creating-App-Password-7.png">
</p>
<div align="center" style="font-weight: bold;">
  Google Account<br>
</div>
<br>

  - After completing the 2-step verification for your account you can choose the app password tab or you can navigate for search and search the app password.

<p align="center">
  <img src="images\\images_readme\\Yourapppassword.png">
</p>
<div align="center" style="font-weight: bold;">
  App Password<br>(This Image doesn't represent a real password.)
</div>
<br>

  - After clicking create an app password you can choose any name app and you will be given 16 digit code which you can implement as password in your codebase.

Such as;
```python
mail = example@gmail.com
password = "yourapppassword"

imap = imaplib.IMAP4_S("example@gmail.com")  # establish connection

imap.login(mail, password)  # login
```
- You can select any inbox and check the contents for your usage. 

Like;
```python
response, mailboxes = imap.list()

for mailbox in mailboxes:
    print(mailbox.decode())

status, messages = imap.select("INBOX")  # select inbox
```
> (\HasNoChildren) "/" "INBOX"
(\HasChildren \Noselect) "/" "[Gmail]"
(\HasNoChildren \Sent) "/" "[Gmail]/G&APY-nderilmi&AV8- Postalar"
(\HasNoChildren \Junk) "/" "[Gmail]/Spam"
(\Drafts \HasNoChildren) "/" "[Gmail]/Taslaklar"
(\All \HasNoChildren) "/" "[Gmail]/T&APw-m Postalar"
(\Flagged \HasNoChildren) "/" "[Gmail]/Y&ATE-ld&ATE-zl&ATE-"
(\HasNoChildren \Trash) "/" "[Gmail]/&AMcA9g-p kutusu"
(\HasNoChildren \Important) "/" "[Gmail]/&ANY-nemli"

## :sparkles: 4. Importing user mails with Imap

- From there on we use our inbox and simple python os to save our mails as txt files and iterate through them like we did before.
```python
for i in range(numOfMessages, 0, -1):
    res, msg = imap.fetch(str(i), "(RFC822)")  # fetches the email using it's ID     
        # Assuming 'msg' is your list of responses
    for response in msg:
        if isinstance(response, tuple):
            raw_email = response[1]
            email_message = email.message_from_bytes(raw_email)
    
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        charset = part.get_content_charset()
                        if charset is None:
                            charset = 'utf-8'
                        body = part.get_payload(decode=True)
                        body = body.decode(charset, errors='replace')
                        # Remove URLs
                        body = re.sub(r'http\S+|www.\S+', '', body, flags=re.MULTILINE)
            else:
                if email_message.get_content_type() == "text/plain":
                    charset = email_message.get_content_charset()
                    if charset is None:
                        charset = 'utf-8'
                    body = email_message.get_payload(decode=True)
                    body = body.decode(charset, errors='replace')
                    # Remove URLs
                    body = re.sub(r'http\S+|www.\S+', '', body, flags=re.MULTILINE)
    
    path = f"own_mails\\email{i}.txt"
    with open(path, 'w', encoding='utf-8') as f:
        # Write the content of the variable to the file
        f.write(body)
```
- We use a lot of if statements to check the condition of our own mails to be able to send back to the model.
  
## :sparkles: 5. Setting up the model
In my model I've used Logistic Regression to define the final accuracy and precision but anything can be used. 

A simple csv based still need some adjustments for our model to be able to tell if the content is spam or not-spam.

- First we need to midigate the email into lower letter simple text and we need to reduce the variety of the words with a stemmer so we can cramp as much as accuracy. 

- Then we will use vectorizer to turn every word into a binary representation. 

The final vectorized version of a single mail content should look like this so our model can use the binary classification to decide based on words.
<p align="center">
  <img src="images\images_readme\ss.png">
</p>
<div align="center" style="font-weight: bold;">
  Vectorized final dataset input example
</div>
<br>

### :sparkles: 5.1 Stemming and processing the dataset

```python
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
    
# Add 'corpus' as a new column in 'df'
df['transformed_text'] = corpus
```
- We can observe how to words cramped and created better usage in the picture below.

<p align="center">
  <img style=" width:100%; height:100%" src="images\images_readme\ss3.png">
</p>
<div align="center" style="font-weight: bold;">
  Before and After Porterstemmer
</div>
<br>
<br>
<div style="display: flex; justify-content: center;">
  <img src="images\images_readme\ham_wc.png" alt="Ham Wordcloud" style="margin-right: 10px;">
  <img src="images\images_readme\spam_wc.png" alt="Spam Wordcloud">
</div><br>
<div style="text-align: center; font-weight: bold;">
  Wordcloud for Ham and Spam
</div><br>

<div style="display: flex; justify-content: center;">
  <img src="images\images_readme\output.png" alt="Ham Wordcloud" style="margin-right: 10px;">
  <img src="images\images_readme\out2put.png" alt="Spam Wordcloud">
</div><br>
<div style="text-align: center; font-weight: bold;">
  Most common words for ham and spam
</div><br>

- We used visualizations for better understanding for spam and ham mails. Further more we create the vectorizer and the model.

### :sparkles: 5.2 Vectorizing and Creating the model
- First we create and fit the dataset to the vectorizer.
```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = df["label_num"]
```
- Then we split the dataset as test and train. We define the Logistic Regression and traing the model.
```python
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 42)

lrc = LogisticRegression(solver = 'liblinear', penalty = 'l1')

def train_classifier(X_train, y_train, X_test, y_test):
    lrc.fit(X_train,y_train)
    y_pred = lrc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy , precision
```
- Lastly we get the score prediction outputs from our model.
```python
accuracy_scores = []
precision_scores = []
current_accuracy, current_precision = train_classifier(X_train, y_train, X_test, y_test)

print()
print("For: ", "lr")
print("Accuracy: ", current_accuracy)
print("Precision: ", current_precision)

accuracy_scores.append(current_accuracy)
precision_scores.append(current_precision)
```
>- For: LR
>- Accuracy:  0.9603864734299516
>- Precision:  0.9290322580645162
- We save our model and the vectorizer to be able to connect back and predict our own mails. 
```python
# Save the model as a pickle file
joblib.dump(lrc, 'spam_model.pkl')
# Save the vectorizer as a pickle file
joblib.dump(vectorizer, 'vectorizer.pkl')
```
## :sparkles: 6. Connecting to the Flask
 - At this point our model prediction is complete. All we need to do is to transform the given input body with the same vectorizer and porterstemmer for better performance then we can predict in our Flask environment.

 First we connect to our Flask route;
 ```python
 app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    body = data.get('body')
    is_spam = predict_spam(body)
    return jsonify({'is_spam': is_spam})

if __name__ == "__main__":
    app.run(port=3000, debug=True)
 ```
- This snippet simply defines a Flask route in the localhost:3000 and we can send the input from the user to our model for prediction. Then we get the result and we can output the result via the frontend we connected.

```python
def predict_spam(body):
    # Load the model and the vectorizer from the file
    lrc = joblib.load('model_files\\spam_model.pkl')
    vectorizer = joblib.load('model_files\\vectorizer.pkl')

    nltk.download("stopwords")

    stemmer = PorterStemmer()
    corpus = []
    stopwords_set = set(stopwords.words("english"))
    stopwords_set.add("subject")
    stopwords_set.add("hou")
    stopwords_set.add("ect")
    stopwords_set.add("enron")

    text = body
    text = text.split()
    text = [word for word in text if word.isalpha() and len(word) > 1]
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = " ".join(text)
    corpus.append(text)

    final_pre = vectorizer.transform(corpus)
    y_pred = lrc.predict(final_pre)

    if y_pred.item() == 1:
        return True
    elif y_pred.item() == 0:
        return False
```

- We apply the same transformations to be able to input our model the same way and be able to get the output.

- So as a result in the body part we can input anything we want our app will still transform, vectorize and predict it, giving the result as an output label seen in the image.

<p align="center">
  <img src="images\images_readme\flask.png">
</p>
<div align="center" style="font-weight: bold;">
  Picture of the Final Website Design
</div>
<br>

## :sparkles: 7. Evaluation & Final Words
- We step by step created a model. Loaded our own mails with IMAP. Created the flask app. Applied the model to the flask route. Now we can use the website freely.

<table style="text-align: center;">
  <caption>Learning Outcomes of the LRC</caption>
  <tr>
    <td></td>
    <td align="center">Training</td>
    <td align="center">Test</td>
  </tr>
  <tr>
    <td align="center">Accuracy</td>
    <td align="center">96.03%</td>
    <td align="center">92.90%</td>
  </tr>
  <tr>
    <td align="center">Precision</td>
    <td align="center">95.26%</td>
    <td align="center">93.27%</td>
  </tr>
</table>
</div>

#:sparkles: 8. Project structure
```
├── dataset_create          
|   ├── dataset      
|   |   ├── enron1.csv      
|   |   ├── enron2.csv       
|   |   ├── enron3.csv    
|   |   ├── enron4.csv      
|   |   └── enron5.csv 
|   |   └── enron6.csv 
|   |   └── combined_enron.csv 
|   ├── raw_txt
|   ├── transform.py
├── own_mails
|   ├── email.txt  
├── model_files      
|   ├── spam_model.pkl      
|   └── vectorizer.pkl       
├── images      
│   └── ham_wc.png
│   └── spam_wc.png
├── static      
│   ├── scripts
│   ├──   └── main.js  
│   ├── styles
│   ├──   └── styles.css 
├── templates      
│   └── index.html
├── LICENSE
├── app.py
├── config.py
├── model.ipynb
├── imap.ipynb
├── README.md
└── requirements.txt
```
