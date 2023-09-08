
# https://www.kaggle.com/datasets/durgeshrao9993/twitter-analysis-dataset-2022
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import string
import seaborn as sns
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
stopwords.words('english')

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# Load the synthetic dataset
df = pd.read_csv("/Users/shraddha/Documents/UEL/MYDES/twitter.csv")

# df.hist(bins = 30, figsize = (13,5), color = '#016A70')
# plt.show()

# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

# Let's test the newly added function
tweets_df_clean = df['tweet'].apply(message_cleaning)

print(tweets_df_clean[5]) # show the cleaned up version

print(df['tweet'][5]) # show the original version

# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = np.uint8)
tweets_countvectorizer = vectorizer.fit_transform(df['tweet'])

vectorizer.get_feature_names_out()

tweets = pd.DataFrame(tweets_countvectorizer.toarray())

X = tweets

y = df["label"].map({0: 1, 1: -1})


# Naive Bayes is a classification technique based on bayes theoram
# The theoram works on conditional Probability and Mathematics

# PRIOR PROBABILITY --> Probability of assuming before, performing the experiment.
# LIKLIHOOD PROBABILITY --> Probability of assuming while performing the experiment.
# POSTERIOR PROBABILITY --> Probability by combining both prior and liklihood probability.

# Formula of Naive Bayes Theoram:
#    P(A|B) = (P(B|A) * p(A)) / P(B)

print(X.shape)
print(y.shape)


# Training the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

# Confusion Matrix --> Compares true value with predicted value 
# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))

def analyze_naive_bayes():
    input_text = input_entry.get()
    input_vector = vectorizer.transform([input_text])
    sentiment_label = NB_classifier.predict(input_vector)[0]
    print("Sentiment Score:", sentiment_label)  # Debugging statement

    display_sentiment(sentiment_label)


# Function to analyze sentiment using TextBlob
def analyze_textblob():
    input_text = input_entry.get()

    blob = TextBlob(input_text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        sentiment = 1
    elif sentiment_score < 0:
        sentiment = -1
    else:
        sentiment = 0
    
    display_sentiment(sentiment)

# Function to analyze sentiment and update the result
def display_sentiment(sentiment_label):


    if sentiment_label == 1:
        sentiment = "Positive"
        image_path = "/Users/shraddha/Documents/UEL/MYDES/positive_image.png"
    elif sentiment_label == -1:
        sentiment = "Negative"
        image_path = "/Users/shraddha/Documents/UEL/MYDES/negative_image.png"
    else:
        sentiment = "Neutral"
        image_path = "/Users/shraddha/Documents/UEL/MYDES/neutral_image.png"

    print("sentiment:", sentiment)  # Debugging statement

    # Update the result label
    result_label.config(text=f"Sentiment: {sentiment}")

    # Load and display the corresponding image
    image = Image.open(image_path)
    image = image.resize((150, 150), Image.ADAPTIVE)
    photo = ImageTk.PhotoImage(image=image)
    image_label.config(image=photo)
    image_label.photo = photo


# Create the main window
app = tk.Tk()
app.title("Sentiment Analysis App by Shraddha")

# Create and configure the input label and entry
input_label = ttk.Label(app, text="Enter Text:")
input_label.pack(pady=10)

# Create a styled text field
style = ttk.Style()
style.configure("TEntry", padding=10, font=("Helvetica", 12))
input_entry = ttk.Entry(app, width=50, style="TEntry")
input_entry.pack()

# Create analyze buttons for different algorithms
analyze_button_textblob = ttk.Button(app, text="Text Blob", command=analyze_textblob, style="TButton")
analyze_button_nb = ttk.Button(app, text="Naive Bayes", command=analyze_naive_bayes, style="TButton")

style.configure("TButton", font=("Helvetica", 12))
analyze_button_textblob.pack(pady=10)
analyze_button_nb.pack(pady=10)

# Create and configure the result label
result_label = ttk.Label(app, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

# Create and configure the image label
image_label = ttk.Label(app)
image_label.pack(pady=10)

# Start the main event loop
app.mainloop()
