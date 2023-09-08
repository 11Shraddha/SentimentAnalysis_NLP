import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from textblob import TextBlob
import numpy as np # linear algebra
import pandas as pd # data processing
pd.options.mode.chained_assignment = None


from wordcloud import WordCloud #Word visualization
import matplotlib.pyplot as plt #Plotting properties
import seaborn as sns #Plotting properties
from sklearn.feature_extraction.text import CountVectorizer #Data transformation
from sklearn.model_selection import train_test_split #Data testing
from sklearn.linear_model import LogisticRegression #Prediction Model
from sklearn.metrics import accuracy_score #Comparison between real and predicted
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder #Variable encoding and decoding for XGBoost
import re #Regular expressions
import nltk
from nltk import word_tokenize
nltk.download('stopwords')

#Validation dataset
val=pd.read_csv("/Users/shraddha/Documents/UEL/MYDES/twitter_validation.csv", header=None)
#Full dataset for Train-Test
train=pd.read_csv("/Users/shraddha/Documents/UEL/MYDES/twitter_training.csv", header=None)

train.columns=['id','information','type','text']
print(train.head())

val.columns=['id','information','type','text']
print(val.head())

train_data=train
val_data=val


#Text transformation
train_data["lower"]=train_data.text.str.lower() #lowercase
train_data["lower"]=[str(data) for data in train_data.lower] #converting all to string
train_data["lower"]=train_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex
val_data["lower"]=val_data.text.str.lower() #lowercase
val_data["lower"]=[str(data) for data in val_data.lower] #converting all to string
val_data["lower"]=val_data.lower.apply(lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) #regex


# word_cloud_text = ''.join(train_data[train_data["type"]=="Positive"].lower)
# #Creation of wordcloud
# wordcloud = WordCloud(
#     max_font_size=100,
#     max_words=100,
#     background_color="black",
#     scale=10,
#     width=800,
#     height=800
# ).generate(word_cloud_text)
# #Figure properties
# plt.figure(figsize=(10,10))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()


# word_cloud_text = ''.join(train_data[train_data["type"]=="Negative"].lower)
# #Creation of wordcloud
# wordcloud = WordCloud(
#     max_font_size=100,
#     max_words=100,
#     background_color="black",
#     scale=10,
#     width=800,
#     height=800
# ).generate(word_cloud_text)
# #Figure properties
# plt.figure(figsize=(10,10))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()


# word_cloud_text = ''.join(train_data[train_data["type"]=="Irrelevant"].lower)
# #Creation of wordcloud
# wordcloud = WordCloud(
#     max_font_size=100,
#     max_words=100,
#     background_color="black",
#     scale=10,
#     width=800,
#     height=800
# ).generate(word_cloud_text)
# #Figure properties
# plt.figure(figsize=(10,10))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()

# word_cloud_text = ''.join(train_data[train_data["type"]=="Neutral"].lower)
# #Creation of wordcloud
# wordcloud = WordCloud(
#     max_font_size=100,
#     max_words=100,
#     background_color="black",
#     scale=10,
#     width=800,
#     height=800
# ).generate(word_cloud_text)

#Choosing english stopwords
stopwords_nltk = nltk.corpus.stopwords
stop_words = stopwords_nltk.words('english')
stop_words[:5]

#Initial Bag of Words
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    stop_words=stop_words, #English Stopwords
    ngram_range=(1, 1) #analysis of one word
)


#Train - Test splitting
reviews_train, reviews_test = train_test_split(train_data, test_size=0.2, random_state=0)

#Creation of encoding related to train dataset
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
#Transformation of test dataset with train encoding
X_test_bow = bow_counts.transform(reviews_test.lower)

#Labels for train and test encoding
y_train_bow = reviews_train['type']
y_test_bow = reviews_test['type']

# Logistic regression
model1 = LogisticRegression(C=1, solver="liblinear",max_iter=1000)
model1.fit(X_train_bow, y_train_bow)
# Prediction
test_pred = model1.predict(X_test_bow)
print("Test Accuracy: ", accuracy_score(y_test_bow, test_pred) * 100)


#Validation data
X_val_bow = bow_counts.transform(val_data.lower)
y_val_bow = val_data['type']
Val_res = model1.predict(X_val_bow)
print("Validation Accuracy: ", accuracy_score(y_val_bow, Val_res) * 100)


# Function to analyze sentiment using Logistic Regression with CountVectorizer
def analyze_logistic_regression_count():
    input_text = input_entry.get()
    input_vector = bow_counts.transform([input_text])
    sentiment_label = model1.predict(input_vector)[0]
    display_sentiment(sentiment_label)


#n-gram of 4 words
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    ngram_range=(1,4)
)
#Data labeling
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)
X_val_bow = bow_counts.transform(val_data.lower)

# model2 = LogisticRegression(C=0.9, solver="liblinear",max_iter=1500)
model2 = LogisticRegression(C=1, solver="liblinear",max_iter=1500)

# Logistic regression
model2.fit(X_train_bow, y_train_bow)
# Prediction
test_pred_2 = model2.predict(X_test_bow)
print("Test Accuracy 2: ", accuracy_score(y_test_bow, test_pred_2) * 100)

y_val_bow = val_data['type']
Val_pred_2 = model2.predict(X_val_bow)
print("Validation Accuracy 2: ", accuracy_score(y_val_bow, Val_pred_2) * 100)


# Function to analyze sentiment using Logistic Regression with TfidfVectorizer
def analyze_logistic_regression_tfidf():
    input_text = input_entry.get()
    input_vector = bow_counts.transform([input_text])
    sentiment_label = model2.predict(input_vector)[0]
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

    print("sentiment_label:", sentiment_label)  # Debugging statement

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
analyze_button_count = ttk.Button(app, text="Logistic Regression n-gram", command=analyze_logistic_regression_count, style="TButton")
analyze_button_tfidf = ttk.Button(app, text="Logistic Regression n-gram of 4", command=analyze_logistic_regression_tfidf, style="TButton")
# analyze_button_nb = ttk.Button(app, text="Naive Bayes", command=analyze_naive_bayes, style="TButton")

style.configure("TButton", font=("Helvetica", 12))
analyze_button_textblob.pack(pady=10)
analyze_button_count.pack(pady=10)
analyze_button_tfidf.pack(pady=10)
# analyze_button_nb.pack(pady=10)

# Create and configure the result label
result_label = ttk.Label(app, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

# Create and configure the image label
image_label = ttk.Label(app)
image_label.pack(pady=10)

# Start the main event loop
app.mainloop()
