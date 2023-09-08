#  https://www.kaggle.com/datasets/pragyakatyayan/hindi-tweets-dataset-for-sarcasm-detection


import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth',None)   #this displays the dataframe in full width
import collections
from collections import Counter
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import emoji
from indicnlp.tokenize import indic_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer #Data transformation
from sklearn.model_selection import train_test_split #Data testing
from sklearn.linear_model import LogisticRegression #Prediction Model
from sklearn.metrics import accuracy_score, classification_report #Comparison between real and predicted

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt



df_sarcastic = pd.read_csv('/Users/shraddha/Documents/UEL/MYDES/Hindi_Tweets-SARCASTIC.csv')
df_non_sarcastic = pd.read_csv('/Users/shraddha/Documents/UEL/MYDES/Hindi_Tweets-NON-SARCASTIC.csv')
df_sarcastic['label'] = 'sarcastic'
df_non_sarcastic['label'] = 'non_sarcastic'
df = pd.concat([df_sarcastic, df_non_sarcastic], axis=0)
df = df.drop(['username','acctdesc','location','following','followers', 'totaltweets', 'usercreatedts', 'tweetcreatedts', 'retweetcount', 'hashtags'] ,axis=1)
df = df.reset_index()
df = df.drop('index',axis=1)


# create a function called count_length() which will count the number of words in the text.
def count_length():
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))

count_length()
# print(df.tail(10))


# Remove All Emojis from Hindi Text Analysis Data
import re
emoji_pattern = re.compile("["                 
        u"U0001F600-U0001F64F"  # emoticons
        u"U0001F300-U0001F5FF"  # symbols & pictographs
        u"U0001F680-U0001F6FF"  # transport & map symbols
        u"U0001F1E0-U0001F1FF"  # flags (iOS)
        u"U00002500-U00002BEF"  # chinese char
        u"U00002702-U000027B0"
        u"U00002702-U000027B0"
        u"U000024C2-U0001F251"
        u"U0001f926-U0001f937"
        u"U00010000-U0010ffff"
        u"u2640-u2642" 
        u"u2600-u2B55"
        u"u200d"
        u"u23cf"
        u"u23e9"
        u"u231a"
        u"ufe0f"  # dingbats
        u"u3030""]+", flags=re.UNICODE)


for i in range(len(df)):
    original_text = df.loc[i, 'text']
    text_without_emojis = emoji.demojize(original_text)
    df.loc[i, 'text'] = text_without_emojis

count_length()

# print(df.tail(10))

def processText(text):
    text = text.lower()
    text = re.sub('((www.[^s]+)|(https?://[^s]+))','',text)
    text = re.sub('@[^s]+','',text)
    text = re.sub('[s]+', ' ', text)
    text = re.sub(r'#([^s]+)', r'1', text)
    text = re.sub(r'[.!:?\-\'"/]', '', text)
    text = text.strip('\'"')
    return text

for i in range(len(df)):
    df.loc[i, 'text'] = processText(df['text'][i])

count_length()

# print(df.tail(10))


# Generating Tokens for Hindi Text Analysis

def tokenization(indic_string):
    tokens = []
    for t in indic_tokenize.trivial_tokenize(indic_string):
        tokens.append(t)
    return tokens
df['text'] = df['text'].apply(lambda x: tokenization(x))

count_length()
# print(df.tail(10))

# Remove ‘\n’ from each token
def remove_newlines(strings_list):
    return [s.replace("\n", "") for s in strings_list]

# Apply the function to the 'text' column
df['text'] = df['text'].apply(remove_newlines)

# Remove Stopwords and Punctuations
stopwords_hi = ['तुम','मेरी','मुझे','क्योंकि','हम','प्रति','अबकी','आगे','माननीय','शहर','बताएं','कौनसी','क्लिक','किसकी','बड़े','मैं','and','रही','आज','लें','आपके','मिलकर','सब','मेरे','जी','श्री','वैसा','आपका','अंदर', 'अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'आदि', 'आप', 'इत्यादि', 'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके', 'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे', 'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर','करता', 'करते', 'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि', 'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी', 'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन', 'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन', 'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो', 'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस', 'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 'दूसरे', 'दो', 'द्वारा', 'न', 'नहीं', 'ना', 'निहायत', 'नीचे', 'ने', 'पर', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही', 'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो', 'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें', 'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वर्ग', 'वह', 'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग', 'सकता', 'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो', 'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते', 'होना', 'होने', 'अपनि', 'जेसे', 'होति', 'सभि', 'तिंहों', 'इंहों', 'दवारा', 'इसि', 'किंहें', 'थि', 'उंहों', 'ओर', 'जिंहें', 'वहिं', 'अभि', 'बनि', 'हि', 'उंहिं', 'उंहें', 'हें', 'वगेरह', 'एसे', 'रवासा', 'कोन', 'निचे', 'काफि', 'उसि', 'पुरा', 'भितर', 'हे', 'बहि', 'वहां', 'कोइ', 'यहां', 'जिंहों', 'तिंहें', 'किसि', 'कइ', 'यहि', 'इंहिं', 'जिधर', 'इंहें', 'अदि', 'इतयादि', 'हुइ', 'कोनसा', 'इसकि', 'दुसरे', 'जहां', 'अप', 'किंहों', 'उनकि', 'भि', 'वरग', 'हुअ', 'जेसा', 'नहिं']
stopwords_en = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
punctuations = ['nn', 'n', '।', '/', '`', '+', '\\', '"', '?', '▁(', '$', '@', '[', '_', "'", '!', ',', ':', '^', '|', ']', '=', '%', '&', '.', ')', '(', '#', '*', '', ';', '-', '}', '|', '"']
to_be_removed = stopwords_hi + punctuations + stopwords_en

for i in range(len(df)):
    df.loc[i, 'text'] = ' '.join([ele for ele in df['text'][i] if ele not in to_be_removed])

count_length()
# print(df.tail(10))

# df.hist(column = 'word_count', by ='label',figsize=(12,4), bins = 5)
# plt.show()

# Remove most frequent unnecessary words from Hindi Text Analysis Data
corpus_list =[]
for i in range(len(df)):
    corpus_list +=df['text'][i]
counter=collections.Counter(corpus_list)

to_remove = ['नेहरू', 'लेते', 'कटाक्ष', 'जय', 'शी', 'अगर', 'मास्टर', 'वो', 'सिगरेट', 'बीवी', 'इश्क़', 'किताब', 'वश', 'पटाकर', 'पिलाकर']
for i in range(len(df)):
    df.loc[i, 'text'] = ''.join([ele for ele in df['text'][i] if ele not in to_remove])
count_length()
# print(df.tail(10))


# Remove least common words/tokens
least_common= [word for word, word_count in Counter(corpus_list).most_common()[:-50:-1]]
for i in range(len(df)):
    df.loc[i, 'text'] = ''.join([ele for ele in df['text'][i] if ele not in least_common])
count_length()

# print(df.tail(10))

# Plotting distribution of tweet-length per Label [After text cleaning and processing]
# df.hist(column = 'word_count', by ='label',figsize=(12,4), bins = 5)
# plt.show()


df_list = []
font = "/Users/shraddha/Downloads/Devanagari/gargi.ttf"

for text in df['text']:
    words = nltk.word_tokenize(text)
    df_list.extend(words)

print(df_list)
dictionary=Counter(df_list)
wordcloud = WordCloud(width = 1000, height = 700,
                background_color ='white',
                min_font_size = 10, font_path= font).generate_from_frequencies(dictionary)
print(wordcloud.max_words)
# plot the WordCloud image                      
# plt.figure(figsize = (18, 8), facecolor = None)
# plt.imshow(wordcloud,interpolation="bilinear")
# plt.axis("off")
# plt.tight_layout(pad = 0)
# plt.show()

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

#Validation dataset
# val=pd.read_csv("/Users/shraddha/Documents/UEL/MYDES/twitter_validation.csv", header=None)
# #Full dataset for Train-Test
# train=pd.read_csv("/Users/shraddha/Documents/UEL/MYDES/twitter_training.csv", header=None)


combined_df = pd.concat([df_sarcastic, df_non_sarcastic], ignore_index=True)

#Train - Test splitting
# reviews_train, reviews_test = train_test_split(combined_df, test_size=0.2, random_state=42)

# #Creation of encoding related to train dataset
# # X_train_bow = bow_counts.fit_transform(reviews_train)
# # #Transformation of test dataset with train encoding
# # X_test_bow = bow_counts.transform(reviews_test.lower)
# X_train_bow = bow_counts.fit_transform(reviews_train['text'])  # Assuming 'text' is the column containing text data
# X_test_bow = bow_counts.transform(reviews_test['text'])  # Transform the test data with the same encoding


# #Labels for train and test encoding
# y_train_bow = reviews_train['label']
# y_test_bow = reviews_test['label']


# # Logistic regression
# model1 = LogisticRegression(C=1, solver="liblinear",max_iter=1000)
# model1.fit(X_train_bow, y_train_bow)
# # Prediction
# test_pred = model1.predict(X_test_bow)
# print("Test Accuracy: ", accuracy_score(y_test_bow, test_pred) * 100)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = combined_df['text']
combined_df['label'] = combined_df['label'].map({'sarcastic': 1, 'non_sarcastic': 0})
y = combined_df['label']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data (you can use TF-IDF or other methods)
# For simplicity, let's use CountVectorizer here
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)

mse_threshold = 0.001  # Adjust as needed
r2_threshold = 0.9    # Adjust as needed
neutral_threshold = 0.2  # Adjust as needed


# Function to analyze sentiment using Logistic Regression with CountVectorizer
def analyze_hindi_tweets():
    input_text = input_entry.get()
    input_vector = vectorizer.transform([input_text])
    sentiment_score = model.predict(input_vector)[0] # Get the probability of being sarcastic
    # sentiment_label = model.predict(input_vector)[0]

    display_sentiment(sentiment_score)

# Function to analyze sentiment and update the result
def display_sentiment(sentiment_score):

    print("sentiment_label:", sentiment_score)  # Debugging statement

    if sentiment_score > mse_threshold and sentiment_score > r2_threshold:
        image_path = "/Users/shraddha/Documents/UEL/MYDES/negative_image.png"
        sentiment = "Sarcastic"
    elif sentiment_score < -mse_threshold and sentiment_score < -r2_threshold:
        sentiment = "Non Sarcastic"
        image_path = "/Users/shraddha/Documents/UEL/MYDES/positive_image.png"
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
analyze_button_textblob = ttk.Button(app, text="analyze hindi tweets", command=analyze_hindi_tweets, style="TButton")

style.configure("TButton", font=("Helvetica", 12))
analyze_button_textblob.pack(pady=10)

# Create and configure the result label
result_label = ttk.Label(app, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

# Create and configure the image label
image_label = ttk.Label(app)
image_label.pack(pady=10)

# Start the main event loop
app.mainloop()
