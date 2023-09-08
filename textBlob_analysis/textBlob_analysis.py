import tkinter as tk
from tkinter import ttk
from textblob import TextBlob
from PIL import Image, ImageTk

# Function to analyze sentiment and update the result
def analyze_sentiment():
    input_text = input_entry.get()
    blob = TextBlob(input_text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        sentiment = "Positive"
        image_path = "/Users/shraddha/Documents/UEL/MYDES/positive_image.png"
    elif sentiment_score < 0:
        sentiment = "Negative"
        image_path = "/Users/shraddha/Documents/UEL/MYDES/negative_image.png"
    else:
        sentiment = "Neutral"
        image_path = "/Users/shraddha/Documents/UEL/MYDES/neutral_image.png"

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

# Create a styled analyze button
analyze_button = ttk.Button(app, text="Analyze Sentiment", command=analyze_sentiment, style="TButton")
style.configure("TButton", font=("Helvetica", 12))
analyze_button.pack(pady=10)

# Create and configure the result label
result_label = ttk.Label(app, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

# Create and configure the image label
image_label = ttk.Label(app)
image_label.pack(pady=10)

# Start the main event loop
app.mainloop()
