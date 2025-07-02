import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import wordnet as wn
import random

# Ensure WordNet is downloaded
nltk.download('wordnet')

# Define categories and their corresponding WordNet synsets
categories = {
    "Racism": ["racism", "discrimination", "prejudice", "bigotry", "xenophobia", "racial_slur"],
    "Religious insults": ["religious_insult", "blasphemy", "sacrilege", "profanity", "heresy"],
    "Body shaming": ["body_shaming", "fat-shaming", "skinny-shaming", "appearance_insult"],
    "Threats and violence": ["threat", "violence", "intimidation", "harassment", "assault", "abuse"],
    "Misogyny": ["misogyny", "sexism", "sexual_harassment", "gender_discrimination", "objectification"],
    "Terrorism": ["terrorism", "terrorist", "extremism", "radicalization", "bombing", "attack"],
    "Hate": ["hate", "hateful", "hatred", "hostility", "animosity", "antagonism"],
    "Abuse": ["abuse", "mistreatment", "cruelty", "maltreatment", "violence"],
    "Child labor": ["child_labor", "child_exploitation", "child_slavery", "child_abuse"],
    "Molesting": ["molest", "molestation", "sexual_abuse", "assault", "harassment"]
}

# Define non-problematic words and their synonyms
non_problematic_words = ["love", "peace", "kindness", "tolerance", "respect", "calm","good","best","optimistic"]
non_problematic_synonyms = []
for word in non_problematic_words:
    synonyms = wn.synsets(word)
    non_problematic_synonyms.extend([lemma.name() for syn in synonyms for lemma in syn.lemmas()])

# Generate training dataset for problematic words
problematic_words = []
for category, words in categories.items():
    problematic_words.extend(words)

problematic_synonyms = []
for word in problematic_words:
    synonyms = wn.synsets(word)
    problematic_synonyms.extend([lemma.name() for syn in synonyms for lemma in syn.lemmas()])

# Combine non-problematic and problematic words into the training dataset
training_dataset = [(word, 1) for word in problematic_words + problematic_synonyms]
training_dataset.extend([(word, 0) for word in non_problematic_words + non_problematic_synonyms])

# Shuffle the dataset
random.shuffle(training_dataset)

# Split dataset into features (words) and labels
words, labels = zip(*training_dataset)

# Convert words to lowercase
words = [word.lower() for word in words]

# Tokenize the words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(words)
sequences = tokenizer.texts_to_sequences(words)

# Pad sequences to ensure uniform length
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert labels to numpy array
labels = np.array(labels)

# Define and train a simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=max_sequence_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=10)

transcribed_text = ""  # Global variable to store transcribed text


# Function to transcribe speech
def transcribe_speech():
    global transcribed_text  # Access global variable
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        print("Transcribing...")

        try:
            text = recognizer.recognize_google(audio)
            print("Transcription:", text)
            transcribed_text = text  # Store transcribed text
            input_text.delete(1.0, tk.END)
            input_text.insert(tk.END, text)  # Display transcribed text in input text area
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))


# Function to detect problematic words using the trained model
def detect_problematic_words(text):
    sequences = tokenizer.texts_to_sequences([text.lower()])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    prediction = model.predict(padded_sequences)[0]
    problematic_probability = prediction[0]  # Probability of being problematic
    return problematic_probability >= 0.5


# Function to perform sentiment analysis
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    return sentiment_score


# Function to extract n-grams from text
def extract_ngrams(text, n):
    tokens = text.split()
    ngrams_list = list(zip(*[tokens[i:] for i in range(n)]))
    return [' '.join(gram) for gram in ngrams_list]


def process_voice_command():
    global transcribed_text  # Access global variable
    if transcribed_text:
        text = transcribed_text
        problematic = False
        detected_words = []
        sentiment_score = analyze_sentiment(text)

        # Tokenize text into words
        words = re.findall(r'\b\w+\b', text.lower())

        # Check if any problematic words are detected
        for word in words:
            if detect_problematic_words(word):
                detected_words.append(word)
                problematic = True

        # Check if any non-problematic words or their synonyms are detected
        for word in words:
            if word in non_problematic_words or word in non_problematic_synonyms:
                problematic = False

        result_text.delete(1.0, tk.END)

        if problematic:
            result_text.insert(tk.END, "Detected problematic words in the speech:\n")
            result_text.insert(tk.END, ", ".join(detected_words) + "\n")
            result_text.insert(tk.END, "Speech is negative.\n")
        else:
            result_text.insert(tk.END, "No problematic words detected in the speech.\n")
            result_text.insert(tk.END, "Speech is positive.\n")


def clear_text():
    input_text.delete(1.0, tk.END)
    result_text.delete(1.0, tk.END)


# Create the main window
root = tk.Tk()
root.title("Problematic Speech Detection")

# Create text area for input
input_text = scrolledtext.ScrolledText(root, width=50, height=5)
input_text.pack(padx=10, pady=10)

# Create text area for displaying results
result_text = scrolledtext.ScrolledText(root, width=50, height=20)
result_text.pack(padx=10, pady=10)

# Button to start voice input
start_button = tk.Button(root, text="Start Voice Input", command=transcribe_speech)
start_button.pack(pady=5)

# Button to process voice command
process_button = tk.Button(root, text="Process Voice Command", command=process_voice_command)
process_button.pack(pady=5)

# Button to clear text
clear_button = tk.Button(root, text="Clear Text", command=clear_text)
clear_button.pack(pady=5)

root.mainloop()