# Problematic-Speech-Detection-using-RNN-
This project involves a real-time speech detection system that detects the hate words in the microphone voice input. The system uses RNN and NLP techniques like N-grams and WordNet to classify the input audio. It is designed to identify and analyze harmful or abusive speech in real time using live voice input. The system captures audio through a microphone, transcribes it using Google Speech Recognition, and then processes the text through a sentiment analysis engine (VADER) and a custom-trained neural network classifier. The classifier is built using a dataset of problematic and non-problematic words, enhanced with WordNet-based synonym expansion to improve generalization. Categories such as racism, misogyny, body shaming, religious insults, threats, terrorism, child labor, and abuse are specifically targeted. Positive or neutral content like "love" or "respect" is also recognized to neutralize false positives. The interface is built using Tkinter, providing a user-friendly GUI where users can start voice input, view transcribed text, and get instant feedback on whether the speech contains harmful content. The application uses tokenization, n-grams (for deeper text analysis), and a simple embedding-based neural network to make predictions. It serves as a foundational tool for real-time content moderation, educational purposes, or AI-driven voice assistance, with potential to expand into multilingual support, sentence-level classification, and cloud deployment.











