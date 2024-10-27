"""Preprocess the text to improve the performance of the speech"""

import re
import inflect  # for number-to-word conversion
from unidecode import unidecode  # for converting special characters to ASCII

# Initialize inflect engine for number-to-words conversion
p = inflect.engine()

def replace_abbreviations(text):
    # Add common abbreviations and their expansions here
    abbreviations = {
        "mr.": "mister",
        "mrs.": "misses",
        "dr.": "doctor",
        "etc.": "et cetera",
        "i.e.": "that is",
        "e.g.": "for example",
        "u.s.": "united states",
        "node.js" : "Nodejs",
        "vue.js" : "viewjs",
        "Inc." : "",
        "inc" : "",
        "API" : "EI PI AI",
        "APIs" : "EI PI AIs",
        # Add more as needed
    }
    
    for abbr, expansion in abbreviations.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)

    return text

def convert_numbers(text):
    # Function to convert numbers to words
    def replace_number(match):
        return p.number_to_words(match.group(0))
    
    # Replace standalone numbers with words
    text = re.sub(r'\b\d+\b', replace_number, text)
    return text

def clean_text(text):
    # Lowercase text
    text = text.lower()
    
    # Remove unwanted characters (e.g., emojis or special characters)
    text = unidecode(text)  # Convert accented characters to ASCII equivalents
    
    # Replace newlines and extra whitespace with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove non-standard characters, keeping basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()'\"]", " ", text)

    # Replace . with "dot"
    text = re.sub(r'\.', ' dot ', text)

    # Replace abbreviations or common contractions
    text = replace_abbreviations(text)

    # Convert numbers to words
    text = convert_numbers(text)

    # Remove extra spaces and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text