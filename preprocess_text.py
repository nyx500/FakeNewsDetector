import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng') 
nltk.download('stopwords')
# Create set out of English stopwords
stop_words = set(stopwords.words('english'))

regex_dict = {
    "hashtags": r'#\w+',
    "mentions": r'@[\w-]+',
    "numbers": r'\b\d+\b',
    "emails": r'^[\w\.-]+@([\w-]+\.)+[\w-]{2,4}$',
    "urls": r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)',
    "non-http-urls": r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
    "times": r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b',
    "dates": r'\b(\d{2})/(\d{2})/\d{4}\b',
    "punctuation": r'[^\w\s]',
}

# Apply each regex pattern
for regex in regex_dict.values():
    text = re.sub(regex, '', text)

# Remove any redundant whitespace
cleaned_text = re.sub(r'\s+', ' ', text).strip()

