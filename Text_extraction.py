import nltk
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# Download all NLTK resources to the same directory
nltk.download('punkt','wordnet','stopwords','averaged_perceptron_tagger')

# Text cleaning function
def cleaned_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

# Sentence splitting function
def sentence_split(text):
    sentences = sent_tokenize(text)
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return clean_sentences

# Keyword extraction function
def extract_keywords(file_df, top_n=20):
    text = file_df['text'][0]
    clean_txt = cleaned_text(text)
    sentences = sentence_split(clean_txt)

    if len(sentences) < 2:
        print("⚠️  Not enough sentences for analysis. Using the whole text.")
        sentences = [clean_txt]

    vectorizer = TfidfVectorizer(
        max_features=200,
        min_df=1,
        max_df=0.8,
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        mean_score = np.mean(tfidf_matrix.toarray(), axis=0)
        results_df = pd.DataFrame({
            'Keywords': feature_names,
            'tfidf_score': mean_score
        })
        results_df = results_df.sort_values('tfidf_score', ascending=False)
        return results_df.head(top_n)
    except Exception as e:
        print(f"Error in keyword extraction: {e}")
        return None

# Read your file into a DataFrame
file_df = pd.DataFrame({'text':[open('87.txt','r',encoding='utf-8').read()]})

# Extract keywords
keywords_df = extract_keywords(file_df, top_n=20)
if keywords_df is not None:
    keywords_df.to_csv('keywords_output.txt', sep='\t', index=False)



