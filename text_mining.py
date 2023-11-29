import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Sample text data
text_data = """
Natural language processing (NLP) is a subfield of artificial intelligence that deals with the interaction between computers and humans using natural language. It enables computers to understand, interpret, and generate human-like text.
"""

# Tokenization
tokens = word_tokenize(text_data.lower())  # Convert to lowercase for consistency
print("Tokens:", tokens)

# Removing stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
print("Filtered Tokens:", filtered_tokens)

# Frequency distribution
fdist = FreqDist(filtered_tokens)
print("Word Frequency Distribution:", fdist)

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
sentiment_score = sia.polarity_scores(text_data)
print("Sentiment Score:", sentiment_score)
