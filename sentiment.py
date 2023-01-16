import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load the dataset of movie reviews
df = pd.read_csv("movie_reviews.csv")

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Create a new column in the DataFrame to store the sentiment scores
df["sentiment"] = df["review"].apply(lambda x: sia.polarity_scores(x)["compound"])

# Print the average sentiment score for positive and negative reviews
print("Average sentiment score for positive reviews:", df[df["sentiment"] > 0]["sentiment"].mean())
print("Average sentiment score for negative reviews:", df[df["sentiment"] <= 0]["sentiment"].mean())
