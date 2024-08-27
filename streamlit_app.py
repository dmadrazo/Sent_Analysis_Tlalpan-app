import streamlit as st
import tweepy
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
'''
# Configure Twitter API v2
def configure_twitter_api_v2():
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAIXPvQEAAAAAfMOHoqaI8wTNfwx%2BtszKOYfbnGc%3D0uXNsHncdDLFENCGXBpjDja8NGNrf4AfMpCMMWI8U8vAYrKUpn'  # Replace with your Twitter API v2 Bearer Token
    client = tweepy.Client(bearer_token=bearer_token)
    return client

# Function to clean the tweets
def clean_tweet(tweet):
    return ' '.join([word for word in tweet.split() if not word.startswith('http') and not word.startswith('@')])

# Function to get the sentiment
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def main():
    st.title("Twitter Sentiment Analysis")

    # Sidebar for input
    st.sidebar.title("Search Tweets")
    query = st.sidebar.text_input("Enter a keyword or hashtag:")
    num_tweets = st.sidebar.slider("Number of tweets to analyze", 10, 100, 50)
    analyze_button = st.sidebar.button("Analyze")

    if analyze_button and query:
        client = configure_twitter_api_v2()
        response = client.search_recent_tweets(query=query, max_results=num_tweets, tweet_fields=["created_at", "lang"])
        
        tweets = response.data
        if not tweets:
            st.error("No tweets found!")
            return

        # Create a DataFrame
        df = pd.DataFrame([tweet.text for tweet in tweets], columns=["Tweet"])
        df["Cleaned Tweet"] = df["Tweet"].apply(clean_tweet)
        df["Sentiment"] = df["Cleaned Tweet"].apply(get_sentiment)

        # Display the DataFrame
        st.write(df)

        # Sentiment Analysis
        st.subheader("Sentiment Analysis")
        sentiment_count = df["Sentiment"].value_counts()
        st.bar_chart(sentiment_count)

        # Word Cloud
        st.subheader("Word Cloud")
        all_words = ' '.join([text for text in df["Cleaned Tweet"]])
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(all_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)

if __name__ == "__main__":
    main()

'''

# Function to perform AFINN-like sentiment analysis
def afinn_sentiment_analysis(text):
    afinn = {
        'bueno': 3, 'feliz': 3, 'genial': 4, 'positivo': 2, 'agradable': 2,
        'malo': -3, 'triste': -2, 'horrible': -4, 'negativo': -2, 'terrible': -3
        # Add more words and their sentiment scores as needed
    }
    score = sum(afinn.get(word.lower(), 0) for word in text.split())
    return score

# Function to perform BING-like sentiment analysis
def bing_sentiment_analysis(text):
    bing_positive = {'bueno', 'feliz', 'genial', 'positivo', 'agradable'}
    bing_negative = {'malo', 'triste', 'horrible', 'negativo', 'terrible'}
    
    words = text.split()
    positive_score = sum(1 for word in words if word.lower() in bing_positive)
    negative_score = sum(1 for word in words if word.lower() in bing_negative)
    
    sentiment = 'Neutral'
    if positive_score > negative_score:
        sentiment = 'Positive'
    elif negative_score > positive_score:
        sentiment = 'Negative'
    
    return sentiment

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Main function to run the Streamlit app
def main():
    st.title("Spanish Sentiment Analysis with Word Cloud")

    # Text input
    st.subheader("Input Text")
    text_input = st.text_area("Enter text in Spanish:")

    if st.button("Analyze"):
        if not text_input:
            st.warning("Please enter some text for analysis.")
        else:
            # AFINN sentiment analysis
            afinn_score = afinn_sentiment_analysis(text_input)
            st.subheader("AFINN Sentiment Analysis")
            st.write(f"AFINN Sentiment Score: {afinn_score}")

            # BING sentiment analysis
            bing_sentiment = bing_sentiment_analysis(text_input)
            st.subheader("BING Sentiment Analysis")
            st.write(f"BING Sentiment: {bing_sentiment}")

            # Word Cloud
            st.subheader("Word Cloud")
            generate_wordcloud(text_input)

if __name__ == "__main__":
    main()