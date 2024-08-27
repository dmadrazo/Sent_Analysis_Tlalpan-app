
import streamlit as st
import pandas as pd
import tweepy
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Configuración de las credenciales de Twitter (reemplaza con tus propias credenciales)
consumer_key = "QUIzYUZNZVUtLUVqWmVXWkdWaUk6MTpjaQ"
consumer_secret = "w2fJFw5HX2NupuLcnCzyczcBcH3GkdnPe5NDdTh2hz5225O4cN"
access_token = "XTqpGDQ8VCL4SQfUL9UNxMghv"
access_token_secret = "6du1vSLiZ1RBGUNd7pMPh2YgEuaPGLqK0YgaGBWS3yAUE5RNa2"

# Autentificación en la API de Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def download_tweets(query):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="es").items(100)
    data = [[tweet.created_at, tweet.text] for tweet in tweets]
    df = pd.DataFrame(data, columns=['fecha', 'texto'])
    return df

def analyze_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    afinn = Afinn()
    df['polaridad_vader'] = df['texto'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['polaridad_afinn'] = df['texto'].apply(lambda x: afinn.score(x))
    return df

def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def main():
    st.title("Análisis de Sentimientos en Twitter")

    query = st.selectbox("Selecciona un término de búsqueda", ["gabyosorio", "Tlalpan"])

    if st.button("Buscar y Analizar"):
        df = download_tweets(query)
        df = analyze_sentiment(df)

        # Visualización de los resultados
        st.dataframe(df)

        # Gráfico de distribución de sentimientos
        plt.figure(figsize=(10, 5))
        plt.hist(df['polaridad_vader'], bins=20, alpha=0.5, label='VADER')
        plt.hist(df['polaridad_afinn'], bins=20, alpha=0.5, label='AFINN')
        plt.xlabel('Polaridad')
        plt.ylabel('Frecuencia')
        plt.legend(loc='upper right')
        st.pyplot()

        # Nube de palabras
        all_words = ' '.join([text for text in df['texto']])
        create_wordcloud(all_words)

if __name__ == "__main__":
    main()