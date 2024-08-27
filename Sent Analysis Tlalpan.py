import streamlit as st
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Descargamos los recursos de NLTK
nltk.download('stopwords')
nltk.download('vader_lexicon')

def analyze_text(text):
    """
    Realiza un análisis de sentimiento utilizando AFINN y VADER, y genera una nube de palabras.

    AFINN: Es un léxico de palabras con puntuaciones numéricas que reflejan su connotación positiva o negativa.
    Los puntajes AFINN suelen variar entre -5 (muy negativo) y 5 (muy positivo).

    VADER: Es un analizador de sentimientos específico para el lenguaje de las redes sociales. 
    Los puntajes VADER van de -1 (muy negativo) a 1 (muy positivo). El puntaje 'compound' que se utiliza aquí representa el sentimiento general del texto.

    Args:
        text: El texto a analizar.

    Returns:
        Una tupla con los resultados del análisis de sentimiento: (sentimiento_afinn, sentimiento_vader)
    """

    # Tokenización y eliminación de stopwords
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))  # Ajusta el idioma según sea necesario
    words = [word for word in words if word not in stop_words]

    # Análisis de sentimiento AFINN
    afinn = Afinn()
    sentiment_afinn = afinn.score(text)

    # Análisis de sentimiento VADER
    sia = SentimentIntensityAnalyzer()
    sentiment_vader = sia.polarity_scores(text)['compound']

    # Generar nube de palabras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))

    # Mostrar y guardar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig("wordcloud.png")  # Guarda la imagen
    st.image("wordcloud.png")  # Muestra la imagen en Streamlit

    # Graficar sentimientos
    plt.figure(figsize=(10, 5))
    plt.bar(['AFINN', 'VADER'], [sentiment_afinn, sentiment_vader])
    plt.ylabel('Sentimiento')
    plt.title('Comparación de Sentimientos (AFINN y VADER)')
    plt.ylim(-1.2, 1.2)  # Ajusta el límite del eje y para visualizar mejor los resultados
    plt.savefig("sentimientos.png")
    st.image("sentimientos.png")

    return sentiment_afinn, sentiment_vader

# Interfaz de usuario
st.title("Analizador de Sentimiento con AFINN y VADER")
st.write("AFINN y VADER son herramientas para analizar el sentimiento de un texto. AFINN asigna puntajes a palabras basadas en un léxico, mientras que VADER está diseñado para entender el lenguaje natural de las redes sociales.")
st.write("**Interpretación de los puntajes:**")
st.write("- **AFINN:** Varía entre -5 (muy negativo) y 5 (muy positivo).")
st.write("- **VADER:** Varía entre -1 (muy negativo) y 1 (muy positivo).")
text_input = st.text_area("Ingrese su texto aquí:")

if st.button("Analizar"):
    sentiment_afinn, sentiment_vader = analyze_text(text_input)
    st.write("Sentimiento AFINN:", sentiment_afinn)
    st.write("Sentimiento VADER:", sentiment_vader)