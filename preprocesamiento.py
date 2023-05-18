"""""#ROSALES ONOFRE TANIA
#6BV1
#Inteligencia Artificial
#17/24/2023

#Código para hacer la limpieza de los datos:stopwords,signos de puntuación,lematización y stemming
#En este caso los datos requeridos son un csv con contenido de noticias y etiquitado
#Funciones
#1.Normalizar->Ayuda al preprocesamiento del texto como quitar stop words,signos de puntuacíón,stemming
2.clen_html->Ayuda a la limpieza de las etiquetas de html
3.frequency_words->Ayuda a verificar la frecuencia de las palabras para poder proceder a verificar palabras no utiles en el modelado"""




import re
import spacy
from unidecode import unidecode
from collections import Counter
import matplotlib.pyplot as plt
import nltk
nlp = spacy.load("es_core_news_sm")
stemmer = nltk.stem.SnowballStemmer('spanish')





stop_words=['ano','mil','mujer','ciento']
# Agregar stop words personalizado al modelo de Spacy
nlp.Defaults.stop_words |= set(stop_words)
#Eliminar etiquetas de html
def clean_html(text):

    if text is None or not isinstance(text, str) or not text.strip():
        return ""

    # Eliminar etiqueta que está al principio
    cleaned_text = re.sub('^<p><em>', '', text)
    # Eliminar al final etiqueta
    cleaned_text = re.sub('</em>$', '', cleaned_text)
    # Eliminar todas las demás etiquetas HTML
    cleaned_text = re.sub('<[^>]*>', '', cleaned_text)
    cleaned_text = re.sub('\d', '', cleaned_text)
    cleaned_text = re.sub('</p>', '', cleaned_text)
    return cleaned_text



def normalizar(n_text):
    clean_text = clean_html(n_text)
    # Eliminar signos de puntuación
    clean_text = re.sub('[^\w\s]', '', clean_text)
    # Acentos
    clean_text = unidecode(clean_text).encode('utf-8').decode('ascii', 'ignore')

    # Convertir texto en  mínisculas
    clean_text = clean_text.lower()
    # Quitar Stop Words
    clean_text = nlp(clean_text)

    # Agregar stop words personalizado al modelo de Spacy
    custom_stop_words = set(stop_words)
    for word in custom_stop_words:
        nlp.vocab[word].is_stop = True

    # Lematizar y Stemming
    tokens = [stemmer.stem(token.lemma_) if token.lemma_ != '-PRON-' else stemmer.stem(token.text) for token in clean_text if not token.is_stop and not token.is_punct and not token.is_space]
    print(tokens)
    #Eliminamos las terminaciones el
    tokens = [token for token in tokens if not token.endswith(' el')]
    tokens = [token for token in tokens if not token.endswith('ano')]



    documento = ' '.join(tokens)

    return documento




#Verificar Frecuencia de palabras
def frequency_words(column):
    # Lista de tokens de la columna 'Documento_Normalizado'
    tokens = [token for doc in column for token in doc.split()]

    # Contar la frecuencia de cada palabra
    frecuencia_palabras = Counter(tokens)

    # Visualizar los resultados
    frecuencia_palabras_grafico = frecuencia_palabras.most_common(20)
    plt.bar(*zip(*frecuencia_palabras_grafico))
    plt.xticks(rotation=90)
    plt.show()

    return frecuencia_palabras




