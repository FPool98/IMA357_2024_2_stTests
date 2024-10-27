import numpy as np
import streamlit as st
import pandas as pd
import requests
from io import StringIO
import math
import copy
from collections import Counter, OrderedDict
from nltk.tokenize import TreebankWordTokenizer
import nltk


def contar_frecuencia(palabra, token) -> int:
            return token.split().count(palabra.lower())


def limpiar_tokenizar(doc, tokenizer= TreebankWordTokenizer()):
        stop_words = set(nltk.corpus.stopwords.words('spanish'))
        simbolos = {',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']', '%', '#', '$', '&', '/', '+', "'"}
        tokens = tokenizer.tokenize(doc)
        return [token for token in tokens if token not in simbolos and token not in stop_words and '//' not in token]


def sim_coseno(vec1: np.ndarray, vec2: np.ndarray) -> float:
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    norm_1 = math.sqrt(sum([x**2 for x in vec1]))
    norm_2 = math.sqrt(sum([x**2 for x in vec2]))
    return dot_prod / (norm_1 * norm_2)


# Vectorizador Bag of Words
def BoW_vec_Streamlit(docs: list) -> tuple:
    doc_tokens = [limpiar_tokenizar(doc) for doc in docs]  # Limpiar y obtener tokens
    all_doc_tokens = sum(doc_tokens, [])  # Juntar tokens totales
    lexico = sorted(set(all_doc_tokens))  # obtener léxico
    zero_vector = OrderedDict((token, 0) for token in lexico)  # obtener vector nulo
    document_bow_vectors = []  # Lista para documentos vectorizados
    for tokens in doc_tokens:  # Iterar sobre documentos tokenizados
        vec = zero_vector.copy()
        token_counts = Counter(tokens)
        vec.update(token_counts)
        document_bow_vectors.append(vec)
    return document_bow_vectors, zero_vector


def buscar_titular_similar(cadena_texto: str, dataframe: pd.DataFrame) -> tuple:
    # Vectorizar los cuerpos de los documentos
    docs = dataframe['Cuerpo'].tolist()
    doc_vectors, zero_vector = BoW_vec_Streamlit(docs)
    # Vectorizar input
    cadena_tokens = limpiar_tokenizar(cadena_texto)
    
    cadena_tokens = [token for token in cadena_tokens if token in zero_vector]
    if not cadena_tokens:
         st.write("La oración no tiene componentes en el léxico formado por los documentos.")
         return None,0

    cadena_vec = zero_vector.copy()
    token_counts = Counter(cadena_tokens)
    cadena_vec.update(token_counts)
    
    # Similitud Coseno
    similitudes = []
    for vec in doc_vectors:
        similitud = sim_coseno(cadena_vec, vec)
        similitudes.append(similitud)
    
    # Encontrar el índice del documento más similar
    idx_mas_similar = np.argmax(similitudes)
    
    # Retornar el titular del documento más similar
    return dataframe.iloc[idx_mas_similar]['Titular'], similitudes[idx_mas_similar]


def contar_frecuencia_tokens(oracion: str, texto: str) -> int:
    # Tokenizar la oración y el texto preprocesado
    tokenizer = TreebankWordTokenizer()
    tokens_oracion = tokenizer.tokenize(oracion.lower())
    tokens_texto = tokenizer.tokenize(texto)

    token_counts = Counter(tokens_texto)  # Contar frecuencias de tokens en texto
    # Calcular la frecuencia acumulada de tokens en oración
    frecuencia_total = sum(token_counts[token] for token in tokens_oracion)
    
    return frecuencia_total



@st.cache_data
def cargar_csv(DATA_URL):
    # Autorización (privilegios limitados)
    token = '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'
    headers = {'Authorization': f'token {token}'}
    response = requests.get(DATA_URL, headers=headers)
    if response.status_code == 200:
        csv_data = response.content.decode('utf-8')
        data = pd.read_csv(StringIO(csv_data))
        return data
    else:
         st.error(f"Error al cargar el archivo: {response.status_code}")
         return None


def main():
    # Titulo
    st.title("App Item 3")
    Data_url = 'https://raw.githubusercontent.com/FPool98/IMA357_2024_2_stTests/main/Grupo_Nro.csv'
    # Leer CSV
    datos_cargados = cargar_csv(Data_url)
    
    # Mostrar DataFrame
    st.write("Tabla de documentos:")
    st.dataframe(datos_cargados)

    # Input para buscador de palabras
    buscar_palabra = st.text_input("Introduzca una palabra:")
    
    # Proceso para palabras
    if buscar_palabra:
        # Aplicar la búsqueda en la columna "Cuerpo"
        datos_cargados['Frecuencia'] = datos_cargados['Cuerpo'].apply(lambda txt: contar_frecuencia(buscar_palabra,txt))
        
        # Filtrar documentos
        resultados = datos_cargados[datos_cargados['Frecuencia'] > 0]

        if not resultados.empty:
            # Mostrar titulares y frecuencia de la palabra por documento:
            st.write(f"Resultados para '{buscar_palabra}':")
            st.table(resultados[['Titular','Frecuencia']])
        else:
            st.write(f"No se encontraron coincidencias para '{buscar_palabra}'")
    
    # Input para oraciones
    input_oracion = st.text_input("Ingrese una oración:")

    # Proceso para oraciones
    if input_oracion:

        datos_cargados['FrecuenciaAcumulada'] = datos_cargados['Cuerpo'].apply(lambda texto: contar_frecuencia_tokens(input_oracion, texto))
        documento_mas_frecuente = datos_cargados.loc[datos_cargados['FrecuenciaAcumulada'].idxmax()]

        titular_similar, similitud = buscar_titular_similar(input_oracion,datos_cargados)
        st.write(f"El titular del cuerpo más similar por similitud coseno es: {titular_similar} (Similitud: {similitud:.4f})")
        st.write(f"- **Título:** {documento_mas_frecuente['Titular']}")
        st.write(f"Mayor coincidencia: {documento_mas_frecuente['FrecuenciaAcumulada']} palabras encontradas")


if __name__ == '__main__':
      main()