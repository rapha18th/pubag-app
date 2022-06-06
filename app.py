import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

import psycopg2
import sqlalchemy
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import requests
from rich import print

#engine = create_engine('postgresql://rairo@pubag:pubag1036!@pubag.postgres.database.azure.com:5432/postgres')

conn = psycopg2.connect(
    host="pubag.postgres.database.azure.com",
    port="5432",
    database="postgres",
    user="rairo@pubag",
    sslmode="require",
    password="")
cursor = conn.cursor()
#conn = engine.connect()



def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level
    DistilBERT model and finds similar vectors using FAISS.
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.

    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.id == idx][column]) for idx in I[0]]


@st.cache
def read_data(data="agri_pub2.csv"):
    """Read the data from local."""
    return pd.read_csv(data)


@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="faiss_index.pickle"):
    """Load and deserialize the Faiss index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)


def detect_language(text, key, region, endpoint):
    # Use the Translator detect function
    path = '/detect'
    url = endpoint + path
    # Build the request
    params = {
        'api-version': '3.0'
    }
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json'
    }
    body = [{
        'text': text
    }]
    # Send the request and get response
    request = requests.post(url, params=params, headers=headers, json=body)
    response = request.json()
    # Get language
    language = response[0]["language"]
    # Return the language
    return language


def translate(text, source_language, target_language, key, region, endpoint):
    # Use the Translator translate function
    url = endpoint + '/translate'
    # Build the request
    params = {
        'api-version': '3.0',
        'from': source_language,
        'to': target_language
    }
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-type': 'application/json'
    }
    body = [{
        'text': text
    }]
    # Send the request and get response
    request = requests.post(url, params=params, headers=headers, json=body)
    response = request.json()
    # Get translation
    translation = response[0]["translations"][0]["text"]
    # Return the translation
    return translation



def main():

    try:
        # Get Configuration Settings
        load_dotenv()
        key = os.getenv('COG_SERVICE_KEY')
        region = os.getenv('COG_SERVICE_REGION')
        endpoint = 'https://api.cognitive.microsofttranslator.com'

        text = 'hello!'
        print('Detected language of "' + text + '":',
              detect_language(text, key, region, endpoint))
        target_lang = 'es'
        print(target_lang + ":", translate(text, 'en',
              target_lang, key, region, endpoint))
    except Exception as ex:
        print(ex)

    # Load data and models
    data = read_data()
    model = load_bert_model()
    faiss_index = load_faiss_index()

    st.title("Agro Science Search Powered by PubAG")

    # User search
    user_input = st.text_area(
        "Search", "")

    # Filters
    st.sidebar.markdown("**Filters**")
    filter_year = st.sidebar.slider(
        "Publication year", 1914, 2021, (1914, 2021), 1)
    num_results = st.sidebar.slider("Number of search results", 3, 10, 3)

    # Fetch results
    if user_input:
        # Get paper IDs

        sc = detect_language(user_input, key, region, endpoint)
        st.write(sc)
        user_input = translate(user_input, sc,'en', key, region, endpoint)
        D, I = vector_search([user_input], model, faiss_index, num_results)
        # Slice data on year
        frame = data[
            (data.publication_year >= filter_year[0])
            & (data.publication_year <= filter_year[1])
        ]
        #st.write(I)
        # Get individual results
        results = []
        for id_ in I.flatten().tolist():
            if id_ in set(frame.id):
                f = frame[(frame.id == id_)]
                a = (user_input, f.iloc[0].title)
                results.append(a)
                print(results)
            else:
                continue

            st.subheader(
                translate(f.iloc[0].title, 'en',
                          sc, key, region, endpoint))
            st.write(f.iloc[0].url)
            st.write(f.iloc[0].publication_year)

            st.markdown(translate(f.iloc[0].abstract, 'en',
                                  sc, key, region, endpoint))

        sql = '''CREATE TABLE IF NOT EXISTS search_queries(search_query VARCHAR, results VARCHAR)'''
        cursor.execute(sql)

        for i in results:
              cursor.execute("""INSERT into search_queries(search_query, results) VALUES (%s, %s)""", i)


        # commit changes
        conn.commit()


if __name__ == "__main__":
    main()
