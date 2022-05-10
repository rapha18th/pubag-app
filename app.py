import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np


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


def main():
    # Load data and models
    data = read_data()
    model = load_bert_model()
    faiss_index = load_faiss_index()

    st.title("Vector-based searches with Sentence Transformers and Faiss")

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
        D, I = vector_search([user_input], model, faiss_index, num_results)
        # Slice data on year
        frame = data[
            (data.publication_year >= filter_year[0])
            & (data.publication_year <= filter_year[1])
        ]
        # Get individual results
        for id_ in I.flatten().tolist():
            if id_ in set(frame.id):
                f = frame[(frame.id == id_)]
            else:
                continue

            st.subheader(
               f.iloc[0].title)
            st.write(f.iloc[0].url)
            st.write(f.iloc[0].publication_year)

            st.subheader(f.iloc[0].abstract)




if __name__ == "__main__":
    main()
