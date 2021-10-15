import os
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
import pickle
import spacy
import string
import seaborn as sns; sns.set()
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

from nltk import ngrams
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string


os.chdir(os.path.dirname(os.path.realpath(__file__)))


DATA_PATH = 'serialized'


st.set_page_config(page_title="Words in judgemente", page_icon="🔤", layout="centered")


# My custom functions
@st.cache(allow_output_mutation=True)
def get_data(path):
    """
    Loads the serialized objects
    
    Parameters
    ----------
    path : str
        The folder where the serialized objects are stored
    
    Returns
    -------
    A tuple with the data pd.DataFrame, the TfidfVectorizer
    and the SVC fitted model.log_fit_n 
    """
    
    narco_data = pickle.load(open(Path(path, "narco_data.pkl"), "rb"))
    narco_schedule = pickle.load(open(Path(path, "narco_schedule.pkl"), "rb"))
    models_1 = pickle.load(open(Path(path, "models_1.pkl"), "rb"))
    models_2 = pickle.load(open(Path(path, "models_2.pkl"), "rb"))
    
    return narco_data, narco_schedule,  models_1, models_2


def most_similar_terms(model_, word, n, value=False):
    lista = []
    for genre, model in model_.items():
        for x in model.wv.most_similar(positive=word)[:n]:
            if value:
                lista.append([x[0],round(x[1], 2)])
            else:
                lista.append(x[0])
    return lista


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_spacy_model():
    return spacy.load("en_core_web_sm")
    
# Pages
def load_homepage(narco_data, narco_schedule,  models_1, models_2):
    st.markdown('''
    WRITE HERE
    ''', unsafe_allow_html = True)
    
   
def load_eda(narco_data, narco_schedule,  models_1, models_2):
    # IMAGES Data Analysis
    #adj = Image.open('images_d/wc_a_hate.png')
    
    narco_schedule_1 = narco_schedule.loc[(narco_schedule.schedule == "narco_1") ] 
    narco_schedule_2 = narco_schedule.loc[(narco_schedule.schedule == "narco_2") ] 
    
    st.write("On this page you can use the interactive tools to explore data and gather intormation about them.") 
    
    st.markdown("<h3><strong>Data Table Visualization</strong></h3>", unsafe_allow_html=True)

    st.write("WRITE HERE.")
    st.write("WRITE HERE.")
    
    st.markdown("---", unsafe_allow_html=True)
    word1 = st.text_input('Write one word here:', "word 1")
    #word2 = st.text_input('Write one word here:', "word 2")
    
    word1 = st.empty()
    #word2 = st.empty()
    if not word1 : #and not word2
        word1 = st.warning('Please write a word')
        #word2 = st.warning('Please write a word')

    if word1 : #and word2
        if word1 is not None : #and word2 is not None
            word1 = st.empty()
            #word2 = st.empty()
        try:
            similar_words  = most_similar_terms(models_1, word1, 10, value=False)
        except:
            print('No')
        
        #similar_list_value = most_similar_terms(
         #   model_, word, n, value=True)

        #prediction_under, color_under = get_text_color(pred_under)
        #prediction_pos, color_pos = get_text_color(pred_pos)

        st.markdown(
            "List of similar words:", similar_words, unsafe_allow_html=True)
        #st.markdown(similar_words)

        #st.markdown(
            #"<h3><strong>Model 2: Logistic Regression including only some Parts of Speech</h3></strong>", unsafe_allow_html=True)
        #st.markdown(
            #f'The sentence has been classified as: <span style="color:{color_pos}">**{prediction_pos}**</span>', unsafe_allow_html=True)
   

         

def load_classif(narco_data, narco_schedule,  models_1, models_2):
    st.markdown('''
    On this page you can test two of the models that have been trained for the project.
    
    In both cases, the lables were balanced with <em>RandomUnderSampler</em>.
    
    1. <ins><em>Support Vector Machine</em></ins> trained on Lemmatized text;
    2. <ins><em>Logistic Regression</em></ins> over data including only some parts of speech.
    
    In particular, the second model takes into account only the following list of parts of speech: composed of nouns, proper nouns, verbs, adjectives and pronouns.
    
    To do so, you need to write down the sentence you want to test in the board below.
    
    As you cas see, there is a sentence displayed by default. It was choosen since it clearly shows that the two models work differently: in this case, the first method performs better in terms of classification between <strong>Hate Speech</strong> and <strong>Not Hate Speech</strong>.
    ''', unsafe_allow_html=True)  
    
    written_sent = st.text_input('Write your sentence here:', "I hate all of you!")

    

def main():
    """Main routine of the app"""
    
    st.markdown("<h1>WORDS IN JUDGEMENT</h1>", unsafe_allow_html=True)
    st.markdown("<h4><strong>Knowledge extraction and information retrieval project</strong></h4>", unsafe_allow_html=True)
    st.write("Martina Viggiano (954603)")
    st.markdown("---", unsafe_allow_html=True)
    
    app_mode = st.sidebar.radio(
        "Go to:", ["Homepage", "Cosine Similarity", "PMI"]
    )
    
    narco_data, narco_schedule, models_1, models_2 = get_data(DATA_PATH)
    nlp = load_spacy_model()
    
    if app_mode == "Homepage":
        load_homepage(narco_data, narco_schedule,  models_1, models_2)
    elif app_mode == "Cosine Similarity":
        load_eda(narco_data, narco_schedule,  models_1, models_2)
    elif app_mode == "PMI":
        load_classif(narco_data, narco_schedule,  models_1, models_2)

if __name__ == "__main__":
    main()
