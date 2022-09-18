# let's import the libraries
#from sentence_transformers import util
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
import sentence_transformers
import time
import sys
import os
import torch
import en_core_web_sm
from email import header
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import io
import netrc
from tqdm import tqdm
tqdm.pandas()

# Load the English STSB dataset
stsb_dataset = load_dataset('stsb_multi_mt', 'en')
stsb_train = pd.DataFrame(stsb_dataset['train'])
stsb_test = pd.DataFrame(stsb_dataset['test'])

# let's create helper functions
nlp = spacy.load("en_core_web_sm")


def text_processing(sentence):
    sentence = [token.lemma_.lower()
                for token in nlp(sentence)
                if token.is_alpha and not token.is_stop]
    return sentence


def cos_sim(sentence1_emb, sentence2_emb):
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)


# let's read the csv file
data = (pd.read_csv("SBERT_data.csv")).drop(['Unnamed: 0'], axis=1)

prompt = "charles"
data['prompt'] = prompt
data.rename(columns={'target_text': 'sentence2',
            'prompt': 'sentence1'}, inplace=True)
data['sentence2'] = data['sentence2'].astype('str')
data['sentence1'] = data['sentence1'].astype('str')

XpathFinder = CrossEncoder("cross-encoder/stsb-roberta-base")
sentence_pairs = []
for sentence1, sentence2 in zip(data['sentence1'], data['sentence2']):
    sentence_pairs.append([sentence1, sentence2])

data['SBERT CrossEncoder_Score'] = XpathFinder.predict(
    sentence_pairs, show_progress_bar=True)

# sorting the values
data.sort_values(by=['SBERT CrossEncoder_Score'], ascending=False)

loaded_model = XpathFinder

# Containers
header_container = st.container()
mod_container = st.container()

# Header
with header_container:

    # different levels of text you can include in your app
    st.title("Xpath Finder App")


# model container
with mod_container:

    # collecting input from user
    prompt = st.text_input("Enter your description below ...")

    # Loading e data
    data = (pd.read_csv("SBERT_data.csv")
            ).drop(['Unnamed: 0'], axis=1)

    data['prompt'] = prompt
    data.rename(columns={'target_text': 'sentence2',
                'prompt': 'sentence1'}, inplace=True)
    data['sentence2'] = data['sentence2'].astype('str')
    data['sentence1'] = data['sentence1'].astype('str')

    # let's pass the input to the loaded_model with torch compiled with cuda
    if prompt:
        # let's get the result
        simscore = loaded_model.predict([prompt])

        from sentence_transformers import CrossEncoder
        loaded_model = CrossEncoder("cross-encoder/stsb-roberta-base")
        sentence_pairs = []
        for sentence1, sentence2 in zip(data['sentence1'], data['sentence2']):
            sentence_pairs.append([sentence1, sentence2])

        # sorting the df to get highest scoring xpath_container
        data['SBERT CrossEncoder_Score'] = loaded_model.predict(sentence_pairs)
        most_acc = data.head(5)
        # predictions
        st.write("Highest Similarity score: ", simscore)
        st.text("Is this one of these the Xpath you're looking for?")
        st.write(st.write(most_acc["input_text"]))
