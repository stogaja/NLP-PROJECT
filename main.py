# let's import the libraries
from sentence-transformers import util
from sentence-transformers import CrossEncoder
from sentence-transformers import SentenceTransformer
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

# let's load the english stsb dataset
stsb_dataset = load_dataset('stsb_multi_mt', 'en')
stsb_train = pd.DataFrame(stsb_dataset['train'])
stsb_test = pd.DataFrame(stsb_dataset['test'])

# let's create helper functions
nlp = en_core_web_sm.load()

#nlp = spacy.load("en_core_web_sm")


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

# loop through the data
XpathFinder = CrossEncoder("cross-encoder/stsb-roberta-base")
sentence_pairs = []
for sentence1, sentence2 in zip(data['sentence1'], data['sentence2']):
    sentence_pairs.append([sentence1, sentence2])

data['SBERT CrossEncoder_Score'] = XpathFinder.predict(
    sentence_pairs, show_progress_bar=True)

loaded_model = XpathFinder

# let's create containers
header_container = st.container()
mod_container = st.container()

# let's create the header
with header_container:
    st.title("SBERT CrossEncoder")
    st.markdown("This is a demo of the SBERT CrossEncoder model")

# let's create the model container
with mod_container:
    # let's get input from the user
    prompt = st.text_input("Enter a description below...")

    if prompt:
        simscore = loaded_model.predict([prompt])
        # sort the values
        data['SBERT CrossEncoder_Score'] = simscore
        most_acc = data.head(5)
        st.write(most_acc)
        st.write("The most accurate sentence is: ",
                 most_acc['sentence2'].iloc[0])
