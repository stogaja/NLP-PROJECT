# let's import the libraries we need
import streamlit as st
import io
import netrc
import pickle
import sys
import pandas as pd
import numpy as np
# let's import sentence_transformers
import sentence_transformers
# let's import cross encoders
from sentence_transformers.cross_encoder import CrossEncoder

from app import XpathFinder

# declaring our containers
header_container = st.container()
mod_container = st.container()

# Header
with header_container:
    st.title("Xpath Finder App")

# model container
with mod_container:
    prompt = st.text_input("Enter your description below ...")
    data = (pd.read_csv('SBERT_data.csv')).drop(['Unnamed: 0'], axis=1)

    data['prompt'] = prompt
    data.rename(columns={'target_text': 'sentence2',
                prompt: 'sentence1'}, inplace=True)
    data['sentence2'] = data['sentence2'].astype('str')
    data['sentence1'] = data['sentence1'].astype('str')

if prompt:
    XpathFinder = CrossEncoder("cross-encoder/stsb-roberta-base")
    sentence_pairs = []
    for sentence1, sentence2 in zip(data['sentence1'], data['sentence2']):
        sentence_pairs.append([sentence1, sentence2])

    simscore = XpathFinder.predict([prompt])

    data['SBERT CrossEncoder_Score'] = XpathFinder.predict(sentence_pairs)
    most_acc = data.head(5)
    # predictions
    st.write("Highest Similarity score: ", simscore)
    st.text("Is this one of these the Xpath you're looking for?")
    st.write(st.write(most_acc["input_text"]))
