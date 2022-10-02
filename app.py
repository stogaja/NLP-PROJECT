import io
import netrc
import pickle
import sys
import pandas as pd
import numpy as np
import streamlit as st
# from sentence_transformers import SentenceTransformer
# import sentence_transformers
#import torch
#######################################

st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# # let's load the saved model
# loaded_model = pickle.load(open('XpathFinder1.sav', 'rb'))
# loaded_model = pickle.load('XpathFinder1.sav', map_location='cpu')


# class CPU_Unpickler(pickle.Unpickler):
#    def find_class(self, module, name):
#        if module == 'torch.storage' and name == '_load_from_bytes':
#            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#        else:
#            return super().find_class(module, name)
#

#loaded_model = CPU_Unpickler(open('XpathFinder1.sav', 'rb')).load()


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
    data = (pd.read_csv("SBERT_data.csv")).drop(['Unnamed: 0'], axis=1)

    data['prompt'] = prompt
    data.rename(columns={'target_text': 'sentence2',
                'prompt': 'sentence1'}, inplace=True)
    data['sentence2'] = data['sentence2'].astype('str')
    data['sentence1'] = data['sentence1'].astype('str')

    # let's pass the input to the loaded_model with torch compiled with cuda
    if prompt:
        # let's get the result
        from sentence_transformers import CrossEncoder
        XpathFinder = CrossEncoder("cross-encoder/stsb-roberta-base")
        sentence_pairs = []
        for sentence1, sentence2 in zip(data['sentence1'], data['sentence2']):
            sentence_pairs.append([sentence1, sentence2])
        simscore = XpathFinder.predict([prompt])

       # sorting the df to get highest scoring xpath_container
        data['SBERT CrossEncoder_Score'] = XpathFinder.predict(sentence_pairs)
        most_acc = data.head(5)
        # predictions
        st.write("Highest Similarity score: ", simscore)
        st.text("Is this one of these the Xpath you're looking for?")
        st.write(st.write(most_acc["input_text"]))
