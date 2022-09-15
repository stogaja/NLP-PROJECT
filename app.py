import io
import netrc
import pickle
import sys
import pandas as pd
import streamlit as st
# let's import sentence transformer
import sentence_transformers
import torch
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
loaded_model = pickle.load(open('XpathFinder1.sav', 'rb'))
#loaded_model = pickle.load('XpathFinder1.sav', map_location='cpu')


#class CPU_Unpickler(pickle.Unpickler):
#    def find_class(self, module, name):
#        if module == 'torch.storage' and name == '_load_from_bytes':
#            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#        else:
#            return super().find_class(module, name)
#

#loaded_model = CPU_Unpickler(open('XpathFinder1.sav', 'rb')).load()


# here is how to create containers
header_container = st.container()
stats_container = st.container()

# the header
with header_container:

    # different levels of text you can include in your app
    st.title("Semantic Text Similarity App")
    st.header("Hello!, Let's match some words!!!")

# Another container
with stats_container:
    # collect input using free text
    text_input = st.text_input("Enter the text in the space below ...")

    # let's pass the input to the loaded_model with torch compiled with cuda
    if text_input:
        # let's get the result
        result = loaded_model.predict([text_input])
        # let's show the result
        st.write(result)
