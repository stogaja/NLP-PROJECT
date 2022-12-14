import io
import netrc
import pickle
import sys
import pandas as pd
import streamlit as st
# let's import sentence transformer
#import sentence_transformers
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
# loaded_model = pickle.load(open('XpathFinder1.sav', 'rb'))
#loaded_model = pickle.load('XpathFinder1.sav', map_location='cpu')


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


loaded_model = CPU_Unpickler(open('XpathFinder1.sav', 'rb')).load()


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
   # # importing data from csv files
   #  data = pd.read_csv('streamlit template/JC-202103-citibike-tripdata.csv')

   #  # 5 --- You can work with data, change it and filter it as you always do using Pandas or any other library
   #  start_station_list = ['All'] + data['start station name'].unique().tolist()
   #  end_station_list = ['All'] + data['end station name'].unique().tolist()

    # 6 --- collecting input from the user
    #		Steamlit has built in components to collect input from users

    # collect input using free text
    text_input = st.text_input("Enter the text in the space below ...")

   # let's pass the input to the model
   # let's pass the input to the model
    if text_input:
        # let's get the input from the user
        input_text = text_input
        # let's get the prediction
        prediction = loaded_model.predict([input_text])
        # let's print the prediction
        st.write(f"Your input: {input_text}")
        st.write(f"Your prediction: {prediction[0]}")
