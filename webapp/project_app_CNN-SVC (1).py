##################################################
# Import librairies
##################################################

import streamlit as st
import pandas as pd
import pickle
import datetime
import re
from sklearn import datasets
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC

import pandas as pd, numpy as np
import gensim, pickle

# Preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

# Neural network modeling
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, AveragePooling1D, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.optimizers import SGD

# Personal tweet cleaner from tweet_cleaner.py
import tweet_cleaner

##################################################
# Preprocessing and pretrained model methods
##################################################

# Text to sequence method based on disaster relevancing tokenizer fit
def relevance_sequence(text):
    
    # Load saved Tokenizer()
    with open('relevance_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    # Convert to sequence    
    sequence = tokenizer.texts_to_sequences([text])
    
    # Returned padded sequence
    return pad_sequences(sequence, maxlen=60)

# Relevance model
def relevance_model(text):
    
    # Load best pretrained model with low loss+variance -- 83% accuracy train+test
        # Embedding+CNN(32)+CNN(64)+Dense(32)+Dense(1)
        # (loss: 0.3918 - acc: 0.8354 - val_loss: 0.4055 - val_acc: 0.8310)
    model = load_model('relevance_model_nonGPU.hdf5')
    
    # Predict relevance-to-disaster probability
    relevance_ratio = model.predict(text)[0,0]
    
    # Responses depending on relevancy
    if relevance_ratio < 0.4:
        st.error(f'Your tweet is considered as not relevant as it is only {round(relevance_ratio*100, 2)}% related to disasters.')
        return relevance_ratio
    elif relevance_ratio < 0.6:
        st.warning(f'Your tweet is considered as not entirely relevant as it is only {round(relevance_ratio*100, 2)}% related to disasters.')
        return relevance_ratio
    else:
        st.success(f'Your statement is {round(relevance_ratio*100, 2)}% related to disasters.')
        return relevance_ratio
    
# Snowball Stemmer analyzer method for credibility model
def stemmed_snow(doc):
    s_stemmer = SnowballStemmer(language='english') 
    analyzer = TfidfVectorizer().build_analyzer()
    return (s_stemmer.stem(w) for w in analyzer(doc))    
    
# Credibility model - SVC piped with TF-IDF vectorizer
def credibility_model(text):
    
    # Load best pretrained model with low loss+variance -- 98% accuracy train+test
        # Embedding+CNN(32)+CNN(64)+Dense(32)+Dense(1)
        # (loss: 0.0403 - acc: 0.9831 - val_loss: 0.0424 - val_acc: 0.9832)
        
    with open('svc_model.pickle', 'rb') as handle:
        model = pickle.load(handle)
    
    # Predict relevance probability
    prediction = model.predict([text])[0]
    
    # Responses depending on credibility                   
    if prediction < 0.4:
        st.error(f'Your statement is not credible')
        st.error("You've provided misinformation, please try telling the truth!")
    elif prediction < 0.6:
        st.warning(f'Your statement is not sufficiently credible')
        st.warning("You've provided information that is not entirely true, please try telling the truth!")
    else:
        st.success(f'Your statement is credible')
        st.success("You've provided valid information. Thanks!") 

def misinformation_verifier(user_input):
    
    # Clean tweet
    cleaned_input = tweet_cleaner.clean(user_input)
    st.subheader('Your "cleaned" tweet to analyze')
    st.write(cleaned_input)
    
    # Sequenced & fed through relevance model
    st.subheader('Is your tweet related to a natural disaster?')
    sequenced_r = relevance_sequence(cleaned_input)
    relevance_ratio = relevance_model(sequenced_r)
    
    # End if statement is irrelevant (un-related to disaster)
    if relevance_ratio < 0.5:
        st.error("IRRELEVANT! Please try another tweet.")
    
    # Otherwise, confirm credibility
    else:
        # Re-sequenced & fed through credibility model
        st.subheader("""
        Tweeter confidence index
        How confident can you be of this tweet being **accurate more than a rumor**?
        """)
        credibility_model(cleaned_input)


##################################################
# Web-app
##################################################

st.write("""
# Twitter confidence index
# Spotting misinformation about natural disasters
This app predicts if your tweet is related to a **natural disasters** and if so, how confident you may be in relying on it as a **true information**!
""")

##################################################
# Input tweet (sidebar)
##################################################

st.sidebar.header('Tweet parameters')
tweet = st.sidebar.text_area('Tweet to analyze')
  
##################################################
# Verify relevance & credibility
##################################################
    
if not tweet:
    st.warning("No tweet to analyse. Please fill out the tweet required field") 
else:
    misinformation_verifier(tweet)
