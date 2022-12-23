

# streamlit run main.py

import streamlit as st
import nltk
import pandas as pd
import os
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import sys
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

pd.options.mode.chained_assignment = None

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#%%
keywords_df = pd.read_csv('D:\\Projects\\Keywords_Clustering\\data\\training_data\\KWR_Thomas.csv', sep=',')




def data_preprocessing(df):
    # check if KEYWORD column does exists:
    if 'Keyword' not in df.columns:
        sys.exit("Keyword column does not exist in the data or is misspelled."
                 "consider fixing this error and try it again ")
    # data translate:
    df.dropna(inplace=True)
    print("transliting")
    df["Keyword_ENG"] = df["Keyword"].apply(lambda x: GoogleTranslator(source='auto', target='en').translate(x))
    # remove question related words, otherwise K-MNEANS will cluster them as a cluster:
    df['Keyword_ENG'] = df['Keyword_ENG'].str.replace(r'what is', '')
    df['Keyword_ENG'] = df['Keyword_ENG'].str.replace(r'why is', '')
    df['Keyword_ENG'] = df['Keyword_ENG'].str.replace(r'what', '')
    df['Keyword_ENG'] = df['Keyword_ENG'].str.replace(r'why', '')
    df['id'] = range(len(df))
    col = df.pop("id")
    df.insert(0, col.name, col)


    return (df)

