import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import re

#@st.cache()
@st.cache(allow_output_mutation=True)
def load_model():
	"""Retrieves the trained model"""
	filename = 'RFC_tuned.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
	return loaded_model, vectorizer
# reference : https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
def cleanhtml(raw_html, CLEANR):
	cleantext = re.sub(CLEANR, '', raw_html)
	return cleantext

# reference : https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
def decontracted(phrase):
	# mathematical symbols 
	phrase = re.sub(r'000,000', 'm',phrase)
	phrase = re.sub(r'000','k',phrase)

	# specific
	phrase = re.sub(r"won\'t", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)

	# general
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"\'s", " is", phrase)
	phrase = re.sub(r"\'d", " would", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'t", " not", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	return phrase

def preprocessText(text):
	# convert text to lower case
	text = text.lower()
	# remove html tags and unknown unicode characters like &nbsm etc
	CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
	text = cleanhtml(text, CLEANR)
	# remove contractions
	text = decontracted(text)
	return text

def getSimilarity(q1,q2, model, vectorizer):
	q1, q2 = preprocessText(q1), preprocessText(q2)
	vect1 = vectorizer.encode(q1)
	vect2 = vectorizer.encode(q2)
	vect = np.concatenate((vect1,vect2))
	prediction = model.predict(vect.reshape(1,-1))[0]
	prediction_proba = round(model.predict_proba(vect.reshape(1,-1))[0][1]*100,2)
	if prediction == 0 : 
		return 'Given questions are not similar with similarity score of ' + str(prediction_proba) + '%'
	elif prediction == 1 : 
		return 'Given questions are similar with similarity score of ' + str(prediction_proba) + '%'

def main():
	model, vectorizer = load_model()
	st.title("Welcome to Questions similarity prediction app")
	q1 = st.text_input('Enter question one')
	q2 = st.text_input('Enter question two')
	if st.button('submit'):
		output = getSimilarity(q1,q2,  model, vectorizer)
		st.write(output)

if __name__ == '__main__' : 
	main()