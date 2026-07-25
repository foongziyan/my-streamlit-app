import streamlit as st
import nltk
from joblib import load

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the saved model
bayes = load('bayes_classifier.joblib')

# Load vocabulary
with open('top_keys.txt', 'r') as f:
    topKeys = [word.strip() for word in f.readlines()]

def review_features_sparse(tokens):
    docSet = set(tokens)
    return {word: (word in docSet) for word in topKeys}

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")

review_input = st.text_area("Review:")

if st.button("Predict Sentiment"):
    if review_input.strip():
        tokens = nltk.word_tokenize(review_input.lower())
        feats = review_features_sparse(tokens)
        prob_dist = bayes.prob_classify(feats)
        pred = prob_dist.max()
        
        st.success(f"Predicted sentiment: {pred}")
        st.write(f"pos = {prob_dist.prob('pos'):.3f}, neg = {prob_dist.prob('neg'):.3f}")
    else:
        st.warning("Please enter a review first.")