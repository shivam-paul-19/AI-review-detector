import streamlit as st
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

with open("models/model.pkl", 'rb') as file:
    model = pickle.load(file)

with open("models/tf_vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

class TextPreprocessor:

    def __init__(self):
        # load once
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.stop_words = set(stopwords.words("english"))

    def process(self, corpus):

        # tokenize
        words = word_tokenize(corpus)

        # remove stopwords
        words = [w.lower() for w in words if w.lower() not in self.stop_words]

        # lemmatize
        doc = self.nlp(" ".join(words))

        return [token.lemma_ for token in doc]

processor = TextPreprocessor()

def process(review):
    p = processor.process(review)
    p = [token for token in p if token != "."]
    return " ".join(p)

def vectorise(text):
    return vectorizer.transform([text]).toarray()

def makeDF(rating, review):
    vector_df = pd.DataFrame(vectorise(process(review)))
    vector_df.columns = vector_df.columns.astype(str)
    rating_df = pd.DataFrame({
        'rating': [float(rating)]
    })
    additional_features = pd.DataFrame({
        'review_len': [0],
        'word_count': [0],
        'avg_word_len': [0],
        'word_to_len_ratio': [0]
    })

    return pd.concat([rating_df, additional_features, vector_df], axis=1)

def predict(rating, review):
    input_df = makeDF(rating=rating, review=review)
    return {
        'result': model.predict(input_df),
        'probab': model.predict_proba(input_df)
    }

st.title("🤖 AI review detector")
rating = st.slider(label="Rating", min_value=0, max_value=5)
review = st.text_area(label="Enter the review")

if(st.button(label="check") and review != ''):
    result = predict(rating=rating, review=review)
    if result['result'] == 0:
        res = "Human written"
        human_perc = max(result['probab'][0])*100
        ai_perc = min(result['probab'][0])*100
    else:
        res = "AI Generated"
        ai_perc = max(result['probab'][0])*100
        human_perc = min(result['probab'][0])*100

    st.markdown(f" ## Result: {res}")
    st.markdown(f" ### Confidence: {max(result['probab'][0])*100:.1f}%")
    st.progress(ai_perc/100, text=f"🤖 **AI pattern** found {ai_perc:.1f}%")
    st.progress(human_perc/100, text=f"🧠 **Human pattern** found {human_perc:.1f}%")

    if(abs(human_perc - ai_perc) < 20):
        st.warning("⚠️ Note: The confidence level for this prediction is relatively moderate. This means the detected patterns are not strongly aligned with either class, so the result should be interpreted with caution.")