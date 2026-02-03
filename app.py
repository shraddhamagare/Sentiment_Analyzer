# app.py

import streamlit as st
import pickle

# --- Load model and vectorizer ---
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# --- Streamlit App Setup ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📝", layout="centered")
st.title("📊 Sentiment Analyzer")
st.write("Enter text below and find out if the sentiment is Positive, Negative, or Neutral!")

# --- Initialize history in session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Function to display sentiment with color and emoji ---
def display_sentiment(prediction):
    if prediction.lower() == "positive":
        return f"😊 Positive"
    elif prediction.lower() == "negative":
        return f"😞 Negative"
    else:
        return f"😐 Neutral"

# --- User input ---
user_input = st.text_area("Enter your text here:")

# --- Analyze button ---
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter some text to analyze.")
    else:
        # Transform the text using vectorizer
        input_vector = vectorizer.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(input_vector)[0]
        
        # Format result with emoji
        sentiment_result = display_sentiment(prediction)
        
        # Display the result in color
        if prediction.lower() == "positive":
            st.markdown(f"<h2 style='color:green'>{sentiment_result}</h2>", unsafe_allow_html=True)
        elif prediction.lower() == "negative":
            st.markdown(f"<h2 style='color:red'>{sentiment_result}</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:blue'>{sentiment_result}</h2>", unsafe_allow_html=True)
        
        # Add to history (keep max 5 items)
        st.session_state.history.insert(0, (user_input, sentiment_result))
        if len(st.session_state.history) > 5:
            st.session_state.history.pop()

# --- Display history ---
if st.session_state.history:
    st.markdown("### 🕒 Last 5 Analyses")
    for idx, (text, sentiment) in enumerate(st.session_state.history, 1):
        st.markdown(f"**{idx}.** {text} → {sentiment}")
