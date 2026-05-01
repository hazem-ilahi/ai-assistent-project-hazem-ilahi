
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.tokenize import word_tokenize
import speech_recognition as sr
import base64
import os
import time
from gtts import gTTS # NEW LIBRARY IMPORT
import io # Used to handle the audio file in memory

# --- NLTK/NLP Setup (Cached, Omitted for brevity) ---
# ... (Assume all previous functions like load_intent_model, recognize_intent,
#      extract_features, listen_and_transcribe are present and functional) ...

# Load the model once
@st.cache_resource
def load_intent_model():
    model_filename = 'intent_classifier.joblib'
    try:
        classifier = joblib.load(model_filename)
        return classifier
    except FileNotFoundError:
        st.error("Error: Intent model file (intent_classifier.joblib) not found. Please run the training script.")
        st.stop()

classifier = load_intent_model()

# Helper function to extract features (must be consistent with training)
def extract_features(words):
    features = {}
    for word in words:
        features[f'has({word.lower()})'] = True
    return features

def recognize_intent(text, classifier):
    words = word_tokenize(text.lower())
    features = extract_features(words)
    intent = classifier.classify(features)
    return intent

def listen_and_transcribe():
    r = sr.Recognizer()
    st.info("Listening... Speak now.")
    
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.error("No speech detected within the timeout limit.")
            return None
        
    st.info("Transcribing...")
    
    try:
        text = r.recognize_google(audio)
        st.success(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        st.error("Could not reach the Google Speech Recognition service (check internet connection).")
        return None


# --- 2. TTS/Audio Handling with gTTS (IMPLEMENTATION) ---

def text_to_audio_for_web1(text):
    """
    Generates audio using gTTS, encodes it to base64, and returns an HTML audio player.
    This method streams the audio file in memory (io.BytesIO) to avoid saving local files.
    """
    try:
        # Use io.BytesIO to handle the MP3 file in memory
        mp3_fp = io.BytesIO()
        
        # Create gTTS object and write to the in-memory file handle
        tts = gTTS(text, lang='en', tld='com.au')
        tts.write_to_fp(mp3_fp)
        
        # Seek back to the start of the file before reading
        mp3_fp.seek(0)
        data = mp3_fp.read()
        
        # Encode data to base64 for HTML embedding
        b64 = base64.b64encode(data).decode()
        
        # Return the HTML audio player code with autoplay
        # since the page is loaded all over after a change, autoplay is not good
        return f"""
        <audio controls style="width: 100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    except Exception as e:
        st.error(f"TTS Error: Could not generate audio. Details: {e}")
        return None

# --- 2b. TTS/Audio Handling with pyttsx3 (IMPLEMENTATION) ---

def text_to_audio_for_web2(text):
    """
    Generates audio using gTTS, encodes it to base64, and returns an HTML audio player.
    This method streams the audio file in memory (io.BytesIO) to avoid saving local files.
    """
    # Test pyttsx3 (simple speak)
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        # output will be overwritten every time
        engine.save_to_file(text, './static/output.mp3')
        # engine.say("Environment setup complete. Ready for AI assistance systems!")
        # Text-to-Speech test failed: This means you probably do not have eSpeak or eSpeak-ng installed!. PyAudio might be missing or misconfigured.
        engine.runAndWait()
        for v in voices:
            print(f"Available voices: {v.id}")
        print(f"Selectected voice: {voices[0].id}")
        print("Text-to-Speech output successful.")
        return f"""
        <audio controls style="width: 100%;">
            <source src="./app/static/output.mp3" type="audio/mp3">
        </audio>
        """
    except Exception as e:
        st.error(f"TTS Error: Could not generate audio. Details: {e}")
        return None

if "which_semester_ai" not in st.session_state:
    st.session_state.which_semester_ai = False
elif "which_semester_cy" not in st.session_state:
    st.session_state.which_semester_cy = False

if "language" not in st.session_state:
    st.session_state.language = False
# --- 3. Assistant Core Logic ---

def assistant_action(intent):
    """Maps the recognized intent to a specific action/response."""
    if st.session_state.which_semester_ai:
        if intent == "semester_1":
            st.session_state.which_semester_ai = False
            return "Great! As Artificial intelligence Student at first semester you have a Programming lecture this afternoon at 2 PM"
        elif intent == "semester_3":
            st.session_state.which_semester_ai = False
            return "Great! As Artificial intelligence Student at third semester you have a database lecture at 10 PM"
        else:
            return "Please give me in which semester you study"
    elif st.session_state.which_semester_cy:
        if intent == "semester_1":
            st.session_state.which_semester_cy = False
            return "Great! As Cyber Security Student at first semester you have a Math lecture at 8 PM"
        elif intent == "semester_3":
            st.session_state.which_semester_cy = False
            return "Great! As Cyber Security Student at third semester you have a database lecture at 10 PM"
        else:
            return "Please give me in which semester you study"

    #to see which language the user wants
    elif st.session_state.language:
        if intent == "english_help":
            st.session_state.language = False
            return "Cool! There are available English courses, there one that will start after the exams Periode in Mars, so you have time to register"
        elif intent == "german_help":
            st.session_state.language = False
            return "Cool! There are German available courses, there is one that starts next week, and you still have time to register."
        else:
            return "You need to tell me which language you want to learn"
        
    actions = {
        "greet": "Hello! I am your Campus Info assistant. How can I help you today?",

        "library_hours": "The library opens from 8 AM to 10 PM on weekdays, at 8 AM to 4 PM on weekend",
      
        "cafeteria_hours": "The cafeteria is open from 7 AM to 4 PM." ,

        "meal_today" : "Today's meal is available. Would you like the vegetarian option or the regular menu?",

        "vegetarian" : "Great! Today's vegetarian option is vegetable curry with rice and salad.",

        "regular" : "Great! Today's regular menu includes Pasta with tomato sauce and salad.",

        "lectures_hour": "Sure! I will tell you, what is your study field?",

        "ai_field": "Great! you study Artificial Intelligence, can you tell me in which semester you study ?",

        "cy_field": "Great! you study Cyber Security, can you tell me in which semester you study ?",

        "language_study_support": "Of Course! I will help you! Do you need language or study support?",

        "study_support": "Great! There are tutoring and study support documents available on the website of the university, you just need to choose what's you study field and which semester then you find them.",

        "language_support": "Great! There available Language courses, which language do you want to study? German or English?"

    }

    #chek for users intent, when it comes to continuing the dialog.
    if intent == "language_support":
        st.session_state.language = True
    elif intent == "ai_field":
        st.session_state.which_semester_ai = True
    elif intent == "cy_field":
        st.session_state.which_semester_cy = True
    
    return actions.get(intent, "I'm sorry, I don't know how to handle that request yet. Try asking me about the weather or a joke.")

# --- 4. Streamlit App Layout and Logic ---

st.title("🗣️ Campus-Info-Assitent-Chat")
st.markdown("Use the **text input** or the **Start Voice Input** button to interact.")


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []



# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["audio_html"]:
            st.markdown(message["audio_html"], unsafe_allow_html=True) 

# Function to handle processing of any prompt (text or voice)
def handle_prompt(prompt):
    """Processes the prompt and updates chat history."""
    if not prompt:
        return

    # 1. User message
    st.session_state.messages.append({"role": "user", "content": prompt, "audio_html": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Recognize Intent and Generate Response
    intent = recognize_intent(prompt, classifier)
    response_text = assistant_action(intent)
    
    # 3. Generate TTS Audio
    with st.spinner("Generating audio response..."):
        # The gTTS function is called here
        audio_html = text_to_audio_for_web1(response_text) 

    # 4. Assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)
        if audio_html:
            st.markdown(audio_html, unsafe_allow_html=True)
        st.caption(f"**Intent Detected:** `{intent}`")

    # 5. Append assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text, 
        "audio_html": audio_html
    })


# --- Voice Input Trigger ---
if st.button("🎤 Start Voice Input"):
    with st.spinner("Waiting for speech..."):
        voice_input = listen_and_transcribe()
    
    if voice_input:
        # If voice input is successful, pass the transcribed text to the handler
        handle_prompt(voice_input)
    st.rerun()
else: 
    # --- Text Input Handler ---
    if prompt := st.chat_input("Type a message or click the mic button..."):
        handle_prompt(prompt)