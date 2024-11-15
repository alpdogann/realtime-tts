import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import torch

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


# Function to load the TTS pipeline
@st.cache_resource
def get_tts_pipeline():
    """
    Load the text-to-speech pipeline using the microsoft/speecht5_tts model.
    This function is cached to avoid reloading the model on every execution.
    """
    return pipeline("text-to-speech", model="microsoft/speecht5_tts")


# Initialize the Streamlit app
st.title("Text-to-Speech Application")
st.subheader("Transform text into lifelike speech using AI.")

# Load the TTS pipeline
tts = get_tts_pipeline()

# Text input from the user
text_input = st.text_area("Enter the text you want to synthesize:", "Hello, welcome to Streamlit!")

# Button to generate speech
if st.button("Generate Speech"):
    if text_input.strip():
        with st.spinner("Generating speech..."):
            # Generate the audio from text
            audio_output = tts(text_input, forward_params={"speaker_embeddings": speaker_embedding})

        # Display the audio player
        st.audio(audio_output["audio"], format="audio/wav", sample_rate=16000)
    else:
        st.warning("Please enter some text to synthesize.")
