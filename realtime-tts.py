import streamlit as st
from transformers import pipeline
from datasets import load_dataset
import torch
from pydub import AudioSegment
import io
import numpy as np

# Load dataset and prepare speaker embedding
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

# Function to adjust audio pitch and speed
def adjust_audio(audio_data, pitch_semitones=0.0, speed_factor=1.0):
    try:
        # Check if the input is a numpy array (raw audio data)
        if isinstance(audio_data, np.ndarray):
            # Normalize the raw audio data (assuming float32 range is -1 to 1)
            audio_data = np.int16(audio_data * 32767)  # Scale to int16 range

            # Convert numpy array to AudioSegment
            audio = AudioSegment(
                audio_data.tobytes(),
                frame_rate=16000,  # Assuming 16kHz sample rate
                sample_width=2,    # 2 bytes for 16-bit audio
                channels=1         # Assuming mono channel
            )
        else:
            # Handle other cases, assuming it's already in WAV format
            audio = AudioSegment.from_wav(io.BytesIO(audio_data))

        # Adjust pitch
        if pitch_semitones != 0.0:
            audio = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * (2 ** (pitch_semitones / 12.0)))
            })

        # Adjust speed
        if speed_factor != 1.0:
            audio = audio.speedup(playback_speed=speed_factor)

        # Export to WAV format
        with io.BytesIO() as buf:
            audio.export(buf, format="wav")
            return buf.getvalue()
    except Exception as e:
        st.error(f"Error adjusting audio: {str(e)}")
        return None

# Initialize the Streamlit app
st.title("Text-to-Speech Application")
st.subheader("Transform text into lifelike speech using AI.")

# Load the TTS pipeline
tts = get_tts_pipeline()

# Text input from the user
text_input = st.text_area("Enter the text you want to synthesize:", "Hello, welcome to Streamlit!")

# Slider for pitch and speed adjustment
pitch = st.slider("Adjust Pitch (semitones)", -5.0, 5.0, 0.0)
speed = st.slider("Adjust Speed (factor)", 0.5, 2.0, 1.0)

# Button to generate speech
if st.button("Generate Speech"):
    if text_input.strip():
        with st.spinner("Generating speech..."):
            # Generate the audio from text
            audio_output = tts(text_input, forward_params={"speaker_embeddings": speaker_embedding})

            # Check if the audio is a numpy.ndarray (which indicates it's not in WAV format)
            if isinstance(audio_output["audio"], np.ndarray):
                audio_data = audio_output["audio"]
            else:
                audio_data = audio_output["audio"]

            # Adjust pitch and speed if needed
            adjusted_audio = adjust_audio(audio_data, pitch_semitones=pitch, speed_factor=speed)

            if adjusted_audio:
                # Display the audio player
                st.audio(adjusted_audio, format="audio/wav", sample_rate=16000)

                # Provide an option to download the generated audio
                st.download_button("Download Audio", adjusted_audio, file_name="generated_speech.wav", mime="audio/wav")
    else:
        st.warning("Please enter some text to synthesize.")
