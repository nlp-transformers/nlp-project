import streamlit as st
from audio_recorder_streamlit import audio_recorder

# audio_bytes = audio_recorder()


# The recording will stop automatically
# 2 sec after the utterance end
audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000)



if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")