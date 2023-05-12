import time
import streamlit as st
import os
import pyaudio
import wave
import threading

class AudioRecorder:

    def __init__(self) -> None:
        self.pa = pyaudio.PyAudio()
        self.stream = pyaudio.Stream
        self.recording = False
        self.audio = pyaudio.PyAudio()

    def record(self):
        st.button('Record', on_click=self.recordVoice)

    def recordVoice(self):
        print(self.recording)
        # Display the record button
        if not self.recording:
            self.recording = True
            threading.Thread(target=self.start).start()
        else:
            self.stop()
            self.recording = False
        print(self.recording)

    def start(self):
        print("record start")
        stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []

        while self.recording:
            data = stream.read(1024)
            frames.append(data)


    def stop(self):
        print("record stop")
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


        



