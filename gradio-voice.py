from transformers import pipeline
import whisper
import gradio as gr
# Import the required module for text 
# to speech conversion
import pyttsx3
 
# init function to get an engine instance for the speech synthesis
engine = pyttsx3.init()

p = pipeline("automatic-speech-recognition")
model = whisper.load_model("base")


def transcribe(audio, state=""):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    result_text = result.text
    print("result text --> ", result_text)
    # Now add the lanfChain logic here to process the text and get the responses.
    # once we get the response, we can output it to the voice.
    
    # say method on the engine that passing input text to be spoken
    engine.say(result_text)
 
    # run and wait method, it processes the voice commands.
    engine.runAndWait()
    return result_text, result_text


# Set the starting state to an empty string

gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(source="microphone", type="filepath", streaming=True), "state"],
    outputs=["textbox", "state"],
    live=True,
).launch()
