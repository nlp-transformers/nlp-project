from transformers import pipeline
import whisper
import gradio as gr

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

    return result_text, result_text


# Set the starting state to an empty string

gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(source="microphone", type="filepath", streaming=True), "state"],
    outputs=["textbox", "state"],
    live=True,
).launch()
