from transformers import pipeline
import whisper
import gradio as gr
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

# Import the required module for text
# to speech conversion
from AppKit import NSSpeechSynthesizer

nssp = NSSpeechSynthesizer

ve = nssp.alloc().init()

# import the openAI
llm = OpenAI(temperature=0.9)

# to get input from speech use the following libs
p = pipeline("automatic-speech-recognition")
model = whisper.load_model("base")

# create a list of tools
tool_names = [
    "serpapi",  # for google search
    "llm-math",  # this particular tool needs an llm too, so need to pass that
]

tools = load_tools(tool_names=tool_names, llm=llm)

# initialize them
agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# core function which will do all the work (POC level code)
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
    output = agent.run(result_text)
    # # say method on the engine that passing input text to be spoken
    ve.startSpeakingString_(output)

    return result_text, result_text


# Set the starting state to an empty string

gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(source="microphone", type="filepath", streaming=False), "state"],
    outputs=["textbox", "state"],
    live=True,
).launch()
