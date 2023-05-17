from transformers import pipeline
import whisper
import gradio as gr
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from deep_translator import GoogleTranslator
from langchain.tools import YouTubeSearchTool
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


# Import the required module for text
# to speech conversion
#from AppKit import NSSpeechSynthesizer
# nssp = NSSpeechSynthesizer
# ve = nssp.alloc().init()

# import the openAI
llm = OpenAI(temperature=0.9)

# to get input from speech use the following libs
model = whisper.load_model("medium")

def youtube_search(query: str):
    query_string = query#+",1"
    url_list = YouTubeSearchTool().run(query_string).strip('][').split(', ')
    video_id = url_list[0].strip("'").split("?v=")[1]
    # print(video_id)
    try:
        video_transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        video_transcript = ','.join(map(str, video_transcript_list))
    except:
        video_transcript = "Could not find transcript for this video."

    # print(video_transcript)
    result_url = 'https://www.youtube.com' + url_list[0].strip("'")
    return "The youtube video on "+query+" is available at "+result_url+" ."+"The transcript is as follows: "+video_transcript

youtube_tool = Tool.from_function(
        func=youtube_search,
        name="YouTube",
        description="Tool to search for YouTube videos. Prefer this over normal search when searching for videos.",
        return_direct=True
        #args_schema=CalculatorInput
        # coroutine= ... <- you can specify an async method if desired as well
    )

# create a list of tools
tool_names = [
    "serpapi",  # for google search
    "llm-math",  # this particular tool needs an llm too, so need to pass that
]

tools = load_tools(tool_names=tool_names, llm=llm)
tools.append(youtube_tool)

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
    detected_language = max(probs, key=probs.get)
    if detected_language == "en":
        print("Detected English language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language="en")
        result = whisper.decode(model, mel, options)
        result_text = result.text
    elif detected_language == "ta":
        print("Detected Tamil language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language="ta")
        tamil = whisper.decode(model, mel, options)
        print(tamil.text)
        result_text = GoogleTranslator(source='ta', target='en').translate(tamil.text)

        # transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium",
        #                       chunk_length_s=30, device="cpu")
        # transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")
        # print('Transcription: ', transcribe(audio)["text"])
        # result_text = "Unknown language"
    else:
        result_text = "Unknown language"

    print("result text --> ", result_text)
    if result_text != "Unknown language":
        # Now add the lanfChain logic here to process the text and get the responses.
        # once we get the response, we can output it to the voice.
        output = agent.run(result_text)
    else:
        output = "I'm sorry I cannot understand the language you are speaking. Please speak in English or Tamil."

    # # say method on the engine that passing input text to be spoken
    # ve.startSpeakingString_(output)

    return output, output


# Set the starting state to an empty string

gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(source="microphone", type="filepath", streaming=False), "state"],
    outputs=["textbox", "state"],
    live=True,
).launch()
