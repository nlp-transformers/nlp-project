#from transformers import pipeline
import whisper
import gradio as gr
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from deep_translator import GoogleTranslator
from random_number_tool import random_number_tool
from youTube_helper import youtube_search
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase

#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
#from pydantic import BaseModel, Field


# **** GOOGLE TextToSpeech *****
from gtts import gTTS
import os

  
# Language in which you want to convert
language = 'en'

# define llm
llm = OpenAI(temperature=0.1)

# to get input from speech use the following libs
model = whisper.load_model("medium")

postdb = SQLDatabase.from_uri("postgresql://abhi:mango@localhost:5432/abhi?sslmode=disable")
toolkit = SQLDatabaseToolkit(db=postdb, llm=llm)

agent = create_sql_agent(
    llm=OpenAI(temperature=1.0),
    toolkit=toolkit,
    verbose=True
)

# define youtube search tool
youtube_tool = Tool.from_function(
        func=youtube_search,
        name="YouTube",
        description="Use this tool only to search for videos, songs and youtube. Prefer this over normal search when searching for videos.",
        return_direct=True
        #args_schema=CalculatorInput
        # coroutine= ... <- you can specify an async method if desired as well
    )





# create a list of tools
tool_names = [
    "serpapi",  # for google search
    "llm-math",  # this particular tool needs an llm too, so need to pass that
    "openweathermap-api",
    "arxiv",
]

tools = load_tools(tool_names=tool_names, llm=llm)
tools.append(youtube_tool, Wiki_tool, random_number_tool)

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
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language=language)
        result = whisper.decode(model, mel, options)
        result_text = result.text
    elif detected_language == "ta":
        print("Detected Tamil language.")
        options = whisper.DecodingOptions(fp16=False, task="transcribe", language="ta")
        tamil = whisper.decode(model, mel, options)
        print(tamil.text)
        result_text = GoogleTranslator(source='ta', target=language).translate(tamil.text)

        # transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium",
        #                       chunk_length_s=30, device="cpu")
        # transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

    else:
        result_text = "Unknown language"

    print("result text --> ", result_text)
    if result_text != "Unknown language":
        # Now add the lanfChain logic here to process the text and get the responses.
        # once we get the response, we can output it to the voice.
        agent_output = agent.run(result_text, )
    else:
        agent_output = "I'm sorry I cannot understand the language you are speaking. Please speak in English or Tamil."

    # init some image and video. Override based on agent output.
    detailed = ''
    image_path = 'supportvectors.png'
    video_path = 'welcome.mp4'

    if "tool" in agent_output:
        print("This is an article.")
        if agent_output["tool"] == "youtube":
            tldr = agent_output["tldr"]
            detailed = agent_output["article"]
            if "video" in agent_output:
                video_path = agent_output["video"]

    else:
        print("This is not an article. It is coming from agent.")
        tldr = agent_output


    # TTS. Marked slow=False meaning audio should have high speed
    myobj = gTTS(text=tldr, lang=language, slow=False)
    # Saving the converted audio in a mp3 file named
    myobj.save("welcome.mp3")
     # Playing the audio
    os.system("mpg123 welcome.mp3")

    # prompt = PromptTemplate(
    #     input_variables=["state"],
    #     template="Is the statement talking about a person or place or animal or a thing? please answer in 2 words what is it name of it {state}?",
    # )
    # prompt.format(state = output)
    # chain = LLMChain(llm=llm, prompt=prompt)
    # author = chain.run(output)
    # print(f'the topic is about {author}')

    return tldr, detailed, image_path, video_path, tldr




# Set the starting state to an empty string
gr.Interface(
    fn=transcribe,
    inputs=[gr.Audio(source="microphone", type="filepath", streaming=False), "state"],
    outputs=["textbox", "textbox", "image", "video", "state"],
    live=True,
).launch(share=True)
