from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
import whisper
import gradio as gr
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from deep_translator import GoogleTranslator
from random_number_tool import random_number_tool
from youTube_helper import youtube_tool
from url_scraping_tool import url_scraping_tool
from current_time_tool import current_time_tool
from wiki_tool import wiki_tool
from weather_tool import weather_tool
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from tool_retrieval import get_tools

reader = PdfReader('')
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text


text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

#docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(tools)]
model_name = "hkunlp/instructor-xl"
vector_store = Chroma.from_texts(texts, HuggingFaceInstructEmbeddings(model_name=model_name))


chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "who are the authors of the article?"
docs = vector_store.similarity_search(query)
chain.run(input_documents=docs, question=query)
# **** GOOGLE TextToSpeech *****
from gtts import gTTS
import os

# Language in which you want to convert
language = 'en'

# to get input from speech use the following libs
model = whisper.load_model("large")

# define llm
llm = OpenAI(temperature=0.1)


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
    print("detected_language --> ", detected_language)
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
    if result_text != "Unknown language" and len(result_text)!= 0:
        # Now add the lanfChain logic here to process the text and get the responses.
        # once we get the response, we can output it to the voice.
        #agent.
        tools = get_tools(result_text)
        agent = initialize_agent(tools=tools,  llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        print("agent tool --> ", agent.tools)
        agent_output = agent.run(result_text)
