from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from summarize_docs import summarize_docs
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import requests

class GutenbergBookRetriever:
    BASE_URL = "http://gutenberg.org/cache/epub/"

    def retrieve_book_content(self, book_id: int):
        url = f"{self.BASE_URL}/{book_id}/pg{book_id}.txt"
        response = requests.get(url)
        text = response.text
        return text

def summarize_docs(docs):
    llm = OpenAI(openai_api_key="", temperature=0.4)
    prompt_template = """Write a concise summary of the following, not more than 50 words:
        {text}
        SUMMARY IN FIVE BULLET POINTS:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    summary_text = ""
    return summary_text
def get_book_title(book_id):
    url = f"http://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    text = response.text
    title = ""
    for line in text.split("\n"):
        if line.startswith("Title:"):
            title = line.replace("Title:", "").strip()
            break
    return title

def get_book_author(book_id):
    url = f"http://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    text = response.text
    author = ""
    for line in text.split("\n"):
        if line.startswith("Author:"):
            author = line.replace("Author:", "").strip()
            break
    return author

def get_release_date(book_id):
    url = f"http://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    text = response.text
    release_date = ""
    for line in text.split("\n"):
        if line.startswith("Release Date:"):
            release_date = line.replace("Release Date:", "").strip()
            break
    return release_date

def get_book_language(book_id):
    url = f"http://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    response = requests.get(url)
    text = response.text
    language = ""
    for line in text.split("\n"):
        if line.startswith("Language:"):
            language = line.replace("Language:", "").strip()
            break
    return language

def get_gutenberg_book_summary(book_id):
    book_retriever = GutenbergBookRetriever()
    book_content = book_retriever.retrieve_book_content(book_id)
    book_title = get_book_title(book_id)
    book_author = get_book_author(book_id)
    release_date = get_release_date(book_id)
    language = get_book_language(book_id)


    text_splitter = CharacterTextSplitter()
    book_content = book_retriever.retrieve_book_content(book_id)
    texts = text_splitter.split_text(book_content)
    docs = [Document(page_content=book_content)]
    # docs = [Document(page_content=book_content) for t in texts[:3]]
    summary_text = summarize_docs(docs)

    result = {
        "tldr": summary_text,
        "book_title": book_title,
        "book_author": book_author,
        "release_date": release_date,
        "language": language,
        # "book_content": book_content
    }
    return result

gutenberg_doc_tool = Tool.from_function(
    name="gutenberg",
    func=get_gutenberg_book_summary,
    return_direct=True,
    description="Use this tool to retrieve a book information from the Gutenberg free text library. Input to the tool should be the book ID.",
)

book_id = 123
summary_result = get_gutenberg_book_summary(book_id)
print(summary_result)
