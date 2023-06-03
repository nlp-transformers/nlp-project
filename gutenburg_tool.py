from langchain.document_loaders import GutenbergLoader
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.tools import Tool
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import requests
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

class GutenbergAPIWrapper:
    BASE_URL = "http://gutenberg.org"

    def retrieve_book(self, book_id: int):
        url = f"{self.BASE_URL}/ebooks/{book_id}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            title_element = soup.find("h1", class_="title")
            if title_element:
                book_title = title_element.text.strip()
                return book_title
        return None

gutenberg_wrapper = GutenbergAPIWrapper()

class GutenbergInput(BaseModel):
    book_id: int = Field()

gutenberg_doc_tool = Tool.from_function(
    name="gutenberg",
    func=gutenberg_wrapper.retrieve_book,
    args_schema=GutenbergInput,
    description="Use this tool to retrieve a book from the Gutenberg free text library. Input to the tool should be the book ID.",
)

def retrieve_book_metadata(book_id):
    loader = GutenbergLoader()
    metadata = loader.get_metadata(book_id)
    return metadata

def retrieve_book_summary(book_id):
    loader = GutenbergLoader()
    summary = loader.get_summary(book_id)
    return summary


def read_book_info(book_title_or_id):
    if isinstance(book_title_or_id, str):
        book = gutenberg_doc_tool(book_title_or_id)
    else:
        book = gutenberg_wrapper.retrieve_book(book_title_or_id)

    if book is not None:
        # Retrieve the book metadata
        book_metadata = retrieve_book_metadata(book)

        # Retrieve the book summary
        book_summary = retrieve_book_summary(book)

        # Return the book metadata and summary
        return book_metadata, book_summary
    else:
        return None
