from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from summarize_docs import summarize_docs
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

def get_wikipedia_article(query):
    wikipedia = WikipediaAPIWrapper()
    text_splitter = CharacterTextSplitter()
    wikipedia_result = wikipedia.run(query)
    texts = text_splitter.split_text(wikipedia_result)
    # print(len(texts))
    docs = [Document(page_content=t) for t in texts[:3]]
    summary_text = summarize_docs(docs)
    # print(summary_text)
    result = {
        "tool": "wikipedia",
        "tldr": summary_text,
        "article": wikipedia_result
    }
    return result

# define wiki search tool
wiki_tool = Tool.from_function(
        func=get_wikipedia_article,
        name="Wikipedia",
        description="Use this tool only to search for articles in wikipedia or detailed information. Prefer other tools over this tool, use this if all the tools fail.",
        return_direct=True
    )