from langchain.utilities import ArxivAPIWrapper
from langchain.tools import Tool



arxiv_client = ArxivAPIWrapper(load_max_docs=1)
#docs = arxiv_client.load(query)
arxiv_tool = Tool.from_function(
        func=arxiv_client.load,
        name="arxiv_client",
        description="Use this tool only to search for any Arxiv search and get the article information",
        return_direct=True
    )