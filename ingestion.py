import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeSparseVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv(override=True)

# Configure the SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10
)

#chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeSparseVectorStore(index_name="langchain-chain-index", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def main():
    """Main async process to automate all the process"""
    log_header("Documentation Ingestion Pipeline")

    log_info(
        "🔍 TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )

    # Crawl the documentation
    tavily_crawl_results = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 1,
        "extract_depth": "advanced",
        "instruction": "Documentatin relevant to ai agents"
    })
    all_docs = [Document(page_content=tavily_crawl_results['raw_content'], metadata={"source": tavily_crawl_results['url']}) for result in tavily_crawl_results['results']]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results)} URLs from documentation site"
    )


if __name__ == '__main__':
    asyncio.run(main())



