"""
This file contains document ingestion and chunking modules.

This stage includes:
1.	Loads documents
2.	Extracts raw text
3.	Splits text into chunks
4.	Attaches metadata (source, page)

"""

# import dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def text_cleaner(docs):
    """
    Args:
    docs: input document with whitespaces, tabs, leading and trailing spaces
    Returns:
    docs: document free of tabs, leading and trailing spaces 
    """
    # for each ith page in docs
    for i in range(len(docs)):
        docs[i].page_content = docs[i].page_content.strip()              # remove leading and traling whitespaces from page_content field and save it back in page_content field
        docs[i].page_content = docs[i].page_content.replace('\t'," ")    # replace tabs with single whitespace from page_content field and save it back in page_content field
    return docs

# loading documents
file_path = "book.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
# print(docs[5].page_content[:300])  # extract first 300 characters of content on 6th page 
# print(docs[5].metadata['author'])  # extract author name from metadata
# print(docs[5].metadata)            # metadata
# print(len(docs))                   # number of pages in document

# clean the document 
clean_doc = text_cleaner(docs)

# chunking from text using recursive character text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)  # chunk size of 800 characters roughly, overlapping characters 150 for context.
texts = text_splitter.split_documents(clean_doc)  # returns list of chunks. Each chunk contains cleaned text content, metadata 