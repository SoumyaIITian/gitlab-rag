import requests
from bs4 import BeautifulSoup
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
def chunk_and_store(text_list: list):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.create_documents(text_list)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    pass


def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP errors
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')
    main_content = soup.find('main')
    if not main_content:
        print("Main content not found.")
        return ""

    text = main_content.get_text()
    url_list = []
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            url_list.append(href)
    return url_list,text.strip()
def library(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP errors
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')
    main_content = soup.find('main')
    if not main_content:
        print("Main content not found.")
        return ""
    text = main_content.get_text()
    return text.strip()
# Test your function with this:
if __name__ == "__main__":
    test_url = "https://handbook.gitlab.com"
    link_list, raw_text = extract_text_from_url(test_url)
    print(len(link_list))
    lib_list = []
    n=0
    for link in link_list:
        if link.startswith("http") :
            extracted_text = library(link)
            if extracted_text:
                n+=1
                lib_list.append(extracted_text)
                time.sleep(1)
                print(f"Processed {n} links")
            

    response = requests.get("https://about.gitlab.com/releases/whats-new/#whats-coming")
    soup = BeautifulSoup(response.content, 'html.parser')
    t1 = soup.find(id='whats-coming').get_text()
    lib_list.append(t1)

    chunk_and_store(lib_list)