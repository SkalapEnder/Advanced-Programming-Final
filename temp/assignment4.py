import streamlit as st
from langchain_community.document_loaders import AsyncChromiumLoader
from googlesearch import search
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import requests
import ollama
import numpy as np

# Initialize the Tokenizer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Wrapper class for embedding functions
class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        if isinstance(embeddings, np.ndarray):
            return embeddings  # Ensure it's an array
        raise ValueError("Embeddings are not in the expected format.")

    def embed_query(self, text):
        if not isinstance(text, str):
            raise ValueError(f"Expected input of type str, got {type(text)}")
        embedding = self.model.encode([text.strip()])[0]  # Ensure string is stripped
        if isinstance(embedding, np.ndarray):
            return embedding  # Ensure it's a vector
        raise ValueError("Query embedding is not in the expected format.")

# Initialize the embedding function wrapper
embedding_function = EmbeddingFunctionWrapper(model)

# Initialize ChromaDB with the wrapped embedding function
vectorstore = Chroma("internet_documents", embedding_function=embedding_function)

# Define search function
def search_query_internet(query):
    urls = []
    try:
        # Perform Google search
        for url in search(query, num_results=5):    
            print(url)
            urls.append(url)
    except Exception as e:
        print(f"Error during Google search: {e}")

    # Fetch webpage content
    docs = fetch_page_content(urls)
    
    content = []
    for doc in docs:
        text = truncate_text(doc)
        content.append(text)
    return content

# Fetch webpage content using BeautifulSoup with headers to avoid denied accesses errors
def fetch_page_content(urls):
    documents = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            page_content = "\n".join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            documents.append(page_content)
            
            
        except Exception as e:
            print(f"Ошибка при загрузке URL {url}: {e}")
    return documents

# Truncate text for context window
def truncate_text(text):
    words = text.split()
    return " ".join(words[:1000])

# Create prompt for Ollama LLM with search results
def create_prompt_for_llm(llm_query, search_results):
    content_start = "Answer the question using only the context below.\n\nContext:\n"
    content_end = f"\n\nQuestion: {llm_query}\nAnswer:"
    content = content_start + "\n\n---\n\n".join(search_results) + content_end
    prompt = [{'role': 'user', 'content': content}]
    return prompt

def generate_response_ollama(prompt):
    completion = ollama.chat(model='Llama3.2', messages=prompt)
    return completion['message']['content']

# Streamlit app
st.title("RAG Chatbot with Internet Search")

with st.form("prompt_form"):
    user_search_query = st.text_area("Search query:", "")
    user_llm_query = st.text_area("LLM prompt:", "")
    submitted = st.form_submit_button("Send")

    if submitted:
        # Search the internet and fetch results
        search_results = search_query_internet(user_search_query)

        # Embed and store in ChromaDB
        for doc in search_results:
            vectorstore.add_texts([doc])
            
        # Retrieve relevant data from ChromaDB
        query_embedding = embedding_function.embed_query(user_llm_query)
        retrieved_docs = vectorstore.similarity_search_by_vector(query_embedding)
        retrieved_content = " ".join([doc.page_content for doc in retrieved_docs])
        
        # Generate prompt and get the response
        prompt = create_prompt_for_llm(user_llm_query, retrieved_content)
        response = generate_response_ollama(prompt)

        # Display the results
        st.write("### LLM Response:")
        st.write(response)

        st.write("### Search Results:")
        for i, result in enumerate(search_results):
            st.write(f"**Result {i + 1}:** {result}")

        expander = st.expander("LLM Prompt")
        expander.write(prompt)
