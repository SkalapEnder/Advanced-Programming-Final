from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from googlesearch import search
from bs4 import BeautifulSoup
import streamlit as st
import requests
import ollama
import numpy as np

API_KEY_M = "AIzaSyB4emWYmzAFTtydMRTYFbA5Jf9laUJUnDc"
API_KEY = "AIzaSyDpZt-JvZcrZ9mBX8G_8V-BIanWg5XeXbk"
CSE_ID_M = "07b55181ea4c246ac"
CSE_ID = "c78c141a8dc4647ab"

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
            urls.append(url)
    except Exception as e:
        print(f"Error during Google search: {e}")

    # Fetch webpage content
    docs = fetch_web_content(urls)
    
    content = []
    for doc in docs:
        text = truncate_text(doc)
        content.append(text)
    return content

# Fetch webpage content using BeautifulSoup with headers to avoid denied accesses errors
def fetch_web_content(urls):
    documents = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract table content as a single text
            table_text = ""
            tables = soup.find_all('table')
            for table in tables:
                for row in table.find_all('tr'):
                    row_text = " ".join(cell.get_text(strip=True) for cell in row.find_all(['th', 'td']))
                    table_text += row_text
            
            # Extract paragraph content
            paragraphs = soup.find_all('p')
            paragraph_text = "\n".join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            # Combine table and paragraph text
            combined_text = f"Table Content:\n{table_text}\n\nParagraph Content:\n{paragraph_text}"
            documents.append(combined_text)

        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
    
    return documents

# Truncate text for context window
def truncate_text(text):
    words = text.split()
    return " ".join(words[:1000])

def google_search(query, search_type='text'):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': API_KEY,
        'cx': CSE_ID,
        'searchType': search_type,
        'q': query
    }

    response = requests.get(url, params=params)
    return response.json()

def scrape_images_from_internet(query, num_images):
    results = google_search(query, "image")

    image_urls = [item['link'] for item in results.get('items', [])[:num_images]]  
    return image_urls

# Create prompt for Ollama LLM with search results and images
def create_prompt_for_llm(llm_query, search_results, with_images, images, only_images=False):
    if only_images: return [{'role': 'user', 'content': 'Describe each image from provided many' + "\n".join(images)}]
    
    content_start = ""
    prompt = []
    
    if with_images: content_start = "Answer the question using the context and provided images. Images:" + ("\n".join(images)) + " \n\nContext:\n"
    else: content_start = "Answer the question using only the context below.\n\nContext:\n"
    
    content_end = f"\n\nQuestion: {llm_query}\nAnswer:"
    content = content_start + "\n".join(search_results) + content_end
    
    prompt = [{'role': 'user', 'content': content}]
    return prompt

# Generate response
def generate_response_ollama(model, prompt):
    completion = ollama.chat(model=model, messages=prompt)
    return completion['message']['content']

# Streamlit app
st.title("RAG Chatbot with Internet Search")

# Chatbot section
with st.form("prompt_form"):
    st.subheader("1. Enter Your Search Query")
    st.markdown("Enter keywords or phrases to search for relevant information on the internet. For best results, be specific and concise.")
    user_search_query = st.text_area("Search query:", "", help="Example: 'Latest developments in artificial intelligence 2024'")
    
    st.subheader("2. Enter Your Question")
    st.markdown("What would you like to know about the search results? Be specific to get more accurate answers.")
    user_llm_query = st.text_area("LLM prompt:", "", help="Example: 'What are the key trends and breakthroughs mentioned in these results?'")
    
    st.subheader("3. Choose Search Options")
    st.markdown("Select how you want to analyze the information:")
    search_type = st.radio(
        "Select search type:",
        ('Text', 'Image', 'Both'),
        help="""
        - Text: Search and analyze text content only
        - Image: Search and analyze images only
        - Both: Combine text and image analysis
        """
    )
    
    st.markdown("#### Number of Images")
    number_images = st.number_input(
        "Select number of images to analyze:", min_value=3, max_value=10, value=3,
        help="More images may increase processing time"
    )
    
    submitted = st.form_submit_button("Send")

    if submitted:
        try:
            # Initialize progress
            progress = st.progress(0)
            status = st.empty()
            
            # Common steps for all types
            status.text("Searching and processing content...")
            search_results = search_query_internet(user_search_query)
            progress.progress(20)
            
            # Store in ChromaDB and retrieve
            for doc in search_results:
                vectorstore.add_texts([doc])
            
            query_embedding = embedding_function.embed_query(user_llm_query)
            retrieved_docs = vectorstore.similarity_search_by_vector(query_embedding)
            retrieved_content = " ".join([doc.page_content for doc in retrieved_docs])
            progress.progress(40)
            
            if search_type == 'Text':
                # Text-only analysis
                status.text("Generating text analysis...")
                prompt = create_prompt_for_llm(user_llm_query, retrieved_content, False, [])
                response_search = generate_response_ollama('llama3.2', prompt)
                progress.progress(100)
                
                # Display results
                st.write("### Text Analysis Response:")
                st.write(response_search)
                
                # Show details in expanders
                with st.expander("Search Results"):
                    st.write(search_results)
                with st.expander("LLM Prompt"):
                    st.write(prompt)
                    
            elif search_type == 'Image':
                # Image-only analysis
                status.text("Fetching and analyzing images...")
                search_images = scrape_images_from_internet(user_search_query, number_images)
                progress.progress(60)
                
                if search_images:
                    status.text("Generating image analysis...")
                    prompt_images = create_prompt_for_llm(user_llm_query, retrieved_content, True, search_images, True)
                    response_images = generate_response_ollama('llava', prompt_images)
                    progress.progress(100)
                    
                    # Display results
                    st.write("### Image Analysis Response:")
                    st.write(response_images)
                    
                    st.write("### Analyzed Images:")
                    for i, image in enumerate(search_images):
                        st.image(image, caption=f"Image {i + 1}")
                    
                    # Show details in expander
                    with st.expander("LLM Prompt for Images"):
                        st.write(prompt_images)
                else:
                    st.error("No images found for the given query.")
                    
            else:  # Both
                # Combined text and image analysis
                status.text("Fetching and analyzing images...")
                search_images = scrape_images_from_internet(user_search_query, number_images)
                progress.progress(60)
                
                status.text("Generating combined analysis...")
                # Text analysis
                prompt = create_prompt_for_llm(user_llm_query, retrieved_content, False, [])
                response_search = generate_response_ollama('llama3.2', prompt)
                progress.progress(75)
                
                if search_images:
                    # Image analysis
                    prompt_2 = create_prompt_for_llm(user_llm_query, retrieved_content, True, search_images, True)
                    response_images = generate_response_ollama('llava', prompt_2)
                    
                    # Combined analysis
                    prompt_3 = create_prompt_for_llm(user_llm_query, retrieved_content, True, search_images)
                    response_search_and_images = generate_response_ollama('llava', prompt_3)
                    progress.progress(100)
                    
                    # Display all results
                    st.write("### Text Analysis:")
                    st.write(response_search)
                    
                    st.write("### Image Analysis:")
                    st.write(response_images)
                    
                    st.write("### Combined Analysis:")
                    st.write(response_search_and_images)
                    
                    st.write("### Analyzed Images:")
                    for i, image in enumerate(search_images):
                        st.image(image, caption=f"Image {i + 1}")
                    
                    # Show details in expanders
                    with st.expander("Search Results"):
                        st.write(search_results)
                    with st.expander("Text Analysis Prompt"):
                        st.write(prompt)
                    with st.expander("Image Analysis Prompt"):
                        st.write(prompt_2)
                    with st.expander("Combined Analysis Prompt"):
                        st.write(prompt_3)
                else:
                    st.error("No images found for the given query. Showing text analysis only.")
                    st.write("### Text Analysis:")
                    st.write(response_search)
            
            status.empty()  # Clear status message
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")