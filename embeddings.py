from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import chromadb
from better_profanity import profanity


llm = Ollama(model="llama3.2")
profanity.load_censor_words()

# Load the sentence-transformers embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Create a persistent vector store
short_term_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
long_term_memory = Chroma(persist_directory="./chroma_memory_db", embedding_function=embedding_model)

# Define a function to process and store documents
def process_and_store_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(documents)

    vectorstore.add_documents(docs)

# Function to retrieve documents using a query
def retrieve_documents(query, k=5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(query)
    results = " ".join([doc.page_content for doc in retrieved_docs])
    return results

# Function to store conversation in long-term memory
def store_conversation(user_input, bot_response):
    conversation_text = f"User: {user_input}\nBot: {bot_response}"
    long_term_memory.add_texts([conversation_text])

# Function to retrieve past conversations
def retrieve_past_conversation(query, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    past_conversations = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in past_conversations])

def find_similar_queries(user_query, k=3):
    query_embedding = embedding_model.embed_query(user_query)
    
    results = vectorstore.similarity_search_with_score(user_query, k=k)
    similar_queries = [(doc.page_content, score) for doc, score in results if score > 0.7]  # Set threshold

    return similar_queries


def retrieve_chat_context(user_query):
    similar_queries = find_similar_queries(user_query)

    if similar_queries:
        context = "\n".join([f"Past Query: {q}\nPast Answer: {a}" for q, a in similar_queries])
        return f"Previous conversations:\n{context}\n\nNew Query: {user_query}"
    else:
        return user_query


def check_swear_words(text):
    return profanity.contains_profanity(text)

def censor_swear_words(text):
    return profanity.censor(text)

def clear_cache(full_reset=False):
    try:
        if full_reset:
            # Delete the entire collection
            chroma_client.delete_collection("internet_documents")
        else:
            # Retrieve the collection and delete all documents
            collection = chroma_client.get_collection("internet_documents")
            all_ids = [doc["id"] for doc in collection.get()["documents"]]  # Extract document IDs
            
            if all_ids:
                collection.delete(ids=all_ids)  # Delete all documents using IDs
            
        return None
    except Exception as e:
        return str(e)

tools = [
    Tool(
        name="Document Search",
        func=lambda query: retrieve_documents(query),
        description="Retrieve relevant documents based on a query"
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=short_term_memory,
    verbose=True
)

def ask_agent(query):
    # Retrieve long-term memory context
    past_conversations = retrieve_past_conversations(query, k=3)
    past_context = "\n".join(past_conversations) if past_conversations else "No relevant past memory."

    # Combine past memory with user query
    full_query = f"Previous Context:\n{past_context}\n\nUser Query:\n{query}"
    
    # Get agent response
    response = agent.run(full_query)

    # Store the conversation in both memory types
    short_term_memory.save_context({"input": query}, {"output": response})
    store_conversation(query, response)

    return response