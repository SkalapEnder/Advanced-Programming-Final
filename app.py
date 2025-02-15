import streamlit as st
from datetime import datetime
from search import *
from embeddings import *
from rag import *
from config import LLM_MODEL_TEXT, LLM_MODEL_IMAGE
from telegram_bot import check_for_new_chat_ids, check_and_callback
from history import *

st.title("RAG Chatbot with Internet Search")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

st.sidebar.header("Telegram Bot Administration")
if st.sidebar.button("Update chat IDs (Check for /start)"):
    check_for_new_chat_ids()
    st.sidebar.write("Chat IDs Updated")

with st.form("prompt_form"):
    st.subheader(":robot_face: Choose model to analyze")
    option = st.selectbox(
        " Choose Ollama model:",
        ("Llama 3.2", "Phi3 Medium", "DeepSeek-R1 (7B)")
    )
    
    st.subheader("🔍 Enter Your Search Query")
    user_search_query = st.text_area("Search query:", "")

    st.subheader("❓ Enter Your Question")
    user_llm_query = st.text_area("LLM prompt:", "")

    st.subheader("⚙️ Choose Search Options")
    search_type = st.radio("📃 Select search type:", ('Text', 'Image', 'Both'))
    number_images = st.number_input("🖼️ Number of images to analyze:", min_value=3, max_value=10, value=3)
    
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        submitted = st.form_submit_button("➡️ Send")
    with col2:
        clear_docs = st.form_submit_button("🗑️ Clear Cache (Short-Term)")
    
    with col3:
        clear_history = st.form_submit_button("🔙 Clear Chat History (Not Memory)")
    with col4:
        full_reset = st.form_submit_button("⚠️ Full Reset (Long-Term)")

    if clear_docs:
        res = clear_cache()
        if res == None: st.success("✅ ChromaDB cache cleared (documents deleted)!")
        else: st.error(f"⚠️ Error clearing cache: {res}")

    if full_reset:
        res = clear_cache(full_reset=True)
        if res == None: st.success("✅ ChromaDB collection deleted successfully!")
        else: st.error(f"⚠️ Error clearing cache: {res}")
    
    if clear_history:
        st.session_state.chat_history = []
        res = clear_chat_history()
        if res == None: st.success("✅ Chat history was deleted!")
        else: st.error(f"⚠️ Error clearing chat history: {res}")
    
    if submitted:
        if not is_ollama_running(): 
            st.error("❌ Ollama is NOT running! Start it with: `ollama serve`")
            st.stop()
            
        if not is_ollama_model_here(LLM_MODEL_IMAGE):
            lists = ["llava", "llava:latest"]
            j = 0
            for i in range(len(lists)):
                if is_ollama_model_here(LLM_MODEL_IMAGE):
                    LLM_MODEL_IMAGE = lists[i]
                    break
                j += 1
            if j == len(lists):
                st.error(f"❌ Ollama model is NOT installed! Write command on command-line console: `ollama run {LLM_MODEL_TEXT}`")
                st.stop()
        
        match option:
            case "Llama 3.2":
                LLM_MODEL_TEXT = "llama3.2:latest"
            case "Phi3 Medium":
                LLM_MODEL_TEXT = "phi3:medium"
            case "DeepSeek-R1 (7B)":
                LLM_MODEL_TEXT = "deepseek-r1:latest"
                
        if not is_ollama_model_here(LLM_MODEL_TEXT):
            st.error(f"❌ Ollama model is NOT installed! Write command on command-line console: `ollama run {LLM_MODEL_TEXT}`")
            st.stop()
        
        similar_query = find_most_similar_query(user_llm_query, user_search_query, search_type)
        
        temp_images = []
        
        if similar_query == None:    
            try:
                # Start point
                progress = st.progress(0)
                status = st.empty()
                
                status.text("Checking for inappropriate words...")
                if check_swear_words(user_llm_query):
                    user_llm_query = censor_swear_words(user_llm_query)
                    st.warning("⚠️ Your message contained inappropriate words. It has been censored.")
                    
                # Retrieve long-term memory
                status.text("Retrieve past conversation...")
                progress.progress(10)
                past_context = retrieve_past_conversation(user_llm_query, k=3)
                
                # Searching part
                status.text("Searching content...")
                progress.progress(20)
                search_results = search_query_internet(user_search_query)
                
                # Process part
                status.text("Processing content...")
                progress.progress(30)
                
                process_and_store_documents(search_results)  # Step 1: Input Understanding	
                retrieved_content = retrieve_documents(user_llm_query, k=5) # Step 2: Information Retrieval    
                progress.progress(40)

                # Analyzing part
                if search_type == 'Text':
                    status.text("Generating text analysis...")
                    prompt = create_prompt_for_llm(user_llm_query, past_context, retrieved_content, False, [])
                    response_search = generate_response_ollama(LLM_MODEL_TEXT, prompt)
                    response = response_search
                    progress.progress(100)
                    st.write("### Text Analysis Response")
                    st.write(response_search)
                    
                    
                    with st.expander("Retrieved Content"):
                        st.write(retrieved_content)
                        
                    with st.expander("User's query (after checking)"):
                        st.write(user_llm_query)

                elif search_type == 'Image':
                    status.text("Fetching images...")
                    search_images = scrape_images_from_internet(user_search_query, number_images)
                    temp_images = search_images
                    progress.progress(60)
                    
                    status.text("Analyzing images...")
                    prompt_images = create_prompt_for_llm(user_llm_query, past_context, retrieved_content, True, search_images, True)
                    response_images = generate_response_ollama(LLM_MODEL_IMAGE, prompt_images)
                    response = response_images
                    progress.progress(100)
                    
                    st.write("### Image Analysis Response:")
                    st.write(response_images)
                    
                    for img in search_images:
                        st.image(img)
                        
                    with st.expander("User's query (after checking)"):
                        st.write(user_llm_query)

                else:
                    search_images = scrape_images_from_internet(user_search_query, number_images)
                    temp_images = search_images
                    progress.progress(60)
                    
                    prompt_text = create_prompt_for_llm(user_llm_query, past_context, retrieved_content, False, [])
                    response_text = generate_response_ollama(LLM_MODEL_TEXT, prompt_text)
                    
                    prompt_combined = create_prompt_for_llm(user_llm_query, past_context, retrieved_content, True, search_images)
                    response_combined = generate_response_ollama(LLM_MODEL_IMAGE, prompt_combined)
                    response = response_combined
                    progress.progress(100)
                    
                    st.write(f"### Text Analysis ({LLM_MODEL_TEXT}):")
                    st.write(response_text)
                    st.write(f"### Combined Analysis ({LLM_MODEL_IMAGE}):")
                    st.write(response_combined)
                    for img in search_images:
                        st.image(img)
                        
                    with st.expander("Retrieved Content"):
                        st.write(retrieved_content)
                        
                    with st.expander("User's query (after checking)"):
                        st.write(user_llm_query)

                status.empty()
                
                # 🔹 Send Telegram Notification
                check_and_callback(user_llm_query, user_search_query, search_type)
                
                # Store in session state for chat history
                currdate = datetime.today().strftime("%H:%M:%S, %d %B %Y")

                # Append user query with timestamp
                st.session_state.chat_history.append({
                    "search": user_search_query,
                    "query": user_llm_query,
                    "response": response,
                    "type": search_type,
                    "images": temp_images,
                    "timestamp": currdate
                })

                # Save chat history to file
                save_chat_history(st.session_state.chat_history)
                store_conversation(user_llm_query, response)

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.write("### Based on previous similar queries")
            st.write(f"#### Search Query: {similar_query['search']}")
            st.write(f"#### LLM Query: {similar_query['query']}")
            st.write(f"#### Type: {similar_query['type']}")
            st.write(similar_query['response'])
            if len(similar_query['images']) > 0:
                for image in similar_query['images']:
                    st.image(image)

st.subheader("Chat History")
if len(st.session_state.chat_history) > 0:
    for index, entry in enumerate(st.session_state.chat_history):
        if not isinstance(entry, dict) or "query" not in entry or "response" not in entry or "search" not in entry:
            continue

        if index % 2 != 0:
            st.divider()
        st.write(f"**Date**: {entry['timestamp']}")
        st.write(f"**User (Search)**: {entry['search']}")
        st.write(f"**User (LLM)**: {entry['query']}")
        st.write(f"**Type of query**: {entry['type']}")
        st.write(f"**Bot**: {entry['response']}")
else:
    st.write("There is no chat history!")