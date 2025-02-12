import streamlit as st
from search import search_query_internet, scrape_images_from_internet
from embeddings import process_and_store_documents, retrieve_documents, ask_agent, clear_cache, store_conversation, retrieve_past_conversations, short_term_memory
from embeddings import check_swear_words, censor_swear_words
from rag import create_prompt_for_llm, generate_response_ollama
from config import LLM_MODEL_TEXT, LLM_MODEL_IMAGE

st.title("RAG Chatbot with Internet Search with Memory 🧠")

with st.form("prompt_form"):
    st.subheader("0. Choose model to analyze")
    option = st.selectbox(
        ":robot_face: Choose Ollama model:",
        ("Llama 3.2", "Phi3 Medium", "DeepSeek-R1 (7B)")
    )
    
    st.subheader("1. Enter Your Search Query")
    user_search_query = st.text_area("Search query:", "")

    st.subheader("2. Enter Your Question")
    user_llm_query = st.text_area("LLM prompt:", "")

    st.subheader("3. Choose Search Options")
    search_type = st.radio("Select search type:", ('Text', 'Image', 'Both'))
    number_images = st.number_input("Number of images to analyze:", min_value=3, max_value=10, value=3)

    submitted = st.form_submit_button("Send")
    
    clear_docs = st.form_submit_button("🗑️ Clear Cache")

    full_reset = st.form_submit_button("⚠️ Full Reset (Delete Collection)")

    if clear_docs:
        res = clear_cache()
        if res == None: st.success("✅ ChromaDB cache cleared (documents deleted)!")
        else: st.error(f"⚠️ Error clearing cache: {res}")

    if full_reset:
        res = clear_cache(full_reset=True)
        if res == None: st.success("✅ ChromaDB collection deleted successfully!")
        else: st.error(f"⚠️ Error clearing cache: {res}")
    
    if submitted:
        match option:
            case "Llama 3.2":
                LLM_MODEL_TEXT = "llama3.2:latest"
            case "Phi3 Medium":
                LLM_MODEL_TEXT = "phi3:medium"
            case "DeepSeek-R1 (7B)":
                LLM_MODEL_TEXT = "deepseek-r1:7b"
        
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
            past_conversations = retrieve_past_conversations(user_llm_query, k=3)
            past_context = "".join(past_conversations) if past_conversations else None
            
            # Searching part
            status.text("Searching content...")
            search_results = search_query_internet(user_search_query)
            progress.progress(20)
            
            # Process part
            status.text("Processing content...")
            progress.progress(30)
            
            process_and_store_documents(search_results)  # Step 1: Input Understanding	
            retrieved_docs = retrieve_documents(user_llm_query, k=5) # Step 2: Information Retrieval    
            retrieved_content = " ".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
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

            else:
                search_images = scrape_images_from_internet(user_search_query, number_images)
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

            status.empty()
            store_conversation(user_llm_query, response)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    