import json
import os
import difflib

CHAT_HISTORY_FILE = "chat_history.json"

# Function to load chat history from file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return []

# Function to save chat history to file
def save_chat_history(chat_history):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(chat_history, file, indent=4, ensure_ascii=False)
        
def clear_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump([], file, indent=4, ensure_ascii=False)  # Overwrite with an empty list
        return None
    except Exception as e:
        return str(e)
    
def find_most_similar_query(user_llm_query, user_search_query, user_type_query, llm_threshold=0.85, search_threshold=0.75):
    chat_history = load_chat_history()
    
    if not chat_history:
        return None

    best_match = None
    best_query_similarity = 0
    best_search_similarity = 0

    for entry in chat_history:
        query = entry.get("query", "")
        response = entry.get("response", "")
        search = entry.get("search", "")
        type_query = entry.get("type", "")
        images = entry.get("images", "")
        
        if user_type_query.lower() == type_query.lower():
            # Calculate similarity for both query and search separately
            query_similarity = difflib.SequenceMatcher(None, user_llm_query, query).ratio()
            search_similarity = difflib.SequenceMatcher(None, user_search_query, search).ratio()

            # Only update best match if either similarity is above the threshold
            if query_similarity >= llm_threshold and search_similarity >= search_threshold:
                if query_similarity > best_query_similarity and search_similarity > best_search_similarity:
                    best_query_similarity = query_similarity
                    best_search_similarity = search_similarity
                    best_match = {
                        "query": query,
                        "search": search,
                        "response": response,
                        "type": type_query,
                        "images": images,
                        "query_similarity": best_query_similarity,
                        "search_similarity": best_search_similarity
                    }

    return best_match