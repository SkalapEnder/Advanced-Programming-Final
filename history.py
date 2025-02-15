import json
import os

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
