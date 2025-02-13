# telegram_bot.py
import requests
import json

TELEGRAM_BOT_TOKEN = "7712984753:AAESoDITk4NUYzh52vx_-e5-gj13uaMMSbU"

def store_chat_id(chat_id):
    """Stores chat_id in a file if it is not already saved."""
    chat_id = str(chat_id)
    try:
        with open("chat_ids.txt", "r") as f:
            saved_ids = set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        saved_ids = set()
    
    if chat_id not in saved_ids:
        with open("chat_ids.txt", "a") as f:
            f.write(chat_id + "\n")
        print(f"New chat_id saved: {chat_id}")
    else:
        print(f"chat_id {chat_id} is already saved.")

def load_chat_ids():
    """Loads the list of chat IDs from the file."""
    try:
        with open("chat_ids.txt", "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

def check_for_new_chat_ids():
    """Fetches updates from the bot (via getUpdates) and saves chat IDs of users who sent /start."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching updates from Telegram.")
        return
    data = response.json()
    new_ids = 0
    for update in data.get("result", []):
        message = update.get("message")
        if message:
            text = message.get("text", "")
            if text.strip() == "/start":
                chat = message.get("chat", {})
                chat_id = chat.get("id")
                if chat_id:
                    store_chat_id(chat_id)
                    new_ids += 1
    print(f"Updated, found {new_ids} new chat_id(s).")

def send_telegram_notification(search_query: str, llm_query: str, search_type: str):
    """Sends a formatted Telegram notification to all stored chat IDs."""
    chat_ids = load_chat_ids()
    if not chat_ids:
        print("No saved chat IDs. Make sure users have sent /start to the bot.")
        return
    
    message = (
        f"🔔 **New Notification** 🔔\n\n"
        f"🔍 **Search Query**: {search_query}\n"
        f"❓ **Question**: {llm_query}\n"
        f"📌 **Type**: {search_type}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for chat_id in chat_ids:
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"Error sending notification to {chat_id}:", response.text)
        else:
            print(f"Notification sent to {chat_id}.")

def check_and_callback(query: str, search_query: str, search_type: str):
    """Triggers a notification for every query (removes keyword filtering)."""
    send_telegram_notification(search_query, query, search_type)

