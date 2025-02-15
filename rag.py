import ollama
import requests

def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            return True
    except requests.ConnectionError:
        return False
    except requests.Timeout:
        return False
    
    return False

def is_ollama_model_here(model):
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(m.get("name") == model for m in models)
        else:
            print(f"Error fetching models: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return False
    

def create_prompt_for_llm(llm_query, past_context, search_results, with_images, images, only_images=False):
    if only_images:
        return [{'role': 'user', 'content': 'Describe these images:\n' + "\n".join(images)}]
    
    condition = ""
    
    if past_context != None: condition += f"\n\nUse previous context: {past_context}"
    condition += "\n\nGive answer using contexts and images:\n\n" if with_images else "Give answer using only context:\n\n"
    
    question = f"\n\nQuestion: {llm_query}\nAnswer: " 
    content = condition + search_results + question

    return [{ "role": "system", "content": (
                "You are an AI assistant that strictly answers questions based on the given context. "
                "Do NOT use any external knowledge."
            ) },
            {'role': 'user', 'content': content}]

def generate_response_ollama(model, prompt):
    completion = ollama.chat(model=model, messages=prompt)
    return completion['message']['content']