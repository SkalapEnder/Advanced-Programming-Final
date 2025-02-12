import ollama

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