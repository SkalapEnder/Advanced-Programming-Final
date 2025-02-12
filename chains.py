from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from better_profanity import profanity

# Define a custom swear word detection chain
class SwearWordCheckChain(LLMChain):
    def __init__(self):
        prompt_template = PromptTemplate.from_template(
            "Check if this text contains offensive language: {text}"
        )
        super().__init__(llm=None, prompt=prompt_template) 

    def check_swear_words(self, text):
        return profanity.contains_profanity(text)
