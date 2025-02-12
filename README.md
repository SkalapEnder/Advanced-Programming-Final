
# Advanced Programming Final Project | IT-2308

### Team members

1) #### Alisher Berik
2) #### Alibi Ospan
3) #### Yeldos Sanakov


## Installation
Install special AI models in command prompt
```
ollama run llama3.2
ollama run llama3.2:latest
ollama run llava
ollama run phi3:medium
ollama run deepseek-r1
```

Execute this command to install necessary packages
```
pip install langchain chromadb better-profanity ollama requests beautifulsoup4 googlesearch-python streamlit
```

Then, open project folder and write next command in terminal:
```
python -m streamlit run app.py
```
OR
```
streamlit run app.py
```


## Instruction

1) Choose AI model that analyze text (Llama 3.2, Phi3 or DeepSeek-R1)

2) Write search query like to browser. Try not to write complex search queries because search system could misunderstand your query

3) Write LLM query. This exactly question to Ollama's model. There give more information or details about your goal.

4) Choose type of searching

Text - retrieve only text from internet and analyze it Text model

Image - retrieve only images from internet. Llava model will analyze them.

Both - return text analyze and combined result (text and image)

P.S.: Please, don't write inappropriate words in all queries.

