import requests
from bs4 import BeautifulSoup
from googlesearch import search
from config import API_KEY, CSE_ID

def search_query_internet(query):
    urls = [url for url in search(query, num_results=5)]
    return fetch_web_content(urls)

def fetch_web_content(urls):
    headers = {"User-Agent": "Mozilla/5.0"}
    documents = []

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = " ".join([p.get_text(strip=True) for p in soup.find_all('p')])
            content = paragraphs + '\n\n'
            headers1 = " ".join([h.get_text(strip=True) for h in soup.find_all("h1")])
            content += headers1 + '\n\n'
            headers2 = " ".join([h.get_text(strip=True) for h in soup.find_all("h2")])
            content += headers2 + '\n\n'
            headers3 = " ".join([h.get_text(strip=True) for h in soup.find_all("h3")])
            content += headers3 + '\n\n'
            headers4 = " ".join([h.get_text(strip=True) for h in soup.find_all("h4")])
            content += headers4 + '\n\n'
            documents.append(content)
            
        except requests.RequestException:
            continue

    return documents

def google_search(query, search_type='text'):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': API_KEY, 'cx': CSE_ID, 'searchType': search_type, 'q': query}
    return requests.get(url, params=params).json()

def scrape_images_from_internet(query, num_images):
    results = google_search(query, "image")
    return [item['link'] for item in results.get('items', [])[:num_images]]