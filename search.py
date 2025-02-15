import requests
from bs4 import BeautifulSoup
from config import SERP_API_KEY

def search_query_internet(query, num_results=5):
    search_results = serpapi_search(query, num_results)
    return fetch_web_content(search_results)

def serpapi_search(query, num_results=5, search_type="text"):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERP_API_KEY
    }
    
    if search_type == "image":
        params["tbm"] = "isch"  # Image search mode

    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        if search_type == "image":
            return [img["original"] for img in results.get("images_results", [])[:num_results]]
        else:
            return [result["link"] for result in results.get("organic_results", [])]
    
    return []

def fetch_web_content(urls):
    """Fetch web content from URLs."""
    headers = {"User-Agent": "Mozilla/5.0"}
    documents = []

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = " ".join([p.get_text(strip=True) for p in soup.find_all('p')])
            headers1 = " ".join([h.get_text(strip=True) for h in soup.find_all("h1")])
            headers2 = " ".join([h.get_text(strip=True) for h in soup.find_all("h2")])
            headers3 = " ".join([h.get_text(strip=True) for h in soup.find_all("h3")])
            headers4 = " ".join([h.get_text(strip=True) for h in soup.find_all("h4")])
            
            content = f"{paragraphs}\n\n{headers1}\n\n{headers2}\n\n{headers3}\n\n{headers4}\n\n"
            documents.append(content)
            
        except requests.RequestException:
            continue

    return documents

def scrape_images_from_internet(query, num_images=5):
    return serpapi_search(query, num_images, search_type="image")
