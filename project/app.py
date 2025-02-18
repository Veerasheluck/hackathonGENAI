from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Cohere

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def crawl_and_process(website_url):
    try:
        response = requests.get(website_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        seo_data = {}

        for link in soup.find_all('a', href=True):
            page_url = link['href']
            if not page_url.startswith('http'):
                page_url = website_url + page_url

            page_response = requests.get(page_url)
            page_soup = BeautifulSoup(page_response.content, 'html.parser')
            meta_description = page_soup.find('meta', attrs={'name': 'description'})
            seo_data[page_url] = {
                'meta_description': meta_description['content'] if meta_description else None
            }
        
        return seo_data
    except requests.exceptions.RequestException as e:
        return None

def create_vector_database(seo_data):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [f"Page: {page}\nMeta Description: {data.get('meta_description', 'No description available')}" for page, data in seo_data.items()]
    return FAISS.from_texts(documents, embeddings)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    if user_input.startswith("http"):
        seo_data = crawl_and_process(user_input)
        if seo_data:
            db = create_vector_database(seo_data)
            cohere_api_key = "G0Z1wWTxM7WWk7z8javEA6Tg190dfcR5bTTkDV6Q"
            llm = Cohere(cohere_api_key=cohere_api_key)
            chatbot = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
            response = chatbot({"query": "Extract SEO insights"})
            return jsonify({"response": response['result']})
        return jsonify({"error": "Failed to retrieve SEO data"}), 500
    
    return jsonify({"response": "Invalid input"}), 400

if __name__ == '__main__':
    app.run(debug=True)
