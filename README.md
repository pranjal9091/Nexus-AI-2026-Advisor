# 🗳️ Nexus AI - 2026 Tech Landscape Advisor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

**Nexus AI** is an advanced **RAG (Retrieval Augmented Generation)** chatbot designed to act as a strategic tech consultant for the year 2026. 

By leveraging **Mistral-7B** and **Vector Embeddings**, it retrieves context-aware insights about future job roles, autonomous agents, and emerging technology stacks, ensuring responses are factual and hallucination-free.

---

## 🚀 Key Features

* **RAG Architecture:** Combines Retrieval (Semantic Search) with Generation (LLM) for high accuracy.
* **Vector Search:** Uses `Sentence-Transformers` and Cosine Similarity to find the most relevant knowledge chunks.
* **Professional UI:** Built with **Streamlit** for a clean, chat-based interface.
* **Performance Optimized:** Implements `@st.cache_resource` to load heavy models only once.
* **Secure:** Environment variables management for API security.

---

## 🛠️ Tech Stack

* **LLM Engine:** Mistral-7B-Instruct-v0.2 (via Hugging Face API)
* **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
* **Orchestration:** LangChain
* **Frontend:** Streamlit
* **Math/Search:** NumPy & Scikit-Learn (Cosine Similarity)

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Nexus-AI-2026-Advisor.git
cd Nexus-AI-2026-Advisor
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## 🧠 How It Works (Internal Logic)

1. **Ingestion:** The app loads a curated "2026 Tech Trends" dataset.
2. **Embedding:** Text is converted into numerical vectors using `all-MiniLM-L6-v2`.
3. **Retrieval:** When a user asks a question, the system performs a **Cosine Similarity** search to find the best matching document.
4. **Augmentation:** The retrieved document is attached to the user prompt as "Context".
5. **Generation:** The LLM (Mistral-7B) answers the question using *only* the provided context.

---

## 🤝 Contribution

Contributions are welcome! Please fork the repository and submit a pull request.

---

**Developed by Pranjal Singh**