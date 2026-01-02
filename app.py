import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.messages import HumanMessage

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Nexus AI | 2026 Tech Advisor",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# Custom CSS for a professional tech aesthetic
st.markdown("""
<style>
    .tech-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        letter-spacing: -1px;
    }
    .stChatInput {
        padding-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Load Models (Cached for Performance) ---
@st.cache_resource(show_spinner=False)
def load_models():
    """
    Initializes and caches the Embedding model and LLM.
    """
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        st.error("System Error: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
        st.stop()

    embeddings = HuggingFaceEndpointEmbeddings(
        repo_id='sentence-transformers/all-MiniLM-L6-v2',
        task='feature-extraction',
    )

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        temperature=0.2, # Lower temperature for factual accuracy
        max_new_tokens=512
    )
    
    chat_model = ChatHuggingFace(llm=llm)
    return embeddings, chat_model

# --- 2. Initialize Knowledge Base (Cached) ---
@st.cache_resource(show_spinner=False)
def initialize_knowledge_base(_embeddings):
    """
    Loads the 2026 future tech dataset and creates vector embeddings.
    """
    knowledge_base = [
        # --- AI & GenAI Trends ---
        "In 2026, Generative AI has evolved into 'Autonomous AI Agents' that execute complex tasks like coding apps and managing finances without human intervention.",
        "Small Language Models (SLMs) running locally on devices (Edge AI) are preferred over cloud models due to privacy and latency. Phones now use built-in NPUs.",
        "GraphRAG (Retrieval Augmented Generation with Knowledge Graphs) has replaced standard RAG, improving answer accuracy by 40%.",
        "Multimodal AI is the standard, processing audio, video, code, and text simultaneously in real-time.",
        # --- Job Roles ---
        "The role of 'Prompt Engineer' has evolved into 'AI Model Orchestrator', focusing on designing workflows for multi-agent systems.",
        "AI Ethics Compliance Officers are in high demand to ensure models adhere to global regulations on bias and copyright.",
        "Full-Stack Developers have transitioned to 'AI-Stack Developers', integrating Vector DBs, LLM orchestration, and AI UI components.",
        "Quantum Algorithm Researchers are emerging as Quantum Computing moves from labs to cloud APIs by IBM and Google.",
        "Cybersecurity Analysts now focus on 'Post-Quantum Cryptography' to protect against quantum decryption.",
        # --- Tech Stack ---
        "Python remains the dominant language for AI, while Mojo and Rust are gaining traction for high-performance infrastructure.",
        "Frontend development is dominated by TypeScript and AI-generated UI components (v0.dev style tools).",
        "Vector Databases (like Pinecone, Weaviate, Milvus) are now fundamental components of modern software architecture.",
        # --- Skills & Startup Ecosystem ---
        "System Design for AI is the most valued skill for freshers, surpassing rote coding syntax.",
        "Open Source contributions to models on Hugging Face are highly valued in recruitment.",
        "The 'One-Person Unicorn' phenomenon involves solo founders building billion-dollar companies using fleets of AI agents.",
        "SaaS has shifted to 'Service-as-a-Software', selling outcomes (e.g., guaranteed leads) rather than just tools.",
        "EdTech focuses on Hyper-Personalized Learning using real-time emotional and pacing adaptation."
    ]
    
    doc_vectors = _embeddings.embed_documents(knowledge_base)
    return knowledge_base, doc_vectors

# --- 3. Core RAG Logic ---
def get_rag_response(query, embeddings, chat_model, knowledge_base, doc_vectors):
    """
    Retrieves relevant context and generates a response.
    """
    query_vector = embeddings.embed_query(query)
    scores = cosine_similarity([query_vector], doc_vectors)[0]
    best_index = np.argmax(scores)
    best_doc = knowledge_base[best_index]
    similarity_score = scores[best_index]

    # Threshold for relevance
    if similarity_score < 0.3:
        return "Accessing Database... Result: Insufficient data points to formulate a confident answer on this specific topic regarding 2026."

    prompt = f"""
    You are Nexus, an advanced AI Tech and Career strategic advisor for the year 2026. 
    Provide a professional, concise, and highly technical answer based STRICTLY on the provided context.
    Do not add fluff or conversational filler.

    Context Data: {best_doc}

    User Query: {query}
    """
    messages = [HumanMessage(content=prompt)]
    response = chat_model.invoke(messages)
    return response.content

# --- 4. Main UI Layout ---
def main():
    # --- Sidebar (System Metrics) ---
    with st.sidebar:
        st.markdown("## ⚙️ System Intelligence")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Model Engine", value="Mistral-7B")
        with col2:
             st.metric(label="Knowledge Base", value="Year 2026")
        
        st.markdown("---")
        with st.expander("💡 Architecture Protocol", expanded=False):
            st.markdown("""
            **Retrieval Augmented Generation (RAG):**
            1.  **Vectorization:** Query conversion to high-dimensional vectors.
            2.  **Semantic Search:** Cosine similarity matching against encrypted knowledge vectors.
            3.  **Synthesis:** Context injection for accurate LLM generation.
            """)
        
        st.markdown("---")
        st.caption("System Status: Online | Latency: Nominal")

    # --- Main Chat Interface ---
    st.markdown('<h1 class="tech-header">Nexus AI // 2026 Tech Landscape Advisor</h1>', unsafe_allow_html=True)
    st.markdown("Generate insights on future Job Roles, Tech Stacks, and Autonomous Agents.")

    # Initialize System Resources
    with st.status("Initializing Neural Pathways...", expanded=True) as status:
        st.write("Loading Embedding Models...")
        embeddings, chat_model = load_models()
        st.write("Indexing 2026 Knowledge Vector Database...")
        knowledge_base, doc_vectors = initialize_knowledge_base(embeddings)
        status.update(label="System Online. Ready for Queries.", state="complete", expanded=False)

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Identity confirmed. I am Nexus AI. My database contains projections for the 2026 technology landscape. What strategic insight do you require today?"}
        ]

    # Render Chat History
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "🗳️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Input Handling
    if prompt := st.chat_input("Execute query regarding 2026..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🗳️"):
            with st.spinner("Processing Query Vectors..."):
                response = get_rag_response(
                    prompt, embeddings, chat_model, knowledge_base, doc_vectors
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()