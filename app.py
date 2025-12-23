import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq   # Groq integration

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="SuperAI Document (Groq)",
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Main background */
    .stApp, .main {
        background: #0d1117;
        color: #f5f5f5;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif;
        color: #f5f5f5;
    }
    
    p, div, input, label, .stTextInput, .stFileUploader, .stButton {
        font-family: 'Inter', sans-serif;
        color: #c9d1d9;
    }
    
    /* Glass containers */
    .css-1d391kg, .css-1y4p8pa {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #00f2fe 0%, #8a2be2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #8a2be2 0%, #00f2fe 100%);
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(0, 242, 254, 0.3);
    }
    
    /* Inputs */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #f5f5f5;
        border-radius: 10px;
        padding: 12px;
    }
    
    /* Uploaded file preview */
    .uploadedFile {
        background-color: rgba(255, 255, 255, 0.04);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #00f2fe;
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* AI response */
    .reply-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 14px;
        margin: 20px 0;
        border-left: 4px solid #8a2be2;
        box-shadow: 0 6px 18px rgba(0,0,0,0.5);
    }
    
    /* Header */
    .header-container {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(138, 43, 226, 0.1) 100%);
        border-radius: 12px;
        margin-bottom: 30px;
    }
    
    .logo {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f2fe 0%, #8a2be2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    /* Divider */
    .luxury-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(0, 242, 254, 0.5) 50%, transparent 100%);
        margin: 25px 0;
    }
    
    /* Steps */
    .step-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Powered by */
    .powered-by {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #c9d1d9;
    }
    
    .groq-badge {
        background: linear-gradient(135deg, #00f2fe 0%, #8a2be2 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>

""", unsafe_allow_html=True)

# ---------------------------
# Helper Functions
# ---------------------------

def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF files"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Split text into smaller chunks for embeddings"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Convert text chunks into embeddings and save FAISS index"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(groq_api_key):
    """Build QA chain with Groq LLaMA"""
    prompt_template = """
    You are a helpful assistant. Use the provided context to answer the question clearly.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question: {question}

    Answer:
    """

    model = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.3-70b-versatile",   # ✅ Updated recommended model
        temperature=0.3,
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, groq_api_key):
    """Search FAISS DB and generate response with Groq"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(groq_api_key)

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.markdown(f"""
    <div class="reply-container">
        <h4>🤖 AI Response:</h4>
        <p>{response["output_text"]}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="logo">SuperAI Documents</div>
        <p style="font-size: 1.2rem; opacity: 0.9;">Get instant insights from your PDFs</p>
        <div class="powered-by">
            <span>Powered by</span>
            <span class="groq-badge">Groq</span>
            <span>+ LLaMA 3.3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How It Works
    st.markdown("### How It Works")
    st.markdown("""
    <div class="step-card">
        <h4>1. Enter Your API Key</h4>
        <p>Get a free API key from <a href="https://console.groq.com/keys" target="_blank" style="color: #8E2DE2; text-decoration: none;">console.groq.com</a>.</p>
    </div>
    <div class="step-card">
        <h4>2. Upload Your Documents</h4>
        <p>Upload one or more PDF files. The system will process and store embeddings.</p>
    </div>
    <div class="step-card">
        <h4>3. Ask a Question</h4>
        <p>Type any question related to your documents and get instant answers powered by Groq.</p>
    </div>
    """, unsafe_allow_html=True)

    # API key
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")

    # Divider
    st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

    # Question input
    user_question = st.text_input("🤖 Ask a question about your uploaded documents:", 
                                 placeholder="Type your question here...")

    if user_question and groq_api_key:
        user_input(user_question, groq_api_key)

    # Sidebar Upload
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2>📁 Document Upload</h2>
            <p>Upload and process your PDF files</p>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        
        if pdf_docs:
            st.markdown("**Uploaded Files:**")
            for pdf in pdf_docs:
                st.markdown(f'<div class="uploadedFile">{pdf.name}</div>', unsafe_allow_html=True)
        
        if st.button("Process Documents"):
            if not groq_api_key:
                st.warning("Please enter your Groq API Key first!")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing your documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.markdown('<div class="success-message">✅ Processing complete! You can now ask questions about your documents.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
