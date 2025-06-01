"""
Medical Assistant RAG System - Main Streamlit Application
"""

import streamlit as st
import os
import tempfile
from datetime import datetime
from src.components.data_processor import DataProcessor
from src.components.vector_store import VectorStore
from src.core.rag_pipeline import RAGPipeline
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

# Page configuration
st.set_page_config(
    page_title="Medical Assistant RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .confidence-score {
        background-color: #e8f4fd;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'cohere_api_key' not in st.session_state:
        st.session_state.cohere_api_key = None
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DataProcessor()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def api_key_form():
    """Form UI to input API key in the center of the screen"""

    
    st.markdown(
        """
        <h2 style='text-align: center;'>🔐 Welcome to the Medical Assistant RAG System</h2>
        <p style='text-align: center;'>Please enter your <strong>Cohere API Key</strong> to get started.</p>
        <p style='text-align: center;'>
            Don't have one? <a href='https://docs.aicontentlabs.com/articles/cohere-api-key/' target='_blank'>Get your API key here</a>.
        </p>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])  # Center content in col2
    with col2:
        with st.form(key="api_key_form", clear_on_submit=False):
            user_key = st.text_input("COHERE_API_KEY", type="password", placeholder="Paste your Cohere API Key here")

            submit_col1, submit_col2, submit_col3, submit_col4, submit_col5= st.columns([1,1,1,1,1])
            with submit_col3:
                submitted = st.form_submit_button("🔑 Submit",type="primary")

            if submitted:
                if not user_key.strip():
                    st.warning("⚠️ Please enter a valid API key!")
                else:
                    st.session_state.cohere_api_key = user_key.strip()
                    st.success("✅ API key saved successfully!")
                    st.rerun()



def process_uploaded_files(uploaded_files):
    """Process uploaded files and create vector store"""
    try:
        all_chunks = []
        
        with st.spinner("Processing uploaded documents..."):
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                # Process document
                chunks = st.session_state.doc_processor.process_document(tmp_path, uploaded_file.name)
                all_chunks.extend(chunks)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                st.success(f"✅ Processed {uploaded_file.name}: {len(chunks)} chunks")
        
        # Create vector store
        if all_chunks:
            with st.spinner("Creating vector embeddings..."):
                st.session_state.vector_store.create(all_chunks)
                st.session_state.documents_processed = True
                
            st.success(f"🎯 Vector store created with {len(all_chunks)} total chunks!")
            logger.info(f"Processed {len(uploaded_files)} files with {len(all_chunks)} total chunks")
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        logger.error(f"Document processing error: {str(e)}")

def display_chat_message(role, content, sources=None):
    """Display a chat message with optional sources"""
    with st.chat_message(role):
        st.write(content)
        
        if sources and role == "assistant":
            with st.expander("📚 View Sources", expanded=False):
                for i, (chunk, score) in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 0.5rem;">
                            <strong>Source {i}</strong>
                            <span class="confidence-score">Relevance: {score:.3f}</span>
                        </div>
                        <p><strong>File:</strong> {chunk.metadata.get('source', 'Unknown')}</p>
                        <p><strong>Content:</strong> {chunk.page_content[:300]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    initialize_session_state()

    # Ask for API key first
    if not st.session_state.cohere_api_key:
        api_key_form()
        st.stop()

    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Assistant RAG System</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about medical policies, guidelines, and procedures based on uploaded documents.")

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        uploaded_files = st.file_uploader("Upload medical documents", type=['pdf', 'txt'], accept_multiple_files=True)
        if uploaded_files and st.button("Process Documents", type="primary"):
            process_uploaded_files(uploaded_files)

        st.header("📊 System Status")
        if st.session_state.documents_processed:
            st.success("✅ Documents processed")
            st.info(f"Vector store ready with {st.session_state.vector_store.get_document_count()} chunks")
        else:
            st.warning("⚠️ No documents processed")

        st.header("⚙️ RAG Settings")
        k_value = st.slider("Number of retrieved chunks", 1, 10, 3)
        temperature = st.slider("Response creativity", 0.0, 1.0, 0.1, 0.1)

        if hasattr(st.session_state.rag_pipeline, 'update_settings'):
            st.session_state.rag_pipeline.update_settings(k=k_value, temperature=temperature)

    # Main chat interface
    st.header("💬 Ask Your Medical Question")
    for message in st.session_state.chat_history:
        display_chat_message(message['role'], message['content'], message.get('sources'))

    if prompt := st.chat_input("Enter your medical question here..."):
        if not st.session_state.documents_processed:
            st.warning("⚠️ Please upload and process documents first!")
            return

        display_chat_message("user", prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                try:
                    response_data = st.session_state.rag_pipeline.generate_response(
                        query=prompt,
                        vectorstore=st.session_state.vector_store,
                        temperature=temperature
                    )
                    st.write(response_data['answer'])
                    if response_data['sources']:
                        with st.expander("📚 View Sources", expanded=False):
                            for i, (chunk, score) in enumerate(response_data['sources'], 1):
                                st.markdown(f"""<div class="source-box">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                        <strong>Source {i}</strong>
                                        <span class="confidence-score">Relevance: {score:.3f}</span>
                                    </div>
                                    <p><strong>File:</strong> {chunk.metadata.get('source', 'Unknown')}</p>
                                    <p><strong>Content:</strong> {chunk.page_content[:300]}...</p>
                                </div>""", unsafe_allow_html=True)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_data['answer'],
                        "sources": response_data['sources']
                    })

                    logger.info(f"Query: {prompt}")
                    logger.info(f"Retrieved {len(response_data['sources'])} sources")
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    logger.error(f"Error generating response: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_message
                    })
    
    # Footer with logs
    # with st.expander("🔍 System Logs", expanded=False):
    #     st.subheader("Recent Activity")
    #     try:
    #         with open("logs/rag_system.log", "r") as f:
    #             logs = f.readlines()
    #             # Show last 20 log entries
    #             recent_logs = logs[-20:] if len(logs) > 20 else logs
    #             for log in recent_logs:
    #                 st.text(log.strip())
    #     except FileNotFoundError:
    #         st.info("No logs available yet.")
    #     except Exception as e:
    #         st.error(f"Error reading logs: {str(e)}")
    
    # # Sample questions
    # st.header("💡 Sample Questions")
    # col1, col2 = st.columns(2)
    
    # sample_questions = [
    #     "What are the clinic's operating hours?",
    #     "What should I do if I have a fever?",
    #     "What are the vaccination requirements?",
    #     "How do I schedule an appointment?",
    #     "What is the policy for prescription refills?",
    #     "What should I bring to my first appointment?"
    # ]
    
    # for i, question in enumerate(sample_questions):
    #     col = col1 if i % 2 == 0 else col2
    #     with col:
    #         if st.button(question, key=f"sample_{i}"):
    #             st.session_state.sample_question = question
    #             st.rerun()

if __name__ == "__main__":
    main()