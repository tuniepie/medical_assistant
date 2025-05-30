# Medical Assistant RAG System

A complete Retrieval-Augmented Generation (RAG) system that answers medical questions based on document corpus using [LangChain](https://python.langchain.com/docs/introduction), [FAISS](https://github.com/facebookresearch/faiss), and [Cohere](https://docs.cohere.com/cohere-documentation).

## 🚀 Live Demo

**Deployed URL**: [Your deployment URL will go here]

## 📋 Features

- **Document Ingestion**: Support for PDF and TXT files
- **Chunking**: text splitting with overlap
- **Vector Search**: FAISS-powered semantic retrieval
- **LLM Generation**: [Cohere LLM](https://docs.cohere.com/v2/docs/models) for accurate responses
- **Web Interface**: Clean Streamlit UI
- **Comprehensive Logging**: Query tracking and debug information
- **Source Attribution**: Shows retrieved document snippets

## 🛠️ Tech Stack

- **Backend**: Python, LangChain, FAISS, Cohere
- **LLM**: Cohere command-r-plus
- **Frontend**: Streamlit
- **Embeddings**: Cohere embed-english-v3.0
- **Deployment**: Streamlit Cloud / HuggingFace Spaces

## 📁 Project Structure

```
medical-rag-assistant/
medical_assistant/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies (updated versions)
├── .env.example                   # Environment variables template
├── .gitignore
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py                # Configuration management
├── src/
│   ├── __init__.py
│   ├── components/               # RAG Components
│   │   ├── __init__.py
│   │   ├── data_loader.py       # Document loading component
│   │   ├── data_processor.py    # Text processing & chunking
│   │   ├── embedding.py         # Embedding generation
│   │   ├── vector_store.py      # Vector storage management
│   │   ├── retriever.py         # Information retrieval
│   │   └── generator.py         # Response generation
│   ├── core/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py      # Orchestrates all components
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging utilities
│       └── helpers.py           # Helper functions
├── data/
│   ├── documents/               # Source documents
│   └── processed/              # Processed data cache
├── tests/                      # Test file
│    ├── __init__.py
│    ├── test_components/
│    │   ├── test_data_loader.py
│    │   ├── test_data_processor.py
│    │   ├── test_embedding.py
│    │   ├── test_vector_store.py
│    │   ├── test_retriever.py
│    │   └── test_generator.py
│    └── test_integration/
│        └── test_rag_pipeline.py
└── logs/                       # Application logs
```

## 🚀 Quick Start

## 📋 Requirements

- Python 3.10
- Cohere API key

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/medical-rag-assistant.git
cd medical-rag-assistant
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Cohere API key:
```
COHERE_API_KEY=your_cohere_api_key_here
```

### 3. Run the Application on local

```bash
pip install -r requirements.txt
streamlit run app.py
```
### 4. Run the Application on local
```bash
docker build -t streamlit-app .
docker run -p 8501:8501 streamlit-app
```

The app will be available at `http://localhost:8501`

## 📚 Usage

1. **Upload Documents**: Use the sidebar to upload PDF or TXT files
2. **Ask Questions**: Enter your medical question in the text input
3. **View Results**: See the AI response along with source documents
4. **Check Logs**: Monitor the logs section for debugging information

## 🔧 Configuration

### Document Processing
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters  
- **Retrieval**: Top 3 most relevant chunks

### LLM Settings
- **Model**: command-r-plus
- **Temperature**: 0.1 (for consistent medical responses)
- **Max Tokens**: 1000

## 📊 Logging

The system logs:
- User queries and timestamps
- Retrieved document chunks with scores
- LLM responses and processing times
- Error handling and debug information

Logs are stored in `logs/rag_system.log` with rotation.

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add environment variables in Streamlit settings
4. Deploy automatically

### HuggingFace Spaces
1. Create new Space with Streamlit SDK
2. Upload repository files
3. Add OpenAI API key to Space secrets
4. Space will auto-deploy

## 🧪 Sample Documents

The `data/sample_documents/` folder contains example medical documents:
- Medical clinic policies
- Treatment guidelines
- General health information

## 🔍 API Structure

### Core Components

- **DocumentProcessor**: Handles file ingestion and text chunking
- **VectorStore**: Manages FAISS index and similarity search
- **RAGPipeline**: Orchestrates retrieval and generation
- **Logger**: Provides structured logging throughout the system




## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---