# Medical Assistant RAG System

A complete Retrieval-Augmented Generation (RAG) system that answers medical questions based on document corpus using LangChain, FAISS, and OpenAI GPT-4.

## 🚀 Live Demo

**Deployed URL**: [Your deployment URL will go here]

## 📋 Features

- **Document Ingestion**: Support for PDF and TXT files
- **Smart Chunking**: Intelligent text splitting with overlap
- **Vector Search**: FAISS-powered semantic retrieval
- **LLM Generation**: OpenAI GPT-4 for accurate responses
- **Web Interface**: Clean Streamlit UI
- **Comprehensive Logging**: Query tracking and debug information
- **Source Attribution**: Shows retrieved document snippets

## 🛠️ Tech Stack

- **Backend**: Python, LangChain, FAISS
- **LLM**: OpenAI GPT-4-turbo
- **Frontend**: Streamlit
- **Embeddings**: OpenAI text-embedding-3-small
- **Deployment**: Streamlit Cloud / HuggingFace Spaces

## 📁 Project Structure

```
medical-rag-assistant/
├── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # Document ingestion & chunking
│   ├── vector_store.py       # FAISS vector store management
│   ├── rag_pipeline.py       # Core RAG logic
│   └── logger.py            # Logging configuration
├── data/
│   ├── sample_documents/    # Sample medical documents
│   └── vector_store/        # FAISS index storage
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/medical-rag-assistant.git
cd medical-rag-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application

```bash
streamlit run app.py
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
- **Model**: gpt-4-turbo
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

## 📝 Example Queries

Try these sample questions:
- "What are the clinic's operating hours?"
- "What should I do if I have a fever?"
- "What are the vaccination requirements?"
- "How do I schedule an appointment?"

## 🛠️ Development

### Adding New Document Types
Extend `DocumentProcessor` to support additional file formats.

### Customizing Retrieval
Modify `VectorStore` to adjust similarity search parameters.

### Enhancing Responses
Update the prompt template in `RAGPipeline` for domain-specific improvements.

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for model access

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Ensure `.env` file exists with valid `OPENAI_API_KEY`

**"No documents found"**
- Upload documents using the sidebar
- Check that files are PDF or TXT format

**"Vector store not initialized"**
- Documents need to be processed first before asking questions

### Getting Help

- Check the logs in the Streamlit interface
- Review `logs/rag_system.log` for detailed error information
- Ensure all dependencies are installed correctly

---

**Built with ❤️ for the AI Engineer Technical Test**