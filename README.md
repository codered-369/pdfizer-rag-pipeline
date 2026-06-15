# 📄 PDFizer: RAG Pipeline

**Live App:** [https://pdfizer-rag-pipeline.streamlit.app/](https://pdfizer-rag-pipeline.streamlit.app/)

PDFizer is a robust Retrieval-Augmented Generation (RAG) web application built to allow users to interact intelligently with their PDF documents. Upload any PDF and immediately start asking questions. The app uses advanced AI embeddings to search your document for relevant context and Google's Gemini LLM to generate precise, factual answers based entirely on your document.

## ✨ Features

- **Document Ingestion:** Upload any PDF to automatically extract text and segment it into optimized context chunks.
- **Pure Python Vector Store:** Uses a lightweight `SKLearnVectorStore` for high-speed, zero-dependency document retrieval, ensuring flawless deployment on cloud servers without C++ build requirements.
- **Intelligent RAG Pipeline:** Combines Google GenAI Embeddings (`models/gemini-embedding-2`) for semantic search and the Gemini (`gemini-3.1-flash-lite`) model for answering queries.
- **Dynamic Retrieval:** Automatically adjusts search constraints (k-nearest neighbors) based on document size to prevent out-of-bounds errors on small documents.
- **Session History:** Keeps track of your conversations and previously uploaded documents.
- **UI/UX:** Built with Streamlit, featuring a modern dark/light mode toggle and a beautiful, intuitive interface.

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **LLM & Embeddings:** Google Generative AI (`gemini-3.1-flash-lite`, `gemini-embedding-2`)
- **Vector Database:** Scikit-Learn (`SKLearnVectorStore`)
- **Orchestration:** LangChain Core & Community (Pure Python implementation to avoid module dependency issues)

## 🚀 Running Locally

### Prerequisites

You need an active Google API Key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/codered-369/pdfizer-rag-pipeline.git
   cd pdfizer-rag-pipeline
   ```

2. **Set up environment variables**
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **Install dependencies**
   Install the required Python packages (Python 3.11+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```
   The app will open automatically in your browser at `http://localhost:8501`.

## ☁️ Deployment

This application is optimized for deployment on Streamlit Community Cloud. It includes a `packages.txt` file to ensure required system-level dependencies (like `build-essential` and `libffi-dev`) are installed prior to the Python environment setup, guaranteeing a smooth deployment process even on newer Python environments.
