# Smart Grievance & Query Resolution Bot 🤖

An interactive, local Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangChain**, **ChromaDB**, and **Google Gemini 3.5 Flash**.

## 🌟 Features

- **PDF Rulebook Processing**: Ingests official PDFs via `PyPDFLoader` and `RecursiveCharacterTextSplitter`.

- **Local Vector Search**: Fast similarity search using **ChromaDB** and HuggingFace embeddings `all-MiniLM-L6-v2`).

- **Grounded Answers**: Queries Google Gemini API for responses citing rulebook sections and page numbers.

- **Real-Time Streaming**: Delivers live word-by-word responses in 3–4 seconds.

## 🚀 Setup & Run

1. **Clone Repository**:

   ```bash

   git clone [[https://github.com/YOUR_USERNAME/smart-query-bot.git](https://github.com/YOUR_USERNAME/smart-query-bot.git)](https://github.com/YOUR_USERNAME/smart-query-bot.git](https://github.com/YOUR_USERNAME/smart-query-bot.git))

   cd smart-query-bot