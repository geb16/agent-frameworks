<div align="center">
   <h1>Agent Frameworks – RAG Policy Chatbot</h1>
   <p><b>Repository:</b> <a href="https://github.com/yourusername/agent-frameworks">agent-frameworks</a></p>
   <p>
      <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
      <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
      <img src="https://img.shields.io/badge/streamlit-ready-brightgreen" alt="Streamlit Ready">
   </p>
   <p><i>Modern, robust, and production-ready RAG chatbot framework for policy Q&A, built with LangChain, ChromaDB, and OpenAI.</i></p>
</div>

# Overview
# This project demonstrates a modern Retrieval-Augmented Generation (RAG) chatbot for answering policy-related questions using LangChain, ChromaDB, and OpenAI models. It includes:
# - Streamlit web app (`lc_app.py`)
# - Runnable RAG script (`lc_rag.py`)
# - Policy data ingestion from `data/policies.txt`

# Agent Frameworks – RAG Policy Chatbot

## Overview
This project demonstrates a modern Retrieval-Augmented Generation (RAG) chatbot for answering policy-related questions using LangChain, ChromaDB, and OpenAI models. It includes:
- Streamlit web app (`lc_app.py`)
- Runnable RAG script (`lc_rag.py`)
- Policy data ingestion from `data/policies.txt`

## Features
- Loads and embeds company policies from a text file into a ChromaDB vector store
- Uses OpenAI embeddings and chat models for semantic search and Q&A
- Answers only from provided context; does not hallucinate
- User-friendly Streamlit interface with chat history and source viewing
- Automatic ingestion: If the vector store is empty, policies are embedded on startup
- Robust path handling for cross-platform compatibility

## File Structure
```
module8_frameworks/
├── lc_app.py           # Streamlit RAG chatbot web app
├── lc_rag.py           # Runnable RAG script for CLI Q&A
├── requirements.txt    # Python dependencies
├── README.md           # This documentation
├── data/
│   └── policies.txt    # Company policies (knowledge base)
└── db/                 # ChromaDB vector store (auto-created)
```

## Setup Instructions
1. **Install Python 3.10+ and create a virtual environment:**
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set your OpenAI API key:**
   - Create a `.env` file in this folder:
     ```
     OPENAI_API_KEY=your-key-here
     ```

## Usage
### Run the Streamlit Web App
```sh
streamlit run lc_app.py
```
- Ask questions about company policies in the chat interface.
- View sources for each answer.

### Run the CLI Script
```sh
python lc_rag.py
```
- Answers a sample question using the RAG pipeline.
- Modify the script to ask custom questions.

## How It Works
- On startup, the app checks if the ChromaDB vector store is empty.
- If empty, it reads `data/policies.txt`, splits it into paragraphs, and embeds them.
- User questions are converted to embeddings and matched against the policy database.
- The answer is generated using only the retrieved context.

## Best Practices
- Always use absolute paths for file and database access.
- Keep your `requirements.txt` up to date and compatible.
- Update `data/policies.txt` as your policies change. Delete the `db/` folder to force re-ingestion if needed.
- Never hardcode secrets; use environment variables and `.env` files.

## Troubleshooting
- **ChromaDB errors (e.g., PanicException):** Delete the `db/` directory and restart the app.
- **File not found:** Ensure paths are correct and use absolute path logic as shown in the code.
- **API errors:** Check your `.env` file and OpenAI API key.

## Dependencies
- `langchain`
- `langchain-openai`
- `langchain-chroma`
- `streamlit`
- `python-dotenv`

## License

This project is licensed under the MIT License – a permissive, business-friendly open source license. You are free to use, modify, and distribute this software in personal, academic, or commercial projects, provided you include the original copyright.

<details>
<summary>MIT License (click to expand)</summary>

```
MIT License

Copyright (c) 2026 Daniel E./ University of London

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
</details>

MIT License. See repository for details.

## Author
Daniel E./ University of London

---
For questions or improvements, open an issue or contact the maintainer.
