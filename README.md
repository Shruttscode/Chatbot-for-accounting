# Chatbot-for-accounting
# AI Chatbot With ChatGPT API

## Presenting to you, *AI Chatbot based on RAG*!

# Financial Statements Chatbot Setup Guide

## Introduction

This repository provides a chatbot that answers queries related to a given PDF document in our case a financial statement related document using a combination of Ollama and Gemini models for natural language understanding and Google Generative AI for embeddings. The bot stores the document data in a vector database,Chroma db for efficient retrieval.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Google API Key for Google Generative AI
- Access to Ollama or Gemini model

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   Create a `.env` file in the root directory of your project and add your Google API Key:
   ```
   GOOGLE_API_KEY=your-google-api-key
   ```

## Usage

### 1. Initializing the Bot

To start the bot, the PDF document needs to be processed and loaded into a vector database if it hasn't been done already. This is handled in the script.

### 2. Running the Bot

Execute the main script to start the bot:
```bash
streamlit run chatbot.py
```

### 3. Interacting with the Bot

Once the Streamlit server is running, you can interact with the bot through the provided UI. The bot maintains a session state to keep track of the conversation history.

## File Structure

- `chatbot.py`: Main script for setting up and running the chatbot.
- `requirements.txt`: List of required Python packages.
- `.env`: Environment file to store sensitive information like API keys.

## Main Components

1. **Model Initialization:**
   - Depending on the `model_type` selected (`ollama` or `gemini`), the script initializes the corresponding model.

2. **Vector Database Setup:**
   - The vector database is created using Chroma. If it's the first time running the script, it will process the provided PDF document, split the text, create embeddings, and persist them.

3. **Querying the Model:**
   - The script sets up a `RetrievalQA` chain using the initialized model and vector database retriever.
   - The chat input and output are managed using Streamlit's session state to preserve the conversation history.

## Example

Place your PDF document in the appropriate path (e.g., `D:\hackathon\module-7-financial-statements.pdf`). Ensure the path is correctly referenced in the script.

## Notes

- Ensure you have the correct API keys and model names set up.
- The document loader and text splitter settings can be adjusted as needed.
- The chatbot UI is built using Streamlit, making it easy to extend or modify.

## Troubleshooting

- If you encounter issues with missing packages, ensure all dependencies are installed from `requirements.txt`.
- Verify the environment variables are correctly set in the `.env` file.
- Check the file paths for the PDF document and vector database directory.

Feel free to customize the setup and configuration based on your specific requirements. Happy coding!
```
And the `requirements.txt` file:

```text
streamlit
langchain
langchain-google-genai
google-generativeai
python-dotenv
```

Make sure to place your PDF document in the appropriate path and update the path in the script accordingly.
