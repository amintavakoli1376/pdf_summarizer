PDF Summarizer & Keyword Extractor
This project provides a Python tool to:
✅ Summarize the content of a PDF file
✅ Extract keywords from the text

It uses Hugging Face Transformers for multilingual summarization (English, Persian/Farsi) and RAKE (Rapid Automatic Keyword Extraction) for keyword extraction.

✨ Features
📄 Read and extract text from PDF files

🌐 Auto-detect language (supports English and Persian/Farsi)

🧠 Generate summaries using state-of-the-art transformer models:

English → t5-base

Farsi → mT5_multilingual_XLSum

🔑 Extract top keywords from the text using RAKE

📦 Requirements
Make sure you have Python 3.7+ installed.

Install dependencies:

bash
Copy
Edit
pip install transformers langdetect huggingface_hub rake-nltk nltk PyPDF2
Also, for Persian stopwords you don’t need additional downloads, but for NLTK:

bash
Copy
Edit
python -m nltk.downloader stopwords
🔐 Hugging Face API Token (Optional)
For better performance or access to private models, set your Hugging Face API token as an environment variable:

bash
Copy
Edit
export HUGGINGFACEHUB_API_TOKEN=your_token_here
🛠 How It Works
The main script:

Reads a PDF file.

Splits the text into manageable chunks.

Summarizes each chunk.

Extracts top keywords from the full text.

Prints the summarized text and keywords.

