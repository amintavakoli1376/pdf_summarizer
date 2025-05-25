from transformers import pipeline, TFAutoModelForSeq2SeqLM, AutoTokenizer , AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
from huggingface_hub import login
from rake_nltk import Rake
from nltk.corpus import stopwords
import nltk
import os
import PyPDF2



class LanguageConfig:
    """Configuration for different languages and their models."""
    SUPPORTED_LANGUAGES = {
        'en': {'model': 't5-base', 'stop_words': stopwords.words('english')},
        'fa': {'model': 'csebuetnlp/mT5_multilingual_XLSum', 'stop_words': ['و', 'در', 'به', 'از', 'که', 'این', 'برای', 'است', 'را', 'با', 'می', 'شود', 'کرد', 'ها', 'های', 'شد', 'هم', 'آن']}
    }

    @classmethod
    def get_config(cls, language: str):
        if language not in cls.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        return cls.SUPPORTED_LANGUAGES[language]


class TextSummarizer:
    """Handles text summarization tasks."""
    def __init__(self):
        self.summarizers = {}
        self._setup_nltk()
        self._login_huggingface()
        DetectorFactory.seed = 0

    def _setup_nltk(self):
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)

    def _login_huggingface(self):
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if token:
            login(token=token)

    def load_summarizer(self, language: str, max_length: int = 70):
        config = LanguageConfig.get_config(language)
        tokenizer = AutoTokenizer.from_pretrained(config['model'])
        model = (TFAutoModelForSeq2SeqLM if language == 'en' else AutoModelForSeq2SeqLM).from_pretrained(config['model'])
        self.summarizers[language] = pipeline('summarization', model=model, tokenizer=tokenizer, max_length=max_length, do_sample=False)

    def summarize_text(self, text: str, language: str = None, max_length: int = 70) -> str:
        language = language or detect(text)
        if language not in self.summarizers:
            self.load_summarizer(language, max_length)
        summarizer = self.summarizers.get(language)
        return summarizer(text, max_length=max_length)[0]['summary_text'] if summarizer else "Summarization failed."



class KeywordExtractor:
    """Handles keyword extraction from text."""
    def __init__(self):
        pass

    @staticmethod
    def extract_keywords(text: str, language: str = 'en', top_n: int = 10):
        stop_words = LanguageConfig.get_config(language)['stop_words']
        rake = Rake(stopwords=stop_words, max_length=3)
        rake.extract_keywords_from_text(text)
        return rake.get_ranked_phrases()[:top_n]


def read_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()



if __name__ == "__main__":
    summarizer = TextSummarizer()
    extractor = KeywordExtractor()

    pdf_path = 'WritingontheWall.pdf'
    text = read_pdf(pdf_path)

    chunk_size = 2000
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    summary = ""
    for chunk in text_chunks:
        summary += summarizer.summarize_text(chunk, max_length=100) + " "
    summary = summary.strip()

    keywords = extractor.extract_keywords(text)

    print("\n--- PDF Content ---")
    print("Summarized Text:\n", summary)
    print("Extracted Keywords:\n", keywords)
