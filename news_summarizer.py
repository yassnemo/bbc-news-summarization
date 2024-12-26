import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from heapq import nlargest
import os
import re

def setup_nltk():
    """Setup NLTK resources"""
    try:
        resources = [
            'punkt',
            'stopwords',
            'averaged_perceptron_tagger',
            'punkt_tab'
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Warning: Could not download {resource}: {e}")
        return True
    except Exception as e:
        print(f"Error setting up NLTK: {e}")
        return False

def clean_text(text):
    """Clean text before processing"""
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s.!?]', '', text)
    return text

def simple_tokenize(text):
    """Regex-based tokenizer"""
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def summarize_text(text, n_sentences=3):
    """Generate text summary with improved fallback"""
    try:
        text = clean_text(text)
        sentences = simple_tokenize(text)  
        
        if len(sentences) <= n_sentences:
            return text
            
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        word_freq = {}
        
        # Calculate word frequencies
        for word in text.lower().split():
            if word.isalnum() and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # Score sentences
        sent_scores = {}
        for sent in sentences:
            for word in sent.lower().split():
                if word in word_freq:
                    sent_scores[sent] = sent_scores.get(sent, 0) + word_freq[word]
                    
        # Get top sentences
        summary_sents = sorted(sent_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:n_sentences]
        return ' '.join(sent for sent, _ in summary_sents)
        
    except Exception as e:
        print(f"Error in summarization: {e}")
        sentences = simple_tokenize(text)
        return ' '.join(sentences[:n_sentences])

def get_bbc_content(url):
    """Extract content from BBC article"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # BBC uses different selectors for different article types
        title = (
            soup.find('h1', {'id': 'main-heading'}) or 
            soup.find('h1', class_='article-headline') or
            soup.find('h1')
        )
        
        # Try different content selectors
        paragraphs = (
            soup.select('article p') or 
            soup.select('.article__body-content p') or
            soup.select('[data-component="text-block"]') or
            soup.select('.story-body p')
        )
        
        print(f"Debug - Title found: {title is not None}")
        print(f"Debug - Paragraphs found: {len(paragraphs)}")
        
        if not title or not paragraphs:
            print("Could not find article content using known selectors")
            return None, None
            
        title_text = title.text.strip()
        content = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])
        
        if not content:
            print("Extracted content is empty")
            return None, None
            
        return title_text, content
        
    except Exception as e:
        print(f"Error fetching article: {e}")
        return None, None

def main():
    if not setup_nltk():
        print("Failed to setup NLTK resources")
        return
        
    while True:
        url = input("\nEnter the news article URL (or 'quit' to exit): ").strip()
        
        if url.lower() == 'quit':
            break
            
        print("\nFetching article...")
        title, content = get_bbc_content(url)
        
        if not content:
            print("Could not extract article content")
            continue
            
        print(f"\nTitle: {title}\n")
        print("Generating summary...")
        
        summary = summarize_text(content)
        if summary:
            print("\nSummary:")
            print(summary)
            
            # Save to file
            with open('summary.txt', 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n\nSummary:\n{summary}")
            print("\nSummary saved to summary.txt")

if __name__ == "__main__":
    main()
