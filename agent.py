import os
import feedparser
import requests
import telegram
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

# --- 1. CONFIGURATION ---
HF_TOKEN = os.environ.get("HF_TOKEN")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

JOURNAL_FEEDS = {
    "Harvard Business Review": "http://feeds.harvardbusiness.org/harvardbusiness/",
    "MIT Sloan Management Review": "https://sloanreview.mit.edu/feed/",
    "California Management Review": "https://cmr.berkeley.edu/feed/",
    "Journal of Management": "https://journals.sagepub.com/rss/loi_jom.xml",
    "Strategic Management Journal": "https://onlinelibrary.wiley.com/feed/1467-6486/most-recent",
    "Organization Science": "https://pubsonline.informs.org/rss/orgsci.xml",
    "Journal of Marketing (AMA)": "https://www.ama.org/feed/?post_type=jm",
    "Journal of Consumer Research (Oxford)": "https://academic.oup.com/jcr/rss/latest",
    "Journal of Business Venturing (Elsevier)": "https://rss.sciencedirect.com/publication/journals/08839026",
    "Journal of the Academy of Marketing Science (Springer)": "https://link.springer.com/journal/11747/rss.xml"
}

DAYS_TO_CHECK = 5

# --- DATA FETCHING FUNCTION (No changes) ---
def get_recent_articles(feeds):
    print("Fetching recent articles...")
    all_articles = []
    cutoff_date = datetime.now() - timedelta(days=DAYS_TO_CHECK)
    for journal, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # Check for a valid published date
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_time = datetime(*entry.published_parsed[:6])
                    if published_time >= cutoff_date:
                        article = {
                            "journal": journal,
                            "title": entry.title,
                            "link": entry.link,
                            "summary": BeautifulSoup(entry.summary, 'html.parser').get_text(separator=' ', strip=True)
                        }
                        all_articles.append(article)
        except Exception as e:
            print(f"Could not fetch or parse feed for {journal}. Error: {e}")
    print(f"Found {len(all_articles)} new articles from RSS feeds.")
    return all_articles

# --- AI LOGIC FUNCTION (Updated for new model and prompt) ---
def analyze_articles_with_ai(articles):
    if not articles:
        return "No new articles found to analyze in the last few days."

    print("Sending articles to Hugging Face for analysis...")
    articles_text = ""
    for article in articles:
        articles_text += f"Journal: {article['journal']}\nTitle: {article['title']}\nSummary: {article['summary']}\n\n---\n\n"

    # A simplified and more direct prompt for the new model
    prompt_template = """
    Task: You are a world-class senior business research analyst. Analyze the following list of recently published articles and generate a concise, markdown-formatted 'Executive Dossier'.
    The dossier must have these sections: Executive Summary, Deep Dive, Strategic Synthesis, and On the Horizon.

    Here is the list of articles to analyze:
    {articles_text}

    Begin the 'Executive Dossier' now:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Initialize the new, more compatible Hugging Face model endpoint
    model = HuggingFaceEndpoint(
        repo_id="google/flan-t5-xxl",  # <--- Changed to a highly compatible model
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.8,
        max_new_tokens=2048,
    )
    
    chain = prompt | model | StrOutputParser()
    
    report = chain.invoke({"articles_text": articles_text})
    return report

# --- TELEGRAM SENDER & MAIN FUNCTIONS (No changes) ---
async def send_report_to_telegram(report):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not found. Cannot send report.")
        return
    
    print("Sending report to Telegram...")
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        full_report = "## 📈 Daily Business & Academic Dossier\n\n" + report
        max_length = 4096
        for i in range(0, len(full_report), max_length):
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=full_report[i:i+max_length],
                parse_mode='Markdown'
            )
        print("Report successfully sent to Telegram.")
    except Exception as e:
        print(f"Failed to send report to Telegram. Error: {e}")

def main():
    import asyncio
    articles = get_recent_articles(JOURNAL_FEEDS)
    report = analyze_articles_with_ai(articles)
    asyncio.run(send_report_to_telegram(report))
    print("\n--- AGENT RUN FINISHED ---")

if __name__ == "__main__":
    main()
