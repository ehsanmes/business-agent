import os
import feedparser
import requests
import telegram # new import
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# Note: We don't need python-dotenv in the cloud version
# from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION ---
# Secrets will be loaded from GitHub's environment, not a .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

JOURNAL_FEEDS = {
    "Harvard Business Review": "https://hbr.org/rss/pre/latest",
    "MIT Sloan Management": "https://sloanreview.mit.edu/feed/",
    "California Management Review": "https://cmr.berkeley.edu/feed/",
    "Strategic Management Journal": "https://onlinelibrary.wiley.com/feed/1467-6486/most-recent",
}
DAYS_TO_CHECK = 2 

# --- DATA FETCHING FUNCTION (No changes needed) ---
def get_recent_articles(feeds):
    print("Fetching recent articles...")
    all_articles = []
    cutoff_date = datetime.now() - timedelta(days=DAYS_TO_CHECK)
    for journal, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published_time = datetime(*entry.published_parsed[:6])
                if published_time >= cutoff_date:
                    article = {
                        "journal": journal, "title": entry.title,
                        "link": entry.link, "summary": BeautifulSoup(entry.summary, 'html.parser').get_text()
                    }
                    all_articles.append(article)
        except Exception as e:
            print(f"Could not fetch or parse feed for {journal}. Error: {e}")
    print(f"Found {len(all_articles)} new articles.")
    return all_articles

# --- AI LOGIC FUNCTION (No changes needed) ---
def analyze_articles_with_ai(articles):
    if not articles:
        return "No new articles found to analyze in the last few days."
    print("Sending articles to AI for analysis...")
    articles_text = ""
    for article in articles:
        articles_text += f"Journal: {article['journal']}\nTitle: {article['title']}\nLink: {article['link']}\nSummary: {article['summary']}\n\n---\n\n"
    
    prompt_template = """
    You are a world-class senior business research analyst and strategist. 
    Analyze the following list of recently published articles and generate a concise, 'Executive Dossier'.
    Go beyond simple summarization. Synthesize information, identify groundbreaking theories, and extract actionable insights.
    The output must be markdown-formatted.

    Here are the latest articles:
    {articles_text}

    Structure your output with these sections:
    1. **Executive Summary:** 2-3 bullet points on the most critical trend or discovery.
    2. **Deep Dive: New Theories & Discoveries:** Analyze 2-3 significant new ideas (Core Idea, Why It Matters, Actionable Insight).
    3. **Strategic Synthesis:** Identify the overarching theme and provide one strategic recommendation.
    4. **On the Horizon:** Mention emerging topics or weak signals.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=OPENAI_API_KEY)
    chain = prompt | model | StrOutputParser()
    report = chain.invoke({"articles_text": articles_text})
    return report

# --- NEW: TELEGRAM SENDER FUNCTION ---
async def send_report_to_telegram(report):
    """Sends the generated report to your Telegram using a bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not found. Cannot send report.")
        return
        
    print("Sending report to Telegram...")
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        # We split the message if it's too long for Telegram
        max_length = 4096
        for i in range(0, len(report), max_length):
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=report[i:i+max_length],
                parse_mode='Markdown'
            )
        print("Report successfully sent to Telegram.")
    except Exception as e:
        print(f"Failed to send report to Telegram. Error: {e}")

# --- EXECUTION ---
def main():
    """Main function to run the agent."""
    import asyncio
    articles = get_recent_articles(JOURNAL_FEEDS)
    report = analyze_articles_with_ai(articles)
    
    # We now send the report instead of saving it to a file
    asyncio.run(send_report_to_telegram(report))
        
    print("\n--- AGENT RUN FINISHED ---")
    print(report)

if __name__ == "__main__":

    main()

