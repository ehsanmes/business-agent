import os
import feedparser
import telegram
import time
import asyncio
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import google.generativeai as genai

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
JOURNAL_FEEDS = {
    "Harvard Business Review": "http://feeds.harvardbusiness.org/harvardbusiness/",
    "MIT Sloan Management Review": "https://sloanreview.mit.edu/feed/",
}
DAYS_TO_CHECK = 3

def get_recent_articles(feeds):
    print("Fetching recent articles...")
    all_articles = []
    # ... (This function is the same and correct)
    cutoff_date = datetime.now() - timedelta(days=DAYS_TO_CHECK)
    for journal, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published_time = datetime(*entry.published_parsed[:6])
                if published_time >= cutoff_date:
                    article = {"journal": journal, "title": entry.title, "link": entry.link, "summary": BeautifulSoup(entry.summary, 'html.parser').get_text()}
                    all_articles.append(article)
        except Exception as e: print(f"Could not fetch or parse feed for {journal}. Error: {e}")
    print(f"Found {len(all_articles)} new articles.")
    return all_articles

def analyze_articles(articles, model):
    if not articles: return "No new articles found."
    print(f"Analyzing {len(articles)} articles one by one...")
    final_report_parts = []
    for i, article in enumerate(articles):
        print(f"Analyzing article {i+1}/{len(articles)}: {article['title']}")
        prompt = f"""You are a senior business analyst. Analyze the following article and provide a concise, 3-point summary in markdown format. Focus on: 1. The core idea, 2. Why it matters, and 3. One actionable insight for a leader.

Please analyze this article:
- Journal: {article['journal']}
- Title: {article['title']}
- Summary: {article['summary']}
"""
        try:
            # Direct call to Google's library, no LangChain
            response = model.generate_content(prompt)
            single_summary = response.text
            final_report_parts.append(f"### {article['title']}\n*Source: {article['journal']}*\n\n{single_summary}\n\n---\n")
        except Exception as e: print(f"Could not analyze article '{article['title']}'. Error: {e}")
        print("Waiting for 65 seconds to respect Google's rate limits...")
        time.sleep(65)
    if not final_report_parts: return "Could not analyze any articles."
    return "## 📈 Daily Strategic Intelligence Dossier\n\n" + "".join(final_report_parts)

async def send_to_telegram(report, token, chat_id):
    if not token or not chat_id: return
    print("Sending report to Telegram...")
    try:
        bot = telegram.Bot(token=token)
        for i in range(0, len(report), 4096):
            await bot.send_message(chat_id=chat_id, text=report[i:i+4096], parse_mode='Markdown')
        print("Report successfully sent.")
    except Exception as e: print(f"Failed to send report. Error: {e}")

def main():
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY secret is not set.")
        return

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro') # Using the standard model name for this library
        print("✅ Google Gemini model initialized successfully.")
    except Exception as e:
        print(f"❌ Critical error during model initialization: {e}")
        return

    articles = get_recent_articles(JOURNAL_FEEDS)
    report = analyze_articles(articles, model)
    asyncio.run(send_to_telegram(report, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID))
    print("\n--- AGENT RUN FINISHED ---")

if __name__ == "__main__":
    main()
