import os
import feedparser
import telegram
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

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
    # ... (Code is the same as before)
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

def analyze_articles(articles, llm):
    if not articles:
        return "No new articles found."
    print(f"Analyzing {len(articles)} articles one by one...")
    final_report_parts = []
    for i, article in enumerate(articles):
        print(f"Analyzing article {i+1}/{len(articles)}: {article['title']}")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a senior business analyst..."),
            ("user", "Please analyze this article:\n\nJournal: {journal}\nTitle: {title}\nSummary: {summary}")
        ])
        chain = prompt_template | llm | StrOutputParser()
        try:
            single_summary = chain.invoke(article)
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
    import asyncio
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY secret is not set.")
        return

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=GOOGLE_API_KEY, temperature=0.5)
        print("✅ Google Gemini model initialized successfully.")
    except Exception as e:
        print(f"❌ Critical error during model initialization: {e}")
        return

    articles = get_recent_articles(JOURNAL_FEEDS)
    report = analyze_articles(articles, llm)
    asyncio.run(send_to_telegram(report, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID))
    print("\n--- AGENT RUN FINISHED ---")

if __name__ == "__main__":
    main()
