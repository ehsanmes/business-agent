import os
import feedparser
import telegram
import time
import asyncio
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from openai import OpenAI

# --- 1. CONFIGURATION ---
AVALAI_API_KEY = os.environ.get("AVALAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
AVALAI_BASE_URL = "https://api.avalai.ir/v1" # استفاده از دامنه اصلی

JOURNAL_FEEDS = {
    "Harvard Business Review": "http://feeds.harvardbusiness.org/harvardbusiness/",
    "MIT Sloan Management Review": "https://sloanreview.mit.edu/feed/",
    # ...می‌توانید منابع RSS بیشتری اینجا اضافه کنید
}
DAYS_TO_CHECK = 3 # بررسی مقالات ۳ روز گذشته
MODEL_TO_USE = "gpt-4o-mini" # <--- مدل نهایی انتخابی شما

# --- 2. INITIALIZE THE AI CLIENT ---
client = None
if AVALAI_API_KEY:
    try:
        client = OpenAI(
            api_key=AVALAI_API_KEY,
            base_url=AVALAI_BASE_URL,
        )
        print("✅ AvalAI client configured successfully.")
    except Exception as e:
        print(f"❌ Critical error during client initialization: {e}")
else:
    print("❌ AVALAI_API_KEY secret is not set.")

# --- 3. FUNCTIONS ---
def get_recent_articles(feeds):
    print("Fetching recent articles...")
    all_articles = []
    cutoff_date = datetime.now() - timedelta(days=DAYS_TO_CHECK)
    for journal, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published_time_struct = getattr(entry, 'published_parsed', None)
                if published_time_struct:
                     published_time = datetime(*published_time_struct[:6])
                     if published_time >= cutoff_date:
                        article = {"journal": journal, "title": entry.title, "link": entry.link, "summary": BeautifulSoup(getattr(entry, 'summary', ''), 'html.parser').get_text()}
                        all_articles.append(article)
        except Exception as e: 
            print(f"Could not fetch or parse feed for {journal}. Error: {e}")
    print(f"Found {len(all_articles)} new articles.")
    return all_articles

def analyze_articles(articles):
    if not articles or client is None: 
        return "No new articles found or AI client is unavailable."

    print(f"Analyzing {len(articles)} articles one by one...")
    final_report_parts = []

    for i, article in enumerate(articles):
        print(f"Analyzing article {i+1}/{len(articles)}: {article['title']}")
        
        system_message = "You are a senior business analyst. Analyze the following article and provide a concise, 3-point summary in markdown format. Focus on: 1. The core idea, 2. Why it matters, and 3. One actionable insight for a leader."
        user_message = f"Please analyze this article:\n\nJournal: {article['journal']}\nTitle: {article['title']}\nSummary: {article['summary']}"

        try:
            completion = client.chat.completions.create(
                model=MODEL_TO_USE,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=512,
                temperature=0.7,
            )
            single_summary = completion.choices[0].message.content.strip()
            final_report_parts.append(f"### {article['title']}\n*Source: {article['journal']}*\n\n{single_summary}\n\n---\n")
        except Exception as e:
            print(f"Could not analyze article '{article['title']}'. Error: {e}")
            final_report_parts.append(f"### {article['title']}\n*Source: {article['journal']}*\n\n---\n*Could not be analyzed due to an API error.*\n\n---\n")
        
        # وقفه کوتاه برای احترام به محدودیت‌های احتمالی API
        print("Waiting for 5 seconds...")
        time.sleep(5) 

    if not final_report_parts: 
        return "Could not analyze any articles due to processing errors."
    
    return "## 📈 Daily Strategic Intelligence Dossier\n\n" + "".join(final_report_parts)

async def send_to_telegram(report, token, chat_id):
    if not token or not chat_id: 
        print("Telegram secrets not found.")
        return
    print("Sending report to Telegram...")
    try:
        bot = telegram.Bot(token=token)
        for i in range(0, len(report), 4096): # ارسال پیام‌های طولانی در چند بخش
            await bot.send_message(chat_id=chat_id, text=report[i:i+4096], parse_mode='Markdown')
        print("Report successfully sent.")
    except Exception as e: 
        print(f"Failed to send report. Error: {e}")

# --- 4. EXECUTION ---
def main():
    if client is None:
        print("Error: AvalAI client could not be initialized. Check API Key secret.")
        return

    articles = get_recent_articles(JOURNAL_FEEDS)
    report = analyze_articles(articles)
    asyncio.run(send_to_telegram(report, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID))
    print("\n--- AGENT RUN FINISHED ---")

if __name__ == "__main__":
    main()
