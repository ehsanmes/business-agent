import os
import feedparser
import requests
import telegram
import time # <--- کتابخانه مدیریت زمان را اضافه کردیم
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. CONFIGURATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

JOURNAL_FEEDS = {
    "Harvard Business Review": "http://feeds.harvardbusiness.org/harvardbusiness/",
    "MIT Sloan Management Review": "https://sloanreview.mit.edu/feed/",
    "California Management Review": "https://cmr.berkeley.edu/feed/",
    "Journal of Management": "https://journals.sagepub.com/rss/loi_jom.xml",
    "Strategic Management Journal": "https://onlineli_brary.wiley.com/feed/1467-6486/most-recent",
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
    # ... (این تابع بدون تغییر باقی می‌ماند)
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
    print(f"Found {len(all_articles)} new articles from RSS feeds.")
    return all_articles

# --- AI LOGIC FUNCTION (THIS IS THE NEW, ROBUST VERSION) ---
def analyze_articles_with_ai(articles):
    if not articles:
        return "No new articles found to analyze in the last few days from any of the monitored journals."
    
    print(f"Analyzing {len(articles)} articles one by one to ensure quality and respect rate limits...")
    
    # Initialize the model only once
    model = ChatGoogleGenerativeAI(
        model="gemini-1.0-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5,
        convert_system_message_to_human=True
    )
    
    final_report_parts = []
    
    # Process each article individually in a loop
    for i, article in enumerate(articles):
        print(f"Analyzing article {i+1}/{len(articles)}: {article['title']}")
        
        article_text = f"Journal: {article['journal']}\nTitle: {article['title']}\nLink: {article['link']}\nSummary: {article['summary']}"
        
        prompt_template = """
        You are a senior business analyst. Analyze the following single article and provide a concise, 3-point summary in markdown format.
        Focus on: 1. The core idea, 2. Why it matters, and 3. One actionable insight for a leader.

        Article to analyze:
        {article_text}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | model | StrOutputParser()
        
        try:
            # Invoke the AI for this single article
            single_summary = chain.invoke({"article_text": article_text})
            
            # Format the output for the final report
            formatted_summary = f"### {article['title']}\n*Source: {article['journal']}*\n\n{single_summary}\n\n---\n"
            final_report_parts.append(formatted_summary)

        except Exception as e:
            print(f"Could not analyze article '{article['title']}'. Error: {e}")
        
        # CRITICAL STEP: Wait for a few seconds before the next request to avoid rate limits
        print("Waiting for 10 seconds before next request...")
        time.sleep(10)

    if not final_report_parts:
        return "Could not analyze any articles due to processing errors."

    # Combine all individual summaries into one final report
    final_report = "## Daily Business & Academic Dossier\n\n" + "\n".join(final_report_parts)
    return final_report

# --- TELEGRAM SENDER FUNCTION (No changes) ---
async def send_report_to_telegram(report):
    # ... (این تابع بدون تغییر باقی می‌ماند)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not found. Cannot send report.")
        return
    print("Sending report to Telegram...")
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
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
    import asyncio
    articles = get_recent_articles(JOURNAL_FEEDS)
    report = analyze_articles_with_ai(articles)
    asyncio.run(send_report_to_telegram(report))
    print("\n--- AGENT RUN FINISHED ---")
    print(report)

if __name__ == "__main__":
    main()
