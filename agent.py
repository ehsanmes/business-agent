import os
import feedparser
import requests
import telegram
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# NEW IMPORTS FOR GOOGLE
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI # <--- Replaced OpenAI with Google

# --- 1. CONFIGURATION ---
# We now use the Google API Key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
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

DAYS_TO_CHECK = 5 # Let's keep it wide for the first test

# --- DATA FETCHING FUNCTION (No changes) ---
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
    print(f"Found {len(all_articles)} new articles from RSS feeds.")
    return all_articles

# --- AI LOGIC FUNCTION (Updated for Google Gemini) ---
def analyze_articles_with_ai(articles):
    if not articles:
        return "No new articles found to analyze in the last few days from any of the monitored journals."
    print("Sending articles to Google Gemini for analysis...")
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
    
    # Initialize the Google Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # A fast and powerful free model
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5,
        convert_system_message_to_human=True # Important for compatibility
    )
    
    chain = prompt | model | StrOutputParser()
    
    report = chain.invoke({"articles_text": articles_text})
    return report

# --- TELEGRAM SENDER FUNCTION (No changes) ---
async def send_report_to_telegram(report):
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
