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
AVALAI_BASE_URL = "https://api.avalai.ir/v1"

JOURNAL_FEEDS = {
    "Harvard Business Review": "http://feeds.harvardbusiness.org/harvardbusiness/",
    "MIT Sloan Management Review": "https://sloanreview.mit.edu/feed/",
    "California Management Review": "https://journals.sagepub.com/action/showFeed?jc=cmr&type=etoc&feed=rss",
    "London Business School Review": "https://admissionsblog.london.edu/feed/",
    "AMR": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=amr",
    "SMJ": "https://onlinelibrary.wiley.com/feed/10970266/most-recent",
    "AMJ": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=amj",
    "OrgSci": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=orsc",
    "ManSci": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=mnsc",
    "JM": "https://journals.sagepub.com/action/showFeed?jc=joma&type=etoc&feed=rss",
    "JoM": "https://journals.sagepub.com/action/showFeed?jc=jmxa&type=etoc&feed=rss",
    "Research Policy": "https://rss.sciencedirect.com/publication/science/00487333",
    "ASQ": "https://journals.sagepub.com/action/showFeed?jc=asqa&type=etoc&feed=rss",
    "Deloitte": "https://deloitteuniversitypress.libsyn.com/rss",
    "Mackinsey": "https://www.mckinsey.com/rss",
    "Insead": "https://knowledge.insead.edu/rss.xml",
    "Knowledge at Wharton": "https://knowledge.wharton.upenn.edu/feed/",
    "Strategy Science": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=stsc"
}
DAYS_TO_CHECK = 3 # شما این را به ۳ تغییر دادید
MODEL_TO_USE = "gpt-4o-mini"

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
            if not feed.entries:
                print(f"No entries found for {journal}.")
                continue
                
            for entry in feed.entries:
                published_time_struct = getattr(entry, 'published_parsed', None)
                if published_time_struct:
                     published_time = datetime(*published_time_struct[:6])
                     if published_time >= cutoff_date:
                        article = {"journal": journal, "title": entry.title, "link": entry.link, "summary": BeautifulSoup(getattr(entry, 'summary', ''), 'html.parser').get_text()}
                        all_articles.append(article)
        except Exception as e: 
            print(f"Could not fetch or parse feed for {journal}. Error: {e}")
    print(f"Found {len(all_articles)} new articles across all feeds.")
    return all_articles

def analyze_articles(articles):
    if not articles or client is None: 
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS_TO_CHECK)
        report_parts = [
            f"## 📈 Daily Strategic Intelligence Dossier\n",
            f"**🗓️ Reporting Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            f"**📊 Articles Reviewed:** 0\n\n",
            f"## Executive Summary\nNo new articles were found in the monitored journals within the last {DAYS_TO_CHECK} days."
        ]
        return "\n".join(report_parts)

    print(f"Analyzing {len(articles)} articles one by one (Step 1: Deep Dives)...")
    
    deep_dive_parts = []
    raw_summaries_for_exec_summary = []
    successful_articles_count = 0 

    for i, article in enumerate(articles):
        print(f"Analyzing article {i+1}/{len(articles)}: {article['title']}")
        
        system_message = "You are a concise business analyst. Summarize the following article in a single, 1-2 line summary. Do not add any extra text."
        user_message = f"Article:\n- Journal: {article['journal']}\n- Title: {article['title']}\n- Summary: {article['summary']}"

        try:
            completion = client.chat.completions.create(
                model=MODEL_TO_USE,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=250,
                temperature=0.5,
            )
            single_summary = completion.choices[0].message.content.strip()
            
            successful_articles_count += 1 
            deep_dive_parts.append(f"{successful_articles_count}. {single_summary} ([Link]({article['link']}))\n") 
            raw_summaries_for_exec_summary.append(single_summary)

        except Exception as e:
            print(f"Could not analyze article '{article['title']}'. Error: {e}")
        
        print("Waiting for 3 seconds...")
        time.sleep(3) 

    if not deep_dive_parts: 
        return "Could not analyze any articles due to processing errors."

    print("Waiting for 5 seconds before generating executive summary...")
    time.sleep(5) 
    
    print("Generating Executive Summary (Step 2)...")
    all_summaries_text = "\n".join(raw_summaries_for_exec_summary)
    
    system_message_exec = "You are a senior business strategist. Read the following list of individual article summaries and write one cohesive paragraph (3-5 sentences) that synthetically summarizes the main, overarching themes and trends for a busy executive."
    user_message_exec = f"Here are today's article summaries:\n{all_summaries_text}"
    
    executive_summary = "Could not generate executive summary due to an API error." 
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_TO_USE,
            messages=[
                {"role": "system", "content": system_message_exec},
                {"role": "user", "content": user_message_exec},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        executive_summary = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Could not generate executive summary. Error: {e}")

    print("Assembling final report...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_CHECK)
    
    report_parts = [
        f"## 📈 Daily Strategic Intelligence Dossier\n",
        f"**🗓️ Reporting Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        f"**📊 Articles Reviewed:** {successful_articles_count}\n\n",
        f"## Executive Summary\n{executive_summary}\n\n",
        f"## Deep Dives\n" + "".join(deep_dive_parts)
    ]
    
    return "\n".join(report_parts)

async def send_to_telegram(report, token, chat_id):
    if not token or not chat_id: 
        print("Telegram secrets not found.")
        return
    print("Sending report to Telegram...")
    try:
        bot = telegram.Bot(token=token)
        for i in range(0, len(report), 4096): 
            await bot.send_message(
                chat_id=chat_id, 
                text=report[i:i+4096], 
                parse_mode='Markdown',
                disable_web_page_preview=True  # <--- این خط اضافه شد
            )
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
