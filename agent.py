import os
import feedparser
import requests
import telegram
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. CONFIGURATION ---
HF_TOKEN = os.environ.get("HF_TOKEN")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

JOURNAL_FEEDS = {
    # ... (Your full list of journals remains here)
    "Harvard Business Review": "http://feeds.harvardbusiness.org/harvardbusiness/",
    "MIT Sloan Management Review": "https://sloanreview.mit.edu/feed/",
    "California Management Review": "https://journals.sagepub.com/action/showFeed?jc=cmr&type=etoc&feed=rss",
    "INSEAD Knowledge": "https://knowledge.insead.edu/rss/all-topics.xml",
    "Knowledge at Wharton (Podcast)": "https://feeds.acast.com/public/shows/621d3ea487eba30014f27133",
    "Insights by Stanford Business": "https://www.gsb.stanford.edu/insights/feed",
    "strategy+business (PwC)": "https://www.strategy-business.com/rss_main",
    "Academy of Management Review (AMR)": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=amr",
    "Strategic Management Journal (SMJ)": "https://onlinelibrary.wiley.com/feed/10970266/most-recent",
    "Academy of Management Journal (AMJ)": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=amj",
    "Organization Science": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=orsc",
    "Journal of Marketing": "https://www.ama.org/feed/?post_type=jm",
    "Journal of Consumer Research": "https://academic.oup.com/jcr/rss/latest",
    "Strategy Science": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=stsc",
    "McKinsey & Company": "https://www.mckinsey.com/~/media/mckinsey/dotcom/rss/featured%20insights/feed",
    "Boston Consulting Group (BCG)": "https://www.bcg.com/en-us/publications/rss",
    "Bain & Company": "https://www.bain.com/insights/rss/",
}

DAYS_TO_CHECK = 1

# --- DATA FETCHING FUNCTION ---
def get_recent_articles(feeds):
    print("Fetching recent articles...")
    all_articles = []
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=DAYS_TO_CHECK)
    for journal, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    if published_time >= cutoff_date:
                        article = { "journal": journal, "title": entry.title, "link": entry.link, "summary": BeautifulSoup(entry.summary, 'html.parser').get_text(separator=' ', strip=True), "published_date": published_time.strftime("%Y-%m-%d") }
                        all_articles.append(article)
        except Exception as e:
            print(f"Could not fetch or parse feed for {journal}. Error: {e}")
    print(f"Found {len(all_articles)} new articles from RSS feeds.")
    return all_articles

# --- AI LOGIC FUNCTION (FINAL RESTRUCTURED VERSION) ---
def analyze_articles_with_ai(articles):
    if not articles:
        return "No new articles found to analyze in the last day."

    print(f"Analyzing {len(articles)} articles one by one for the new report structure...")

    # Define the connection to the Hugging Face Inference API
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        temperature=0.2, # Lower temperature for more factual summaries
        max_new_tokens=2048, # Increased token limit to prevent cut-off
    )
    model = ChatHuggingFace(llm=llm)
    
    deep_dive_parts = []
    
    # Process each article to create a one-line summary
    for i, article in enumerate(articles):
        print(f"Summarizing article {i+1}/{len(articles)}: {article['title']}")
        
        # We only need title and summary for the one-line summary
        article_text = f"Title: {article['title']}\nSummary: {article['summary']}"
        
        messages = [
            SystemMessage(content="You are a business analyst. Your task is to write a single, compelling, one-sentence summary of the provided article."),
            HumanMessage(content=f"Summarize this article:\n\n{article_text}"),
        ]
        
        try:
            result = model.invoke(messages)
            one_line_summary = result.content.strip()
            
            # Format for the numbered list: "1. Summary [Link]"
            formatted_summary = f"{i+1}. {one_line_summary} [Link]({article['link']})\n"
            deep_dive_parts.append(formatted_summary)
            
            # A small, polite delay
            time.sleep(2)
        except Exception as e:
            print(f"Could not summarize article '{article['title']}'. Error: {e}")

    if not deep_dive_parts:
        return "Could not generate summaries for any articles due to processing errors."

    # Combine all one-line summaries into the Deep Dive section
    deep_dive_section = "".join(deep_dive_parts)

    # Now, create the Executive Summary based on the Deep Dive list
    print("Generating Executive Summary...")
    
    summary_messages = [
        SystemMessage(content="You are a strategic analyst. Based on the following list of article summaries, write a single, cohesive paragraph for the 'Executive Summary' section of a report. This paragraph should synthesize the main themes and insights found across all the articles."),
        HumanMessage(content=f"Here is the list of summaries:\n\n{deep_dive_section}")
    ]
    
    executive_summary = model.invoke(summary_messages).content
    
    # Assemble the final report in the desired structure
    final_report = (
        f"## Executive Summary\n{executive_summary}\n\n"
        f"## Deep Dive\n{deep_dive_section}"
    )
    
    return final_report

# --- TELEGRAM SENDER & MAIN FUNCTIONS ---
async def send_report_to_telegram(report, article_count):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not found. Cannot send report.")
        return
    
    print("Sending report to Telegram...")
    
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from_date = (datetime.now(timezone.utc) - timedelta(days=DAYS_TO_CHECK)).strftime("%Y-%m-%d")
    
    header = (
        f"## 📈 Daily Strategic Intelligence Dossier\n\n"
        f"**🗓️ Reporting Period:** {from_date} to {to_date}\n"
        f"**📊 Articles Reviewed:** {article_count}\n\n"
        "---"
    )

    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        full_report = header + "\n" + report
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
    asyncio.run(send_report_to_telegram(report, len(articles)))
    print("\n--- AGENT RUN FINISHED ---")

if __name__ == "__main__":
    main()
