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
    # == Business & Management Publications ==
    "Harvard Business Review": "http://feeds.harvardbusiness.org/harvardbusiness/",
    "MIT Sloan Management Review": "https://sloanreview.mit.edu/feed/",
    "California Management Review": "https://journals.sagepub.com/action/showFeed?jc=cmr&type=etoc&feed=rss",
    "INSEAD Knowledge": "https://knowledge.insead.edu/rss/all-topics.xml",
    "Knowledge at Wharton (Podcast)": "https://feeds.acast.com/public/shows/621d3ea487eba30014f27133",
    "Insights by Stanford Business": "https://www.gsb.stanford.edu/insights/feed",
    "strategy+business (PwC)": "https://www.strategy-business.com/rss_main",
    
    # == Top Academic Journals ==
    "Academy of Management Review (AMR)": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=amr",
    "Strategic Management Journal (SMJ)": "https://onlinelibrary.wiley.com/feed/10970266/most-recent",
    "Academy of Management Journal (AMJ)": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=amj",
    "Organization Science": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=orsc",
    "Management Science": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=mnsc",
    "Journal of Management": "https://journals.sagepub.com/action/showFeed?jc=joma&type=etoc&feed=rss",
    "Journal of Marketing": "https://www.ama.org/feed/?post_type=jm",
    "Journal of Consumer Research": "https://academic.oup.com/jcr/rss/latest",
    "Strategy Science": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=stsc",
    
    # == Top Consulting Firms ==
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
                        article = {
                            "journal": journal,
                            "title": entry.title,
                            "link": entry.link,
                            "summary": BeautifulSoup(entry.summary, 'html.parser').get_text(separator=' ', strip=True),
                            "published_date": published_time.strftime("%Y-%m-%d")
                        }
                        all_articles.append(article)
        except Exception as e:
            print(f"Could not fetch or parse feed for {journal}. Error: {e}")
    print(f"Found {len(all_articles)} new articles from RSS feeds.")
    return all_articles

# --- AI LOGIC FUNCTION ---
def analyze_articles_with_ai(articles):
    if not articles:
        return "No new articles found to analyze in the last day."

    print(f"Analyzing {len(articles)} articles one by one...")

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        temperature=0.7,
        max_new_tokens=1024,
    )
    model = ChatHuggingFace(llm=llm)
    
    final_report_parts = []
    
    for i, article in enumerate(articles):
        print(f"Analyzing article {i+1}/{len(articles)}: {article['title']}")
        
        article_text = f"Journal: {article['journal']}\nTitle: {article['title']}\nPublished Date: {article['published_date']}\nSummary: {article['summary']}"
        
        messages = [
            SystemMessage(content="You are a business analyst. Your task is to summarize a single article. Focus on the core idea, its importance, and one actionable insight. The output must be a concise, 3-point summary in markdown format."),
            HumanMessage(content=f"Summarize this article:\n\n{article_text}"),
        ]
        
        try:
            result = model.invoke(messages)
            single_summary = result.content
            
            formatted_summary = f"### [{article['title']}]({article['link']})\n*Source: {article['journal']} ({article['published_date']})*\n\n{single_summary}\n\n---\n"
            final_report_parts.append(formatted_summary)
            
            time.sleep(2)
        except Exception as e:
            print(f"Could not analyze article '{article['title']}'. Error: {e}")

    if not final_report_parts:
        return "Could not analyze any articles due to processing errors."

    full_text_of_summaries = "".join(final_report_parts)
    
    print("Generating Executive Summary and Strategic Synthesis...")
    
    final_synthesis_messages = [
        SystemMessage(content="Based on the following collection of article summaries, write a two-part response. First, a high-level 'Executive Summary' (2-3 bullet points) of the most important overarching themes. Second, a 'Strategic Synthesis' section that identifies common threads and provides one or two strategic recommendations for leaders."),
        HumanMessage(content=full_text_of_summaries)
    ]
    
    synthesis_result = model.invoke(final_synthesis_messages).content
    
    final_report = (
        f"{synthesis_result}\n\n"
        f"## Deep Dive: Key Articles\n{full_text_of_summaries}"
    )
    
    return final_report

# --- TELEGRAM SENDER (UPDATED) & MAIN FUNCTIONS ---
async def send_report_to_telegram(report, article_count): # Added article_count
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not found. Cannot send report.")
        return
    
    print("Sending report to Telegram...")
    
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from_date = (datetime.now(timezone.utc) - timedelta(days=DAYS_TO_CHECK)).strftime("%Y-%m-%d")
    
    # New header including the article count
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
    # Pass the number of articles to the sender function
    asyncio.run(send_report_to_telegram(report, len(articles)))
    print("\n--- AGENT RUN FINISHED ---")

if __name__ == "__main__":
    main()
