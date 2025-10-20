import os
import feedparser
import requests
import telegram
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. CONFIGURATION ---
HF_TOKEN = os.environ.get("HF_TOKEN")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# The final, comprehensive list of RSS feeds
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
    "Deloitte Insights (Podcast)": "https://deloitteuniversitypress.libsyn.com/rss",
    "Accenture Newsroom": "https://newsroom.accenture.com/rss.xml"
}

# Set to check for articles from the last day
DAYS_TO_CHECK = 1

# --- DATA FETCHING FUNCTION (No changes needed) ---
def get_recent_articles(feeds):
    print("Fetching recent articles...")
    all_articles = []
    cutoff_date = datetime.now() - timedelta(days=DAYS_TO_CHECK)
    for journal, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_time = datetime(*entry.published_parsed[:6])
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

# --- AI LOGIC FUNCTION (Improved with links, dates, and better instructions) ---
def analyze_articles_with_ai(articles):
    if not articles:
        return "No new articles found to analyze in the last day."

    print("Sending articles to Hugging Face (Mistral model) for analysis...")
    
    articles_text = ""
    for article in articles:
        articles_text += f"- Journal: {article['journal']}\n"
        articles_text += f"  Title: [{article['title']}]({article['link']})\n"
        articles_text += f"  Published Date: {article['published_date']}\n"
        articles_text += f"  Summary: {article['summary']}\n\n"

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        temperature=0.7,
        max_new_tokens=3072,
    )

    model = ChatHuggingFace(llm=llm)

    messages = [
        SystemMessage(
            content="You are a world-class senior business research analyst. Your task is to analyze the provided list of articles and generate a concise, professional, markdown-formatted 'Executive Dossier'. Crucially, you must follow these rules:\n1. When mentioning an article, use its title as a clickable markdown link using the provided URL.\n2. Use the exact 'Published Date' provided for each article and do not invent dates.\n3. Your summary must cover a diverse range of sources from the list, not just one or two journals.\n4. The dossier must have these sections: Executive Summary, Deep Dive, Strategic Synthesis, and On the Horizon."
        ),
        HumanMessage(content=f"Here is the list of articles to analyze:\n\n{articles_text}"),
    ]
    
    result = model.invoke(messages)
    
    report = result.content
    return report

# --- TELEGRAM SENDER & MAIN FUNCTIONS (No changes needed) ---
async def send_report_to_telegram(report):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not found. Cannot send report.")
        return
    
    print("Sending report to Telegram...")
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        full_report = "## 📈 Daily Strategic Intelligence Dossier\n\n" + report
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
