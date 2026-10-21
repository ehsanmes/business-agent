import os
import sys
import html
import time
import socket
import asyncio
import calendar
import feedparser
import telegram
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
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
# Values can be overridden with environment variables without editing the code.
DAYS_TO_CHECK = int(os.environ.get("DAYS_TO_CHECK", "1"))
MODEL_TO_USE = os.environ.get("MODEL_TO_USE", "gpt-4o-mini")
# Pause between API calls to stay under the provider's rate limit.
API_DELAY_SECONDS = float(os.environ.get("API_DELAY_SECONDS", "10"))
# Cap on article-summary length sent to the model, to keep token cost bounded.
MAX_INPUT_CHARS = 3000
FEED_TIMEOUT_SECONDS = 30
# Some publishers (SAGE, Wiley, INFORMS) reject feedparser's default user agent.
USER_AGENT = "Mozilla/5.0 (compatible; BusinessDossierBot/1.0; +https://t.me/Business_dossier)"
TELEGRAM_MAX_LENGTH = 4096
FOOTER = "@Business_dossier 🚀"


# --- 2. INITIALIZE THE AI CLIENT ---
def create_client():
    if not AVALAI_API_KEY:
        print("❌ AVALAI_API_KEY secret is not set.")
        return None
    try:
        client = OpenAI(
            api_key=AVALAI_API_KEY,
            base_url=AVALAI_BASE_URL,
            max_retries=3,
            timeout=60,
        )
        print("✅ AvalAI client configured successfully.")
        return client
    except Exception as e:
        print(f"❌ Critical error during client initialization: {e}")
        return None


# --- 3. FUNCTIONS ---
def entry_datetime(entry):
    """Return the entry's publish (or update) time as an aware UTC datetime, or None."""
    time_struct = getattr(entry, 'published_parsed', None) or getattr(entry, 'updated_parsed', None)
    if not time_struct:
        return None
    # feedparser normalizes parsed times to UTC.
    return datetime.fromtimestamp(calendar.timegm(time_struct), tz=timezone.utc)


def get_recent_articles(feeds, now=None):
    print("Fetching recent articles...")
    all_articles = []
    seen_links = set()
    now = now or datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=DAYS_TO_CHECK)
    socket.setdefaulttimeout(FEED_TIMEOUT_SECONDS)
    for journal, url in feeds.items():
        try:
            feed = feedparser.parse(url, agent=USER_AGENT)
            if not feed.entries:
                print(f"No entries found for {journal}.")
                continue

            skipped_undated = 0
            for entry in feed.entries:
                published_time = entry_datetime(entry)
                if published_time is None:
                    skipped_undated += 1
                    continue
                if published_time < cutoff_date:
                    continue
                link = getattr(entry, 'link', '')
                if link in seen_links:
                    continue
                seen_links.add(link)
                article = {
                    "journal": journal,
                    "title": getattr(entry, 'title', 'Untitled').strip(),
                    "link": link,
                    "summary": BeautifulSoup(getattr(entry, 'summary', ''), 'html.parser').get_text(" ", strip=True),
                }
                all_articles.append(article)
            if skipped_undated:
                print(f"Skipped {skipped_undated} undated entries in {journal}.")
        except Exception as e:
            print(f"Could not fetch or parse feed for {journal}. Error: {e}")
    print(f"Found {len(all_articles)} new articles across all feeds.")
    return all_articles


def ask_model(client, system_message, user_message, max_tokens, temperature):
    completion = client.chat.completions.create(
        model=MODEL_TO_USE,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()


def format_header(start_date, end_date, count):
    return (
        "<b>📈 Daily Strategic Intelligence Dossier</b>\n\n"
        f"<b>🗓️ Reporting Period:</b> {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
        f"<b>📊 Articles Reviewed:</b> {count}"
    )


def analyze_articles(articles, client, now=None):
    """Build the report as a list of HTML blocks; each block is safe to send on its own."""
    end_date = now or datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS_TO_CHECK)

    if not articles:
        return [
            format_header(start_date, end_date, 0),
            f"<b>Executive Summary</b>\nNo new articles were found in the monitored journals within the last {DAYS_TO_CHECK} days.",
        ]

    print(f"Analyzing {len(articles)} articles one by one (Step 1: Deep Dives)...")

    deep_dive_parts = []
    raw_summaries_for_exec_summary = []

    system_message = "You are a sharp business analyst. Summarize the following article in a clear, 3-4 sentence paragraph. Explain the core argument and its main takeaway for a business leader in less than 100 words. Do not use markdown or special characters."

    for i, article in enumerate(articles):
        if i > 0:
            time.sleep(API_DELAY_SECONDS)
        print(f"Analyzing article {i+1}/{len(articles)}: {article['title']}")

        user_message = f"Article:\n- Journal: {article['journal']}\n- Title: {article['title']}\n- Summary: {article['summary'][:MAX_INPUT_CHARS]}"

        try:
            single_summary = ask_model(client, system_message, user_message, max_tokens=250, temperature=0.5)
        except Exception as e:
            print(f"Could not analyze article '{article['title']}'. Error: {e}")
            continue

        number = len(deep_dive_parts) + 1
        link = html.escape(article['link'], quote=True)
        deep_dive_parts.append(
            f"{number}. <b>{html.escape(article['title'])}</b> <i>({html.escape(article['journal'])})</i>\n"
            f"{html.escape(single_summary)} <a href=\"{link}\">[Link]</a>"
        )
        raw_summaries_for_exec_summary.append(single_summary)

    failed_count = len(articles) - len(deep_dive_parts)
    if not deep_dive_parts:
        return [
            format_header(start_date, end_date, 0),
            f"⚠️ Found {len(articles)} new articles but could not analyze any of them due to API errors.",
        ]

    time.sleep(API_DELAY_SECONDS)
    print("Generating Executive Summary (Step 2)...")
    all_summaries_text = "\n".join(raw_summaries_for_exec_summary)

    system_message_exec = "You are a senior business strategist. Read the following list of individual article summaries and write one cohesive paragraph (3-5 sentences) that synthetically summarizes the main, overarching themes and trends for a busy executive. Do not use markdown or special characters."
    user_message_exec = f"Here are today's article summaries:\n{all_summaries_text}"

    try:
        executive_summary = ask_model(client, system_message_exec, user_message_exec, max_tokens=1024, temperature=0.7)
    except Exception as e:
        print(f"Could not generate executive summary. Error: {e}")
        executive_summary = "Could not generate executive summary due to an API error."

    print("Assembling final report...")

    blocks = [
        format_header(start_date, end_date, len(deep_dive_parts)),
        f"<b>Executive Summary</b>\n{html.escape(executive_summary)}",
        "<b>Deep Dives</b>",
        *deep_dive_parts,
    ]
    if failed_count:
        blocks.append(f"<i>⚠️ {failed_count} article(s) could not be analyzed.</i>")
    return blocks


def split_into_messages(blocks, limit=TELEGRAM_MAX_LENGTH):
    """Pack report blocks into messages under Telegram's length limit without
    cutting through an HTML tag."""
    messages = []
    current = ""
    for block in blocks:
        if len(block) > limit:
            # A single oversized block: fall back to plain text so cutting it is safe.
            # Escaping grows a character up to 5x ("&" -> "&amp;"), so cut before escaping.
            text = html_to_text(block)
            step = limit // 5
            pieces = [html.escape(text[i:i + step]) for i in range(0, len(text), step)]
        else:
            pieces = [block]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= limit:
                current = candidate
            else:
                messages.append(current)
                current = piece
    if current:
        messages.append(current)
    return messages


def html_to_text(message):
    return BeautifulSoup(message, 'html.parser').get_text()


async def send_to_telegram(blocks, token, chat_id):
    """Send the report; return True if every message was delivered."""
    if not token or not chat_id:
        print("Telegram secrets not found.")
        return False

    messages = split_into_messages([*blocks, FOOTER])

    print(f"Sending report to Telegram in {len(messages)} message(s)...")
    bot = telegram.Bot(token=token)
    all_sent = True
    async with bot:
        for message in messages:
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
            except telegram.error.BadRequest as e:
                # Fall back to plain text for this message only, so nothing is sent twice.
                print(f"HTML send failed ({e}). Sending this part as plain text...")
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=html_to_text(message),
                        disable_web_page_preview=True
                    )
                except Exception as e2:
                    print(f"Failed to send as plain text. Final error: {e2}")
                    all_sent = False
            except Exception as e:
                print(f"Failed to send report part. Error: {e}")
                all_sent = False
    if all_sent:
        print("Report successfully sent.")
    return all_sent


# --- 4. EXECUTION ---
def main():
    client = create_client()
    if client is None:
        print("Error: AvalAI client could not be initialized. Check API Key secret.")
        return 1

    articles = get_recent_articles(JOURNAL_FEEDS)
    report = analyze_articles(articles, client)

    sent = asyncio.run(send_to_telegram(report, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID))
    print("\n--- AGENT RUN FINISHED ---")
    # A non-zero exit makes the GitHub Actions run show as failed.
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
