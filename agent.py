import os
import re
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

# Feed URL patterns: Wiley "feed/<eISSN>/most-recent", Elsevier "publication/science/<ISSN>",
# AOM/INFORMS/SAGE "showFeed?...&jc=<journal code>".
# "tag" becomes a Telegram hashtag, so it may only contain letters, digits and "_".
JOURNAL_FEEDS = {
    # Practitioner reviews and business-school research outlets
    "Harvard Business Review": {"url": "http://feeds.harvardbusiness.org/harvardbusiness/", "tag": "HBR"},
    "MIT Sloan Management Review": {"url": "https://sloanreview.mit.edu/feed/", "tag": "MITSloan"},
    "California Management Review": {"url": "https://journals.sagepub.com/action/showFeed?jc=cmr&type=etoc&feed=rss", "tag": "CMR"},
    "Knowledge at Wharton": {"url": "https://knowledge.wharton.upenn.edu/feed/", "tag": "Wharton"},
    "INSEAD Knowledge": {"url": "https://knowledge.insead.edu/rss.xml", "tag": "INSEAD"},
    "McKinsey": {"url": "https://www.mckinsey.com/rss", "tag": "McKinsey"},
    "Deloitte Insights": {"url": "https://deloitteuniversitypress.libsyn.com/rss", "tag": "Deloitte"},
    # Academy of Management
    "Academy of Management Review": {"url": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=amr", "tag": "AMR"},
    "Academy of Management Journal": {"url": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=amj", "tag": "AMJ"},
    "Academy of Management Annals": {"url": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=annals", "tag": "AMAnnals"},
    "Academy of Management Perspectives": {"url": "https://journals.aom.org/action/showFeed?type=etoc&feed=rss&jc=amp", "tag": "AMP"},
    # INFORMS
    "Organization Science": {"url": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=orsc", "tag": "OrgSci"},
    "Management Science": {"url": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=mnsc", "tag": "ManSci"},
    "Strategy Science": {"url": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=stsc", "tag": "StrategySci"},
    "Information Systems Research": {"url": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=isre", "tag": "ISR"},
    "Marketing Science": {"url": "https://pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc=mksc", "tag": "MktSci"},
    # SAGE
    "Administrative Science Quarterly": {"url": "https://journals.sagepub.com/action/showFeed?jc=asqa&type=etoc&feed=rss", "tag": "ASQ"},
    "Journal of Management": {"url": "https://journals.sagepub.com/action/showFeed?jc=joma&type=etoc&feed=rss", "tag": "JOM"},
    "Journal of Marketing": {"url": "https://journals.sagepub.com/action/showFeed?jc=jmxa&type=etoc&feed=rss", "tag": "JM"},
    # Wiley
    "Strategic Management Journal": {"url": "https://onlinelibrary.wiley.com/feed/10970266/most-recent", "tag": "SMJ"},
    "Journal of Management Studies": {"url": "https://onlinelibrary.wiley.com/feed/14676486/most-recent", "tag": "JMS"},
    "Strategic Entrepreneurship Journal": {"url": "https://onlinelibrary.wiley.com/feed/1932443x/most-recent", "tag": "SEJ"},
    "Journal of Product Innovation Management": {"url": "https://onlinelibrary.wiley.com/feed/15405885/most-recent", "tag": "JPIM"},
    "Journal of Operations Management": {"url": "https://onlinelibrary.wiley.com/feed/18731317/most-recent", "tag": "JOpsM"},
    # Elsevier
    "Research Policy": {"url": "https://rss.sciencedirect.com/publication/science/00487333", "tag": "ResearchPolicy"},
    "Journal of Business Venturing": {"url": "https://rss.sciencedirect.com/publication/science/08839026", "tag": "JBV"},
    "Long Range Planning": {"url": "https://rss.sciencedirect.com/publication/science/00246301", "tag": "LRP"},
    "Journal of World Business": {"url": "https://rss.sciencedirect.com/publication/science/10909516", "tag": "JWB"},
    "The Leadership Quarterly": {"url": "https://rss.sciencedirect.com/publication/science/10489843", "tag": "LeadershipQuarterly"},
    "Organizational Behavior and Human Decision Processes": {"url": "https://rss.sciencedirect.com/publication/science/07495978", "tag": "OBHDP"},
}
DAYS_TO_CHECK = int(os.environ.get("DAYS_TO_CHECK", "1"))
# Fixed on purpose: gpt-4o-mini keeps the cost low.
MODEL_TO_USE = "gpt-4o-mini"
# Pause between API calls to stay under the provider's rate limit.
API_DELAY_SECONDS = float(os.environ.get("API_DELAY_SECONDS", "10"))
# "HH:MM" in UTC. When set, the report is held until this time before being sent.
PUBLISH_AT_UTC = os.environ.get("PUBLISH_AT_UTC", "").strip()
# Fetch feeds and print what was found, without calling the model or Telegram.
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")
# Cap on article-summary length sent to the model, to keep token cost bounded.
MAX_INPUT_CHARS = 3000
FEED_TIMEOUT_SECONDS = 30
# Some publishers (SAGE, Wiley, INFORMS) reject feedparser's default user agent.
USER_AGENT = "Mozilla/5.0 (compatible; BusinessDossierBot/1.0; +https://t.me/Business_dossier)"
TELEGRAM_MAX_LENGTH = 4096
FOOTER = "━━━━━━━━━━━━━━━\n📌 @Business_dossier · #Dossier"


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
    for journal, source in feeds.items():
        try:
            feed = feedparser.parse(source["url"], agent=USER_AGENT)
            if not feed.entries:
                error = feed.get('bozo_exception') or f"HTTP {feed.get('status', '?')}"
                print(f"⚠️ {journal}: no entries ({error}).")
                continue

            recent = undated = 0
            for entry in feed.entries:
                published_time = entry_datetime(entry)
                if published_time is None:
                    undated += 1
                    continue
                if published_time < cutoff_date:
                    continue
                link = getattr(entry, 'link', '')
                if link in seen_links:
                    continue
                seen_links.add(link)
                recent += 1
                all_articles.append({
                    "journal": journal,
                    "tag": source["tag"],
                    "title": getattr(entry, 'title', 'Untitled').strip(),
                    "link": link,
                    "summary": BeautifulSoup(getattr(entry, 'summary', ''), 'html.parser').get_text(" ", strip=True),
                })
            print(f"✅ {journal}: {len(feed.entries)} entries, {recent} recent, {undated} undated.")
        except Exception as e:
            print(f"❌ {journal}: could not fetch or parse feed. Error: {e}")
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


def parse_sections(text, labels):
    """Split model output of the form "LABEL: ..." into a dict keyed by lower-case label.
    Text before the first label goes to the first label."""
    pattern = re.compile(rf"^\s*\**({'|'.join(labels)})\**\s*:\s*\**\s*",re.IGNORECASE | re.MULTILINE)
    sections = {label.lower(): "" for label in labels}
    matches = list(pattern.finditer(text))
    if not matches:
        sections[labels[0].lower()] = text.strip()
        return sections
    if matches[0].start() > 0:
        sections[labels[0].lower()] = text[:matches[0].start()].strip()
    for match, nxt in zip(matches, matches[1:] + [None]):
        body = text[match.end():nxt.start() if nxt else len(text)].strip()
        key = match.group(1).lower()
        sections[key] = f"{sections[key]} {body}".strip()
    return sections


def bullet_lines(text):
    lines = [re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
    return [line for line in lines if line]


def format_header(start_date, end_date, article_count, source_count):
    return (
        "📈 <b>Strategic Intelligence Dossier</b>\n"
        f"<i>{end_date.strftime('%A, %d %B %Y')}</i>\n\n"
        f"🗓 <b>Coverage:</b> {start_date.strftime('%d %b')} → {end_date.strftime('%d %b %Y')}\n"
        f"📚 <b>{article_count}</b> articles from <b>{source_count}</b> sources"
    )


def format_article(number, article, summary, takeaway):
    block = (
        f"<b>{number}. <a href=\"{html.escape(article['link'], quote=True)}\">{html.escape(article['title'])}</a></b>\n"
        f"🏛 <i>{html.escape(article['journal'])}</i> · #{article['tag']}\n"
        f"<blockquote expandable>{html.escape(summary)}</blockquote>"
    )
    if takeaway:
        block += f"\n💡 <b>Takeaway:</b> {html.escape(takeaway)}"
    return block


def format_executive_summary(overview, themes):
    block = f"🧭 <b>Executive Summary</b>\n<blockquote>{html.escape(overview)}</blockquote>"
    if themes:
        block += "\n\n🔑 <b>Key Themes</b>\n" + "\n".join(f"• {html.escape(t)}" for t in themes)
    return block


ARTICLE_PROMPT = (
    "You are a sharp business analyst writing for busy executives. Read the article details and reply in exactly this format, "
    "with no markdown:\n"
    "SUMMARY: 2-3 clear sentences (under 80 words) explaining the core argument or finding.\n"
    "TAKEAWAY: one sentence (under 25 words) on what a business leader should do or watch."
)

EXECUTIVE_PROMPT = (
    "You are a senior business strategist. Read the article summaries and reply in exactly this format, with no markdown:\n"
    "OVERVIEW: one cohesive paragraph (3-4 sentences) synthesizing the overarching themes for a busy executive.\n"
    "THEMES:\n- theme one in under 12 words\n- theme two\n- theme three"
)


def analyze_articles(articles, client, now=None):
    """Build the report as a list of HTML blocks; each block is safe to send on its own."""
    end_date = now or datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS_TO_CHECK)

    if not articles:
        return [
            format_header(start_date, end_date, 0, 0),
            f"📭 No new articles were published in the monitored journals in the last {DAYS_TO_CHECK} day(s).",
        ]

    print(f"Analyzing {len(articles)} articles one by one (Step 1: Deep Dives)...")

    deep_dive_parts = []
    analyzed = []

    for i, article in enumerate(articles):
        if i > 0:
            time.sleep(API_DELAY_SECONDS)
        print(f"Analyzing article {i+1}/{len(articles)}: {article['title']}")

        user_message = f"Journal: {article['journal']}\nTitle: {article['title']}\nAbstract: {article['summary'][:MAX_INPUT_CHARS]}"

        try:
            reply = ask_model(client, ARTICLE_PROMPT, user_message, max_tokens=250, temperature=0.5)
        except Exception as e:
            print(f"Could not analyze article '{article['title']}'. Error: {e}")
            continue

        sections = parse_sections(reply, ["SUMMARY", "TAKEAWAY"])
        deep_dive_parts.append(format_article(len(deep_dive_parts) + 1, article, sections["summary"], sections["takeaway"]))
        analyzed.append((article, sections["summary"]))

    failed_count = len(articles) - len(deep_dive_parts)
    if not deep_dive_parts:
        return [
            format_header(start_date, end_date, 0, 0),
            f"⚠️ Found {len(articles)} new articles but could not analyze any of them due to API errors.",
        ]

    time.sleep(API_DELAY_SECONDS)
    print("Generating Executive Summary (Step 2)...")
    all_summaries_text = "\n".join(f"- [{a['journal']}] {a['title']}: {s}" for a, s in analyzed)

    try:
        reply = ask_model(client, EXECUTIVE_PROMPT, all_summaries_text, max_tokens=600, temperature=0.7)
        sections = parse_sections(reply, ["OVERVIEW", "THEMES"])
        overview, themes = sections["overview"], bullet_lines(sections["themes"])[:5]
    except Exception as e:
        print(f"Could not generate executive summary. Error: {e}")
        overview, themes = "Could not generate executive summary due to an API error.", []

    print("Assembling final report...")

    source_count = len({a['journal'] for a, _ in analyzed})
    blocks = [
        format_header(start_date, end_date, len(deep_dive_parts), source_count),
        format_executive_summary(overview, themes),
        "📰 <b>Deep Dives</b>",
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


def seconds_until_publish(now, publish_at):
    """Seconds to wait until today's HH:MM (UTC); 0 if empty or already past."""
    if not publish_at:
        return 0
    hour, minute = (int(part) for part in publish_at.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return max(0, (target - now).total_seconds())


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
    if DRY_RUN:
        print("🧪 Dry run: fetching feeds only, nothing will be sent.")
        get_recent_articles(JOURNAL_FEEDS)
        return 0

    client = create_client()
    if client is None:
        print("Error: AvalAI client could not be initialized. Check API Key secret.")
        return 1

    articles = get_recent_articles(JOURNAL_FEEDS)
    report = analyze_articles(articles, client)

    wait = seconds_until_publish(datetime.now(timezone.utc), PUBLISH_AT_UTC)
    if wait:
        print(f"Report ready. Holding for {wait / 60:.0f} minutes until {PUBLISH_AT_UTC} UTC...")
        time.sleep(wait)

    sent = asyncio.run(send_to_telegram(report, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID))
    print("\n--- AGENT RUN FINISHED ---")
    # A non-zero exit makes the GitHub Actions run show as failed.
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
