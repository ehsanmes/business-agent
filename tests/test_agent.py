import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import agent


def test_entry_datetime_is_utc():
    entry = SimpleNamespace(published_parsed=time.struct_time((2026, 10, 20, 12, 0, 0, 0, 293, 0)))
    assert agent.entry_datetime(entry) == datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def test_entry_datetime_falls_back_to_updated():
    entry = SimpleNamespace(updated_parsed=time.struct_time((2026, 10, 20, 12, 0, 0, 0, 293, 0)))
    assert agent.entry_datetime(entry) is not None
    assert agent.entry_datetime(SimpleNamespace()) is None


def test_entry_datetime_from_sciencedirect_description():
    entry = SimpleNamespace(summary="<p>Publication date: Available online 20 October 2026</p><p>Source: Research Policy</p>")
    assert agent.entry_datetime(entry) == datetime(2026, 10, 20, tzinfo=timezone.utc)
    assert agent.entry_datetime(SimpleNamespace(summary="<p>Publication date: December 2026</p>")) is None


def test_get_recent_articles_filters_and_dedupes(monkeypatch):
    now = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    fresh = (now - timedelta(hours=2)).timetuple()
    old = (now - timedelta(days=5)).timetuple()
    entries = [
        SimpleNamespace(title="New", link="https://a/1", summary="<p>Hi &amp; bye</p>", published_parsed=fresh),
        SimpleNamespace(title="Dup", link="https://a/1", summary="", published_parsed=fresh),
        SimpleNamespace(title="Old", link="https://a/2", summary="", published_parsed=old),
        SimpleNamespace(title="Undated", link="https://a/3", summary=""),
    ]
    monkeypatch.setattr(agent.feedparser, "parse", lambda url, agent=None: FakeFeed(entries=entries))

    articles = agent.get_recent_articles({"J1": {"url": "u1", "tag": "J1"}, "J2": {"url": "u2", "tag": "J2"}}, now=now)

    assert [a["title"] for a in articles] == ["New"]
    assert articles[0]["summary"] == "Hi & bye"


class FakeFeed(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def test_analyze_articles_escapes_and_numbers(monkeypatch):
    monkeypatch.setattr(agent, "API_DELAY_SECONDS", 0)
    articles = [
        {"journal": "J", "tag": "J", "title": "A", "link": "https://x/?a=1&b=2", "summary": ""},
        {"journal": "J", "tag": "J", "title": "B", "link": "https://x/b", "summary": ""},
        {"journal": "K", "tag": "K", "title": "C <new>", "link": "https://x/c", "summary": ""},
    ]
    client = FakeClient([
        "SUMMARY: R&D <matters>\nTAKEAWAY: Invest now.",
        RuntimeError("boom"),
        "No labels here",
        "OVERVIEW: Big picture.\nTHEMES:\n- AI\n- Teams\n- Cost",
    ])

    blocks = agent.analyze_articles(articles, client)
    report = "\n".join(blocks)

    assert "<blockquote expandable>R&amp;D &lt;matters&gt;</blockquote>" in report
    assert "💡 <b>Takeaway:</b> Invest now." in report
    assert 'href="https://x/?a=1&amp;b=2"' in report
    assert "<b>2. <a href=\"https://x/c\">C &lt;new&gt;</a></b>" in report
    assert "<blockquote expandable>No labels here</blockquote>" in report
    assert "<blockquote>Big picture.</blockquote>" in report
    assert "• AI\n• Teams\n• Cost" in report
    assert "#K" in report
    assert "1 article(s) could not be analyzed" in report
    assert "<b>2</b> articles from <b>2</b> sources" in report


def test_parse_sections_handles_bold_labels_and_preamble():
    sections = agent.parse_sections("Intro\n**SUMMARY:** text\n**Takeaway**: do it", ["SUMMARY", "TAKEAWAY"])
    assert sections == {"summary": "Intro text", "takeaway": "do it"}


def test_seconds_until_publish():
    now = datetime(2026, 10, 21, 16, 0, tzinfo=timezone.utc)
    assert agent.seconds_until_publish(now, "17:30") == 90 * 60
    assert agent.seconds_until_publish(now, "15:00") == 0
    assert agent.seconds_until_publish(now, "") == 0


def test_analyze_articles_without_articles():
    blocks = agent.analyze_articles([], client=None)
    assert "No new articles" in "\n".join(blocks)


def test_split_into_messages_respects_limit_and_tags():
    blocks = [f"<b>Item {i}</b> " + "x" * 50 for i in range(10)]
    messages = agent.split_into_messages(blocks, limit=200)
    assert all(len(m) <= 200 for m in messages)
    for m in messages:
        assert m.count("<b>") == m.count("</b>")
    assert "\n\n".join(messages) == "\n\n".join(blocks)


def test_split_into_messages_handles_oversized_block():
    messages = agent.split_into_messages(["<b>" + "&" * 500 + "</b>"], limit=100)
    assert all(len(m) <= 100 for m in messages)
    assert all("<b>" not in m for m in messages)
