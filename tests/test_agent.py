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
    monkeypatch.setattr(agent.feedparser, "parse", lambda url, agent=None: SimpleNamespace(entries=entries))

    articles = agent.get_recent_articles({"J1": "u1", "J2": "u2"}, now=now)

    assert [a["title"] for a in articles] == ["New"]
    assert articles[0]["summary"] == "Hi & bye"


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
        {"journal": "J", "title": "A", "link": "https://x/?a=1&b=2", "summary": ""},
        {"journal": "J", "title": "B", "link": "https://x/b", "summary": ""},
        {"journal": "J", "title": "C <new>", "link": "https://x/c", "summary": ""},
    ]
    client = FakeClient(["R&D <matters>", RuntimeError("boom"), "Third", "Exec"])

    blocks = agent.analyze_articles(articles, client)
    report = "\n".join(blocks)

    assert "R&amp;D &lt;matters&gt;" in report
    assert 'href="https://x/?a=1&amp;b=2"' in report
    assert "2. <b>C &lt;new&gt;</b>" in report
    assert "1 article(s) could not be analyzed" in report
    assert "<b>📊 Articles Reviewed:</b> 2" in report


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
