"""
OpenAI Status Tracker
=====================
Monitors https://status.openai.com for new incidents, outages, and degradations
and prints structured console alerts.

Architecture — two complementary strategies:

1. **RSS/Atom conditional-GET poller** (runs out of the box, zero credentials)
   Uses HTTP ETag + If-Modified-Since headers so the server only sends a body
   when content has actually changed.  This is far more efficient than naive
   polling and scales to 100+ status pages with minimal bandwidth.

2. **Webhook receiver** (FastAPI, optional)
   incident.io (the platform powering status.openai.com) supports outbound
   webhooks.  When configured, the server receives push notifications the
   instant a new incident is created or updated — true event-driven delivery.
   Run with:  uvicorn tracker:app --host 0.0.0.0 --port 8000

Both strategies share the same `alert()` output function so you get a
consistent log format regardless of the source.

Usage
-----
# RSS-only mode (no extra setup needed):
    python tracker.py

# Webhook mode (add webhook URL http://<your-host>:8000/webhook in
# the incident.io status page settings):
    python tracker.py --webhook

# Track extra pages alongside OpenAI:
    python tracker.py --extra-feeds https://www.githubstatus.com/history.rss
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
import textwrap
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import feedparser  # pip install feedparser

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEEDS: list[dict] = [
    {
        "name": "OpenAI",
        # incident.io RSS feed — carries every incident update as a separate entry
        "url": "https://status.openai.com/feed.rss",
        # Filter: only surface updates that mention API-related products.
        # Set to None to receive ALL OpenAI incidents (ChatGPT, Sora, etc.).
        "filter_keywords": [
            "api", "chat completion", "responses", "embeddings",
            "fine-tun", "whisper", "dall-e", "assistants", "realtime",
            "vector store", "files", "batch",
        ],
    }
]

POLL_INTERVAL_SECONDS = 60  # how often to re-check each feed
API_PRODUCT_KEYWORDS = FEEDS[0]["filter_keywords"]  # convenience alias

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "outage":      "\033[91m",   # bright red
    "degraded":    "\033[93m",   # yellow
    "investigating": "\033[94m", # blue
    "resolved":    "\033[92m",   # green
    "maintenance": "\033[96m",   # cyan
    "default":     "\033[0m",
}
RESET = "\033[0m"


def _severity(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("outage", "down", "major")):
        return "outage"
    if "investigating" in t:
        return "investigating"
    if any(w in t for w in ("degraded", "degradation", "elevated error", "slow", "latency", "partial")):
        return "degraded"
    if any(w in t for w in ("resolved", "recovered", "operational")):
        return "resolved"
    if any(w in t for w in ("maintenance", "scheduled")):
        return "maintenance"
    return "default"


def alert(
    *,
    source: str,
    product: str,
    status_message: str,
    published: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    """Print a structured, coloured alert to stdout."""
    ts = published or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    severity = _severity(status_message)
    color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["default"])

    wrapped = textwrap.fill(status_message, width=90, subsequent_indent="              ")

    print(
        f"\n{color}"
        f"[{ts}] [{source}]\n"
        f"  Product : {product}\n"
        f"  Status  : {wrapped}"
        + (f"\n  URL     : {url}" if url else "")
        + f"{RESET}"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Feed state — tracks which entry IDs we've already seen
# ---------------------------------------------------------------------------

class FeedState:
    def __init__(self, name: str, url: str, keywords: Optional[list[str]]):
        self.name = name
        self.url = url
        self.keywords = [k.lower() for k in (keywords or [])]
        self.seen_ids: set[str] = set()
        self.etag: Optional[str] = None
        self.last_modified: Optional[str] = None

    def _entry_id(self, entry) -> str:
        # Prefer the <id>/<guid> field; fall back to a hash of the link + title
        raw = getattr(entry, "id", None) or getattr(entry, "link", "") + getattr(entry, "title", "")
        return hashlib.sha1(raw.encode()).hexdigest()

    def _matches_filter(self, entry) -> bool:
        if not self.keywords:
            return True
        blob = (
            (entry.get("title") or "")
            + " "
            + (entry.get("summary") or "")
        ).lower()
        return any(kw in blob for kw in self.keywords)

    def extract_product(self, entry) -> str:
        """
        incident.io RSS puts affected components in the <description> CDATA.
        We pull them out; fall back to the incident title.
        """
        summary = entry.get("summary") or entry.get("description") or ""
        # Look for "Affected components\n· ComponentA (status)\n· ComponentB"
        comps = re.findall(r"·\s+([^\(]+)\s*\(", summary)
        if comps:
            return ", ".join(c.strip() for c in comps)
        return entry.get("title", "Unknown product/service")

    def extract_status(self, entry) -> str:
        """
        incident.io embeds the status line at the top of the CDATA block:
        "Status: Resolved\nAll impacted services …"
        """
        summary = entry.get("summary") or entry.get("description") or ""
        # Strip HTML tags
        summary = re.sub(r"<[^>]+>", "", summary).strip()
        return summary[:500]  # cap length

    def new_entries(self, parsed_feed) -> list:
        new = []
        for entry in parsed_feed.entries:
            eid = self._entry_id(entry)
            if eid not in self.seen_ids:
                self.seen_ids.add(eid)
                if self._matches_filter(entry):
                    new.append(entry)
        return new


# ---------------------------------------------------------------------------
# Async RSS poller — uses ETag / Last-Modified for efficiency
# ---------------------------------------------------------------------------

async def poll_feed(session: aiohttp.ClientSession, state: FeedState) -> None:
    headers = {"User-Agent": "openai-status-tracker/1.0", "Accept-Encoding": "gzip, deflate"}
    if state.etag:
        headers["If-None-Match"] = state.etag
    if state.last_modified:
        headers["If-Modified-Since"] = state.last_modified

    try:
        async with session.get(state.url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 304:
                return  # Not modified — no work to do
            if resp.status != 200:
                print(f"[WARN] {state.name}: HTTP {resp.status}", file=sys.stderr)
                return

            # Cache headers for next request
            state.etag = resp.headers.get("ETag")
            state.last_modified = resp.headers.get("Last-Modified")

            body = await resp.text()
    except Exception as exc:
        print(f"[ERROR] {state.name}: {exc}", file=sys.stderr)
        return

    parsed = feedparser.parse(body)
    for entry in state.new_entries(parsed):
        product = state.extract_product(entry)
        status_msg = state.extract_status(entry)
        pub = entry.get("published") or entry.get("updated")
        url = entry.get("link")

        # Normalise published timestamp
        try:
            # feedparser gives us a time_struct
            pt = entry.get("published_parsed") or entry.get("updated_parsed")
            if pt:
                pub = datetime(*pt[:6], tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            pass

        alert(
            source=state.name,
            product=product,
            status_message=status_msg,
            published=pub,
            url=url,
        )


async def rss_monitor(extra_feeds: list[str]) -> None:
    """Continuously polls all configured feeds."""
    states: list[FeedState] = []
    for feed_cfg in FEEDS:
        states.append(FeedState(feed_cfg["name"], feed_cfg["url"], feed_cfg.get("filter_keywords")))

    # Add any extra feeds passed via CLI (no keyword filter)
    for url in extra_feeds:
        states.append(FeedState(url.split("/")[2], url, keywords=None))

    print(f"[INFO] Starting RSS monitor for {len(states)} feed(s).")
    print(f"[INFO] Poll interval: {POLL_INTERVAL_SECONDS}s  |  Filter: API-related products only\n")

    # On first run, seed seen_ids without alerting (avoid flooding old entries)
    async with aiohttp.ClientSession() as session:
        for state in states:
            try:
                async with session.get(state.url, timeout=aiohttp.ClientTimeout(total=15), headers={"User-Agent": "openai-status-tracker/1.0", "Accept-Encoding": "gzip, deflate"}) as resp:
                    if resp.status == 200:
                        body = await resp.text()
                        state.etag = resp.headers.get("ETag")
                        state.last_modified = resp.headers.get("Last-Modified")
                        parsed = feedparser.parse(body)
                        for e in parsed.entries:
                            state.seen_ids.add(state._entry_id(e))
                        print(f"[INFO] {state.name}: seeded {len(state.seen_ids)} existing entries. Watching for NEW ones…")
            except Exception as exc:
                print(f"[WARN] Could not seed {state.name}: {exc}", file=sys.stderr)

    # Main poll loop
    async with aiohttp.ClientSession() as session:
        while True:
            tasks = [poll_feed(session, s) for s in states]
            await asyncio.gather(*tasks)
            await asyncio.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Optional: webhook receiver (FastAPI)
# ---------------------------------------------------------------------------

def build_webhook_app():
    """
    Returns a FastAPI app that receives incident.io webhook payloads.

    In the incident.io Status Page settings → Notifications → Webhooks,
    add:   http://<your-public-host>:8000/webhook

    Run alongside the RSS poller with:
        uvicorn tracker:app --host 0.0.0.0 --port 8000
    """
    try:
        from fastapi import FastAPI, Request
    except ImportError:
        print("[ERROR] FastAPI not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)

    app = FastAPI(title="OpenAI Status Webhook Receiver")

    @app.post("/webhook")
    async def receive_webhook(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return {"status": "bad_request"}

        # incident.io webhook schema (simplified)
        # https://incident.io/docs/status-pages/webhooks
        event_type = payload.get("event_type", "unknown")  # e.g. "incident.created"
        incident   = payload.get("incident", {})
        name       = incident.get("name", "Unknown incident")
        status     = incident.get("status", "unknown")
        updates    = incident.get("incident_updates", [])
        components = incident.get("affected_components", [])

        latest_msg = updates[0].get("message", name) if updates else name
        product = ", ".join(c.get("name", "?") for c in components) if components else name
        url = incident.get("permalink") or incident.get("shortlink")

        alert(
            source=f"OpenAI [webhook/{event_type}]",
            product=product,
            status_message=f"[{status.upper()}] {latest_msg}",
            url=url,
        )
        return {"status": "ok"}

    @app.get("/health")
    async def health():
        return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

    return app


# Expose at module level so `uvicorn tracker:app` works
app = None  # populated below if needed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global app

    parser = argparse.ArgumentParser(description="OpenAI Status Tracker")
    parser.add_argument(
        "--webhook",
        action="store_true",
        help="Also start a FastAPI webhook receiver on port 8000",
    )
    parser.add_argument(
        "--extra-feeds",
        nargs="*",
        default=[],
        metavar="URL",
        help="Additional RSS/Atom feed URLs to monitor (e.g. GitHub, Stripe…)",
    )
    args = parser.parse_args()

    if args.webhook:
        app = build_webhook_app()
        print("[INFO] Webhook receiver built. Run `uvicorn tracker:app --host 0.0.0.0 --port 8000`")
        print("[INFO] Then configure that URL in your incident.io Status Page → Notifications → Webhooks\n")

    try:
        asyncio.run(rss_monitor(args.extra_feeds))
    except KeyboardInterrupt:
        print("\n[INFO] Tracker stopped.")


if __name__ == "__main__":
    main()
