#!/usr/bin/env python3
import os
import re
import sys
import logging
import argparse
from typing import List, Tuple, Dict, Any

import praw
from dotenv import load_dotenv
import google.generativeai as genai

# ─── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_POST_LIMIT = 20
DEFAULT_COMMENT_LIMIT = 20
MODEL_NAME = "gemini-1.5-flash"
PROMPT_TEMPLATE = """
Analyze the following Reddit user's posts and comments to generate a detailed user persona.
For each trait or insight, cite the source post or comment (full text or snippet) in the format: [source: ...].

User: {username}

Posts:
{posts_section}

Comments:
{comments_section}

Output format example:
User Persona: {username}

**Personality:** ... [source: ...]
**Interests:** ... [source: ...]
**Tone:** ... [source: ...]
**Values & Communication:** ... [source: ...]

Be concise but insightful.
"""


# ─── Logging Setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─── Core Functions ────────────────────────────────────────────────────────────

def extract_username(url: str) -> str:
    """Extract Reddit username from profile URL."""
    pattern = r"reddit\.com/user/([\w-]+)/?"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(f"Invalid Reddit profile URL: {url}")
    return match.group(1)


def load_reddit_instance() -> praw.Reddit:
    """Initialize and return a PRAW Reddit instance from environment."""
    load_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")
    if not all([client_id, client_secret, user_agent]):
        logging.error("Missing one of REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env")
        sys.exit(1)

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def fetch_user_content(
    reddit: praw.Reddit,
    username: str,
    post_limit: int,
    comment_limit: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch latest submissions and comments for a Reddit user."""
    posts, comments = [], []
    try:
        user = reddit.redditor(username)
        for sub in user.submissions.new(limit=post_limit):
            posts.append({"title": sub.title, "body": sub.selftext})
        for com in user.comments.new(limit=comment_limit):
            comments.append({"body": com.body})
        logging.info(f"Fetched {len(posts)} posts and {len(comments)} comments for u/{username}.")
    except Exception as e:
        logging.exception(f"Error fetching content for u/{username}: {e}")
    return posts, comments


def build_prompt(
    posts: List[Dict[str, str]],
    comments: List[Dict[str, str]],
    username: str,
) -> str:
    """Assemble the LLM prompt using posts and comments."""
    def section(items: List[Dict[str, str]], key_names: List[str]) -> str:
        lines = []
        for item in items:
            snippet = " ".join(item.get(k, "") for k in key_names if item.get(k))
            clean = snippet.replace("\n", " ").strip()
            # Truncate to avoid overly long prompt
            if len(clean) > 300:
                clean = clean[:300].rsplit(" ", 1)[0] + "…"
            lines.append(f"- {clean}")
        return "\n".join(lines)

    posts_section = section(posts, ["title", "body"])
    comments_section = section(comments, ["body"])
    return PROMPT_TEMPLATE.format(
        username=username,
        posts_section=posts_section,
        comments_section=comments_section,
    )


def call_gemini_api(prompt: str) -> str:
    """Send prompt to Gemini API and return the response text."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY not set in environment.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.exception(f"Gemini API call failed: {e}")
        sys.exit(1)


def post_process_persona(text: str) -> str:
    """Ensure headings like 'Personality:', 'Interests:', etc. are bolded."""
    def repl(match):
        return f"**{match.group(1)}:**"
    return re.sub(r"^([A-Za-z &]+):", repl, text, flags=re.MULTILINE)


def save_persona(username: str, persona: str) -> None:
    """Write persona text to a file."""
    filename = f"{username}_persona.txt"
    try:
        persona = post_process_persona(persona)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(persona)
        logging.info(f"Persona saved to {filename}.")
    except IOError as e:
        logging.exception(f"Failed to write persona file: {e}")


def run(username: str, post_limit: int, comment_limit: int) -> None:
    """Orchestrate data fetching, LLM calling, and saving."""
    reddit = load_reddit_instance()
    posts, comments = fetch_user_content(reddit, username, post_limit, comment_limit)
    if not posts and not comments:
        logging.error(f"No content found for u/{username}. Exiting.")
        sys.exit(1)

    prompt = build_prompt(posts, comments, username)
    logging.info("Calling Gemini API...")
    persona = call_gemini_api(prompt)
    save_persona(username, persona)


# ─── CLI Entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Reddit user persona using Gemini 1.5 Flash."
    )
    parser.add_argument("url", help="Reddit profile URL (e.g., https://reddit.com/user/kojied/)")
    parser.add_argument("--posts", type=int, default=DEFAULT_POST_LIMIT, help="Number of posts to fetch")
    parser.add_argument("--comments", type=int, default=DEFAULT_COMMENT_LIMIT, help="Number of comments to fetch")
    args = parser.parse_args()

    try:
        user = extract_username(args.url)
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    run(user, args.posts, args.comments)


if __name__ == "__main__":
    main()
