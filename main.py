import os
import re
import html
import httpx
import logging
from fastapi import FastAPI, Request, BackgroundTasks
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
claude = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

ROCKETLANE_API_KEY = os.environ.get("ROCKETLANE_API_KEY")
RL_BASE = "https://api.rocketlane.com/api/1.0"
RL_HEADERS = {
    "api-key": ROCKETLANE_API_KEY,
    "Content-Type": "application/json"
}


def strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


# ── Rocketlane helpers ─────────────────────────────────────────────────────────

async def update_task_note(task_id: str, note: str):
    async with httpx.AsyncClient() as client:
        url = f"{RL_BASE}/tasks/{task_id}"
        r = await client.put(url, headers=RL_HEADERS, json={"taskPrivateNote": note})
        logger.info("PUT %s → %s: %s", url, r.status_code, r.text[:500])
        if r.status_code not in (200, 201, 204):
            logger.error("Failed to update task note: %s", r.text)
            return None
        return r.json()


async def get_project_members(project_id: str) -> list:
    async with httpx.AsyncClient() as client:
        url = f"{RL_BASE}/projects/{project_id}"
        r = await client.get(url, headers=RL_HEADERS)
        logger.info("GET %s → %s: %s", url, r.status_code, r.text[:300])
        if r.status_code != 200:
            return []
        return r.json().get("teamMembers", {}).get("members", [])


# ── Claude helpers ─────────────────────────────────────────────────────────────

def extract_requirements(transcript: str) -> str:
    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""You are helping an Implementation Manager process a client call transcript.

Read this transcript and extract any NEW client requirements or questions that need internal follow-up.

For each requirement:
1. Summarize it in one sentence
2. Identify what TYPE of team would own it (e.g. Legal, Engineering, Finance, Security, Networking, Solutions)
3. Write a clear, specific question to ask that team

Format your response EXACTLY like this for each requirement:
REQUIREMENT: <one sentence summary>
TEAM_TYPE: <team type>
QUESTION: <specific question to ask>
---

If there are no actionable requirements, respond with: NO_ACTION

Transcript:
{transcript}"""
        }]
    )
    return message.content[0].text


def generate_brief(extraction: str, members: list) -> str:
    member_list = "\n".join([
        f"- {m.get('firstName', '')} {m.get('lastName', '')} ({m.get('emailId', '')})".strip()
        for m in members
    ]) or "No project members found."

    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""You are helping an Implementation Manager after a client call.

Here are the requirements extracted from the transcript:

{extraction}

Project team members:
{member_list}

Write a concise action plan (3-5 sentences) for the IM:
- What needs follow-up and from which team
- Suggested owners from the member list above
- What to confirm before the next client call

Be direct and specific."""
        }]
    )
    return message.content[0].text


# ── Core pipeline ──────────────────────────────────────────────────────────────

async def process_transcript(task_id: str, project_id: str, description: str):
    transcript = strip_html(description)

    if len(transcript) < 50:
        logger.info("Task %s: description too short, skipping.", task_id)
        return

    extraction = extract_requirements(transcript)

    if "NO_ACTION" in extraction:
        await update_task_note(task_id, "<p>✅ Transcript processed — no actionable requirements found.</p>")
        return

    members = await get_project_members(project_id)
    brief = generate_brief(extraction, members)

    note = (
        "<h3>AI Brief - Transcript Analysis</h3>"
        f"<p><strong>Action Plan:</strong></p><p>{html.escape(brief)}</p>"
        f"<p><strong>Extracted Requirements:</strong></p><pre>{html.escape(extraction)}</pre>"
    )
    await update_task_note(task_id, note)
    logger.info("Task %s: brief written to private note.", task_id)


# ── Webhook endpoint ───────────────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    event_type = payload.get("eventType")

    task = payload.get("data", {}).get("task", {})
    task_id = str(task.get("taskId", ""))
    project_id = str(task.get("project", {}).get("projectId", ""))
    description = task.get("taskDescription", "")

    if not task_id or not project_id:
        return {"status": "ignored"}

    if event_type == "TASK_CREATED":
        background_tasks.add_task(process_transcript, task_id, project_id, description)
        return {"status": "processing"}

    if event_type == "TASK_UPDATED":
        changed_fields = payload.get("changeLog", {}).get("changedFields", [])
        new_status = payload.get("changeLog", {}).get("to", {}).get("status", {}).get("label", "")
        if "status" in changed_fields and new_status.lower() == "done":
            background_tasks.add_task(process_transcript, task_id, project_id, description)
            return {"status": "processing"}

    return {"status": "ignored"}


@app.get("/health")
async def health():
    return {"status": "ok"}
