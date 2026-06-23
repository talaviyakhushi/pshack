import os
import re
import html
import json
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


def strip_html(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw).strip()


# ── Rocketlane helpers ─────────────────────────────────────────────────────────

async def update_task_note(task_id: str, note: str):
    async with httpx.AsyncClient() as client:
        url = f"{RL_BASE}/tasks/{task_id}"
        r = await client.put(url, headers=RL_HEADERS, json={"taskPrivateNote": note})
        logger.info("PUT %s → %s", url, r.status_code)
        if r.status_code not in (200, 201, 204):
            logger.error("update_task_note failed: %s", r.text)
        return r


async def create_subtask(project_id: str, parent_task_id: str, name: str, description: str, assignee_user_id: int):
    body = {
        "taskName": name,
        "project": {"projectId": int(project_id)},
        "parent": {"taskId": int(parent_task_id)},
        "taskDescription": description,
        "assignees": {"members": [{"userId": assignee_user_id}]}
    }
    async with httpx.AsyncClient() as client:
        url = f"{RL_BASE}/tasks"
        r = await client.post(url, headers=RL_HEADERS, json=body)
        logger.info("POST %s → %s: %s", url, r.status_code, r.text[:300])
        return r


async def get_project_members(project_id: str) -> list:
    async with httpx.AsyncClient() as client:
        url = f"{RL_BASE}/projects/{project_id}"
        r = await client.get(url, headers=RL_HEADERS)
        logger.info("GET %s → %s", url, r.status_code)
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


def assign_to_members(extraction: str, members: list) -> list:
    member_list = "\n".join([
        f"- userId:{m.get('userId')} | {m.get('firstName', '')} {m.get('lastName', '')} | {m.get('emailId', '')}"
        for m in members
    ])

    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""You are assigning follow-up questions to project team members.

Extracted requirements:
{extraction}

Team members:
{member_list}

For each requirement block above, assign it to the most suitable team member based on their name/email.
Avoid assigning to "Rocketlane Admin" unless no one else fits.

Return a JSON array ONLY, no other text:
[
  {{
    "requirement": "<requirement summary>",
    "question": "<question to ask>",
    "team_type": "<team type>",
    "userId": <integer userId>,
    "memberName": "<first last>"
  }}
]"""
        }]
    )
    raw = message.content[0].text.strip()
    # extract JSON array from response
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    return json.loads(match.group())


# ── Core pipeline ──────────────────────────────────────────────────────────────

async def process_transcript(task_id: str, project_id: str, description: str):
    transcript = strip_html(description)

    if len(transcript) < 50:
        logger.info("Task %s: description too short, skipping.", task_id)
        return

    extraction = extract_requirements(transcript)

    if "NO_ACTION" in extraction:
        await update_task_note(task_id, "<p>Transcript processed — no actionable requirements found.</p>")
        return

    members = await get_project_members(project_id)
    if not members:
        await update_task_note(task_id, f"<p>Requirements extracted but could not fetch project members.</p><pre>{html.escape(extraction)}</pre>")
        return

    assignments = assign_to_members(extraction, members)
    if not assignments:
        await update_task_note(task_id, f"<p>Could not parse assignments.</p><pre>{html.escape(extraction)}</pre>")
        return

    # Create one subtask per assignment
    created = []
    for a in assignments:
        subtask_name = f"[AI] {a.get('team_type', 'Follow-up')}: {a.get('requirement', '')[:60]}"
        subtask_desc = f"<p><strong>Question for {html.escape(a.get('memberName', ''))}:</strong></p><p>{html.escape(a.get('question', ''))}</p>"
        r = await create_subtask(project_id, task_id, subtask_name, subtask_desc, a["userId"])
        if r.status_code in (200, 201):
            created.append(f"{a.get('memberName')} — {a.get('requirement', '')[:60]}")
            logger.info("Subtask created for userId %s", a["userId"])
        else:
            logger.error("Subtask creation failed for userId %s: %s", a["userId"], r.text)

    summary = "<h3>AI Follow-up Tasks Created</h3><ul>" + \
        "".join(f"<li>{html.escape(c)}</li>" for c in created) + "</ul>"
    await update_task_note(task_id, summary)
    logger.info("Task %s: %d subtasks created.", task_id, len(created))


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
