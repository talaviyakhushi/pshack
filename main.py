import os
import httpx
import asyncio
from fastapi import FastAPI, Request, BackgroundTasks
from anthropic import Anthropic

app = FastAPI()
claude = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

ROCKETLANE_API_KEY = os.environ.get("ROCKETLANE_API_KEY")
RL_BASE = "https://api.rocketlane.com/api/1.0"
RL_HEADERS = {
    "api-key": ROCKETLANE_API_KEY,
    "Content-Type": "application/json"
}


# ── Rocketlane helpers ─────────────────────────────────────────────────────────

async def get_task_attachments(task_id: str) -> list:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{RL_BASE}/tasks/{task_id}/attachments", headers=RL_HEADERS)
        r.raise_for_status()
        return r.json().get("data", [])


async def download_attachment(url: str) -> str:
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=RL_HEADERS)
        r.raise_for_status()
        return r.text


async def get_project_members(project_id: str) -> list:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{RL_BASE}/projects/{project_id}", headers=RL_HEADERS)
        r.raise_for_status()
        data = r.json()
        return data.get("teamMembers", {}).get("members", [])


async def post_comment(task_id: str, body: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{RL_BASE}/tasks/{task_id}/comments",
            headers=RL_HEADERS,
            json={"body": body}
        )
        r.raise_for_status()
        return r.json()


async def get_comments(task_id: str) -> list:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{RL_BASE}/tasks/{task_id}/comments", headers=RL_HEADERS)
        r.raise_for_status()
        return r.json().get("data", [])


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


def generate_brief(requirements_and_answers: str) -> str:
    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""You are helping an Implementation Manager prepare for their next client call.

Based on the following requirements and expert answers, write a concise brief (3-5 sentences) the IM can use going into the next call. Be direct and actionable.

{requirements_and_answers}

Brief:"""
        }]
    )
    return message.content[0].text


# ── Core pipeline ──────────────────────────────────────────────────────────────

async def process_transcript(task_id: str, project_id: str):
    # 1. Get transcript attachment
    attachments = await get_task_attachments(task_id)
    if not attachments:
        await post_comment(task_id, "⚠️ No attachments found. Please upload the transcript file and mark Done again.")
        return

    transcript_url = attachments[0].get("url")
    transcript = await download_attachment(transcript_url)

    # 2. Extract requirements via Claude
    extraction = extract_requirements(transcript)

    if "NO_ACTION" in extraction:
        await post_comment(task_id, "✅ Transcript processed — no actionable requirements found in this meeting.")
        return

    # 3. Get project members
    members = await get_project_members(project_id)
    member_list = "\n".join([
        f"- {m['firstName']} {m['lastName']} ({m['emailId']})"
        for m in members
    ])

    # 4. Ask IM to assign teammates
    comment_body = f"""📋 **Transcript processed. Here's what I found:**

{extraction}

---
**Please reply telling me who handles each team type above.**
Current project members:
{member_list}

Example reply format:
Legal → sarah@company.com
Engineering → ravi@company.com"""

    await post_comment(task_id, comment_body)

    # 5. Poll for IM's assignment reply
    await poll_for_assignments(task_id, extraction)


async def poll_for_assignments(task_id: str, extraction: str, max_attempts: int = 20):
    initial_count = len(await get_comments(task_id))

    for _ in range(max_attempts):
        await asyncio.sleep(30)
        comments = await get_comments(task_id)
        if len(comments) > initial_count:
            new_comment = comments[-1]["body"]
            await handle_assignments(task_id, extraction, new_comment)
            return

    await post_comment(task_id, "⏰ Timed out waiting for assignments. Please reply and I'll continue.")


async def handle_assignments(task_id: str, extraction: str, assignment_reply: str):
    requirements = extraction.strip().split("---")
    posted_questions = []

    for req in requirements:
        if "REQUIREMENT:" not in req:
            continue

        req_data = {}
        for line in req.strip().split("\n"):
            if line.startswith("REQUIREMENT:"):
                req_data["requirement"] = line.replace("REQUIREMENT:", "").strip()
            elif line.startswith("TEAM_TYPE:"):
                req_data["team_type"] = line.replace("TEAM_TYPE:", "").strip()
            elif line.startswith("QUESTION:"):
                req_data["question"] = line.replace("QUESTION:", "").strip()

        if not req_data:
            continue

        # Match team type to assigned email from IM's reply
        assigned_email = None
        for line in assignment_reply.split("\n"):
            if req_data.get("team_type", "").lower() in line.lower() and "→" in line:
                assigned_email = line.split("→")[-1].strip()
                break

        if assigned_email:
            comment = f"@{assigned_email}\n\n**Re: {req_data['requirement']}**\n\n{req_data['question']}"
        else:
            comment = f"**{req_data['requirement']}** (Team: {req_data.get('team_type')})\n\n{req_data['question']}"

        await post_comment(task_id, comment)
        posted_questions.append(req_data)

    comment_count = len(await get_comments(task_id))
    await poll_for_answers(task_id, posted_questions, comment_count)


async def poll_for_answers(task_id: str, questions: list, baseline_count: int, max_attempts: int = 40):
    for _ in range(max_attempts):
        await asyncio.sleep(30)
        comments = await get_comments(task_id)
        new_comments = comments[baseline_count:]

        if len(new_comments) >= len(questions):
            answers_text = "\n\n".join([
                f"Q: {q['question']}\nA: {new_comments[i]['body']}"
                for i, q in enumerate(questions)
                if i < len(new_comments)
            ])
            brief = generate_brief(answers_text)
            await post_comment(task_id, f"✅ **Brief ready for your next call:**\n\n{brief}")
            return

    await post_comment(task_id, "⏰ Still waiting on some answers. I'll post the brief once everyone has replied.")


# ── Webhook endpoint ───────────────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    event_type = payload.get("eventType")

    task = payload.get("data", {}).get("task", {})
    task_id = str(task.get("taskId", ""))
    project_id = str(task.get("project", {}).get("projectId", ""))

    if not task_id or not project_id:
        return {"status": "ignored"}

    if event_type == "TASK_CREATED":
        background_tasks.add_task(process_transcript, task_id, project_id)
        return {"status": "processing"}

    if event_type == "TASK_UPDATED":
        changed_fields = payload.get("changeLog", {}).get("changedFields", [])
        new_status = payload.get("changeLog", {}).get("to", {}).get("status", {}).get("label", "")
        if "status" in changed_fields and new_status.lower() == "done":
            background_tasks.add_task(process_transcript, task_id, project_id)
            return {"status": "processing"}

    return {"status": "ignored"}


@app.get("/health")
async def health():
    return {"status": "ok"}
