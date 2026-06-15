import json

from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:3b"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


# ---------------------------------------------------------------------------
# 1. Explanation agent — SHAP values -> plain English
# ---------------------------------------------------------------------------
def explain_alert(txn: dict) -> str:
    reasons_text = "\n".join(
        f"- {r['feature']} (impact {r['impact']:+.3f})" for r in txn["top_reasons"]
    )
    prompt = f"""You are a fraud analyst assistant. Explain to a human investigator,
in 2-3 short sentences and plain English, why this transaction was flagged.

Transaction:
- amount: {txn['amount']}
- sender: {txn['nameOrig']}
- receiver: {txn['nameDest']}
- risk score: {txn['risk_score']} / 100

Top contributing model features (SHAP values, positive = increases fraud risk):
{reasons_text}

Write the explanation now, do not repeat the raw numbers, translate them into
plain language (e.g. "the account balance was fully drained", "this amount is
unusually high for this sender")."""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 2. Triage agent — for mid-risk scores (40-70)
# ---------------------------------------------------------------------------
def triage_decision(txn: dict, sender_history: list) -> dict:
    prompt = f"""You are a fraud triage agent. A transaction has a MEDIUM risk score
and needs a decision. Choose exactly one action: "step_up_auth", "escalate", or "clear".

Transaction:
- amount: {txn['amount']}
- sender: {txn['nameOrig']}
- receiver: {txn['nameDest']}
- risk score: {txn['risk_score']} / 100
- top reasons: {json.dumps(txn['top_reasons'])}

Sender's recent transaction history (last {len(sender_history)} txns):
{json.dumps(sender_history, indent=2)}

Respond ONLY with a JSON object in this exact format:
{{"action": "step_up_auth" | "escalate" | "clear", "reason": "<one sentence>"}}"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    content = resp.choices[0].message.content.strip()
    try:
        # strip markdown code fences if present
        if content.startswith("```"):
            content = content.strip("`").lstrip("json").strip()
        return json.loads(content)
    except json.JSONDecodeError:
        return {"action": "escalate", "reason": f"Could not parse agent response: {content}"}


# ---------------------------------------------------------------------------
# 3. Case summary agent — narrative report for escalated cases
# ---------------------------------------------------------------------------
def case_summary(txn: dict, sender_history: list, linked: dict) -> str:
    prompt = f"""You are a fraud case writer. Write a short case summary report
(4-6 sentences) for an investigator's audit trail, covering: what happened,
why it was flagged, the sender's recent activity, and any linked accounts of note.

Transaction under review:
{json.dumps(txn, indent=2)}

Sender's recent transaction history:
{json.dumps(sender_history, indent=2)}

Linked accounts (graph lookup):
{json.dumps(linked, indent=2)}

Write the report now in plain prose, no headings, no bullet points."""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 4. Investigator copilot — tool-calling chat agent
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_account_history",
            "description": "Get recent transactions for an account (as sender or receiver).",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "Account ID, e.g. C1305486145"},
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_linked_accounts",
            "description": "Get accounts linked to this account via send/receive transactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "Account ID, e.g. C1305486145"},
                },
                "required": ["account_id"],
            },
        },
    },
]


def copilot_chat(question: str, txn: dict, tool_executor) -> str:
    """
    tool_executor: callable(tool_name: str, args: dict) -> dict
    Resolves a tool call against the in-memory dataset (app.py supplies this).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an investigator copilot for a fraud detection dashboard. "
                "You have tools to look up account history and linked accounts. "
                "Use them when needed to answer the investigator's question. "
                f"The transaction currently under review is: {json.dumps(txn)}"
            ),
        },
        {"role": "user", "content": question},
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        temperature=0.2,
    )
    msg = resp.choices[0].message

    # handle tool calls (single round)
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = tool_executor(call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

        resp2 = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
        )
        return resp2.choices[0].message.content.strip()

    return msg.content.strip()
