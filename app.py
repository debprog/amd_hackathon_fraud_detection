import json
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feature_engineering import load_data, engineer_features, get_feature_columns
import agents

app = FastAPI(title="FraudLens Scoring Service")

# ---------------- load model + data at startup ----------------
model = joblib.load("fraud_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
explainer = joblib.load("shap_explainer.pkl")

raw_df = load_data()
df = engineer_features(raw_df)

X = df[feature_cols]
probs = model.predict_proba(X)[:, 1]
df["risk_score"] = (probs * 100).round(1)

DECISIONS_FILE = "decisions.json"
if os.path.exists(DECISIONS_FILE):
    with open(DECISIONS_FILE) as f:
        decisions = json.load(f)
else:
    decisions = []


def shap_top_reasons(row_idx, top_n=3):
    row = X.iloc[[row_idx]]
    shap_values = explainer.shap_values(row)[0]
    contrib = pd.Series(shap_values, index=feature_cols)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    return [{"feature": k, "impact": round(float(v), 3)} for k, v in top.items()]


def recommended_action(score):
    if score < 40:
        return "allow"
    elif score < 70:
        return "review"
    else:
        return "block"


# ---------------- schemas ----------------
class Decision(BaseModel):
    txn_index: int
    action: str  # "confirm_block" | "false_positive" | "escalate"
    notes: str = ""


# ---------------- endpoints ----------------
@app.get("/")
def root():
    return {"status": "ok", "total_transactions": len(df)}


@app.get("/transactions")
def get_transactions(limit: int = 50, offset: int = 0):
    subset = df.iloc[offset: offset + limit]
    return subset[
        ["step", "type", "amount", "nameOrig", "nameDest", "risk_score", "isFraud"]
    ].reset_index().rename(columns={"index": "txn_index"}).to_dict(orient="records")


@app.get("/transaction/{txn_index}")
def get_transaction(txn_index: int):
    if txn_index < 0 or txn_index >= len(df):
        raise HTTPException(404, "Transaction not found")
    row = df.iloc[txn_index]
    score = row["risk_score"]
    return {
        "txn_index": txn_index,
        "step": int(row["step"]),
        "amount": float(row["amount"]),
        "nameOrig": row["nameOrig"],
        "nameDest": row["nameDest"],
        "risk_score": float(score),
        "recommended_action": recommended_action(score),
        "top_reasons": shap_top_reasons(txn_index),
        "is_fraud_label": int(row["isFraud"]),
    }


@app.get("/alerts")
def get_alerts(min_score: float = 40, limit: int = 50):
    subset = df[df["risk_score"] >= min_score].head(limit)
    results = []
    for idx, row in subset.iterrows():
        results.append({
            "txn_index": int(idx),
            "step": int(row["step"]),
            "type": [c.replace("type_", "") for c in feature_cols
                     if c.startswith("type_") and row[c] == 1][0],
            "amount": float(row["amount"]),
            "nameOrig": row["nameOrig"],
            "nameDest": row["nameDest"],
            "risk_score": float(row["risk_score"]),
            "recommended_action": recommended_action(row["risk_score"]),
        })
    return results


@app.get("/account/{account_id}/history")
def account_history(account_id: str, limit: int = 20):
    subset = df[(df["nameOrig"] == account_id) | (df["nameDest"] == account_id)].head(limit)
    if subset.empty:
        raise HTTPException(404, "Account not found")
    return subset[
        ["step", "amount", "nameOrig", "nameDest", "risk_score", "isFraud"]
    ].reset_index().rename(columns={"index": "txn_index"}).to_dict(orient="records")


@app.get("/account/{account_id}/linked")
def linked_accounts(account_id: str, limit: int = 20):
    """Accounts that share a destination/origin with this account (graph lookup)."""
    sent_to = df.loc[df["nameOrig"] == account_id, "nameDest"].unique().tolist()
    received_from = df.loc[df["nameDest"] == account_id, "nameOrig"].unique().tolist()
    return {
        "account_id": account_id,
        "sent_to": sent_to[:limit],
        "received_from": received_from[:limit],
    }


@app.post("/decision")
def post_decision(decision: Decision):
    if decision.txn_index < 0 or decision.txn_index >= len(df):
        raise HTTPException(404, "Transaction not found")
    record = decision.dict()
    decisions.append(record)
    with open(DECISIONS_FILE, "w") as f:
        json.dump(decisions, f, indent=2)
    return {"status": "logged", "total_decisions": len(decisions)}


@app.get("/decisions")
def get_decisions():
    return decisions


class CopilotQuery(BaseModel):
    txn_index: int
    question: str


def _tool_executor(name: str, args: dict):
    account_id = args.get("account_id")
    if name == "get_account_history":
        try:
            return account_history(account_id, limit=10)
        except HTTPException:
            return {"error": "account not found"}
    elif name == "get_linked_accounts":
        return linked_accounts(account_id)
    return {"error": f"unknown tool {name}"}


@app.get("/agent/explain/{txn_index}")
def agent_explain(txn_index: int):
    txn = get_transaction(txn_index)
    explanation = agents.explain_alert(txn)
    return {"txn_index": txn_index, "explanation": explanation}


@app.get("/agent/triage/{txn_index}")
def agent_triage(txn_index: int):
    txn = get_transaction(txn_index)
    if not (40 <= txn["risk_score"] < 70):
        raise HTTPException(400, "Triage agent only applies to medium-risk (40-70) transactions")
    history = account_history(txn["nameOrig"], limit=10)
    decision = agents.triage_decision(txn, history)
    return {"txn_index": txn_index, **decision}


@app.get("/agent/case_summary/{txn_index}")
def agent_case_summary(txn_index: int):
    txn = get_transaction(txn_index)
    history = account_history(txn["nameOrig"], limit=10)
    linked = linked_accounts(txn["nameOrig"])
    summary = agents.case_summary(txn, history, linked)
    return {"txn_index": txn_index, "summary": summary}


@app.post("/agent/copilot")
def agent_copilot(query: CopilotQuery):
    txn = get_transaction(query.txn_index)
    answer = agents.copilot_chat(query.question, txn, _tool_executor)
    return {"txn_index": query.txn_index, "answer": answer}


@app.get("/metrics")
def metrics():
    total = len(df)
    alerts = (df["risk_score"] >= 40).sum()
    blocked = (df["risk_score"] >= 70).sum()
    fp = sum(1 for d in decisions if d["action"] == "false_positive")
    return {
        "total_transactions": int(total),
        "alerts": int(alerts),
        "auto_blocked": int(blocked),
        "decisions_logged": len(decisions),
        "false_positives_marked": fp,
    }
