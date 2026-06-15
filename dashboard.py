import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="FraudLens", layout="wide")
st.title("FraudLens — live transaction monitor")

# ---------------- metrics ----------------
metrics = requests.get(f"{API}/metrics").json()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions scanned", f"{metrics['total_transactions']:,}")
c2.metric("Alerts (score >= 40)", f"{metrics['alerts']:,}")
c3.metric("Auto-blocked (score >= 70)", f"{metrics['auto_blocked']:,}")
fp_rate = (
    metrics["false_positives_marked"] / metrics["decisions_logged"] * 100
    if metrics["decisions_logged"] else 0
)
c4.metric("False positive rate", f"{fp_rate:.1f}%")

st.divider()

# ---------------- layout: alert feed + detail ----------------
left, right = st.columns([1, 1.3])

with left:
    st.subheader("Alert feed")
    min_score = st.slider("Minimum risk score", 0, 100, 40, step=5)
    alerts = requests.get(f"{API}/alerts", params={"min_score": min_score, "limit": 30}).json()

    if not alerts:
        st.info("No alerts at this threshold.")

    selected_idx = st.session_state.get("selected_idx")

    for a in alerts:
        label = (
            f"[{a['risk_score']:.0f}] {a['type']} ${a['amount']:,.2f} "
            f"{a['nameOrig']} -> {a['nameDest']} ({a['recommended_action']})"
        )
        if st.button(label, key=f"alert_{a['txn_index']}"):
            st.session_state["selected_idx"] = a["txn_index"]
            selected_idx = a["txn_index"]

with right:
    st.subheader("Alert detail")

    if not selected_idx:
        st.info("Select an alert from the feed to see details.")
    else:
        txn = requests.get(f"{API}/transaction/{selected_idx}").json()

        st.markdown(f"**Transaction #{txn['txn_index']}**  ·  step {txn['step']}")
        st.markdown(f"`{txn['nameOrig']}` → `{txn['nameDest']}`  ·  amount **${txn['amount']:,.2f}**")

        score = txn["risk_score"]
        if score >= 70:
            st.error(f"Risk score: {score:.1f} / 100 — recommended action: **{txn['recommended_action'].upper()}**")
        elif score >= 40:
            st.warning(f"Risk score: {score:.1f} / 100 — recommended action: **{txn['recommended_action'].upper()}**")
        else:
            st.success(f"Risk score: {score:.1f} / 100 — recommended action: **{txn['recommended_action'].upper()}**")

        st.markdown("**Why this was flagged (top SHAP features):**")
        for r in txn["top_reasons"]:
            direction = "increases" if r["impact"] > 0 else "decreases"
            st.markdown(f"- `{r['feature']}` {direction} fraud risk (impact {r['impact']:+.3f})")

        # ---- AI explanation ----
        if st.button("Explain in plain English (AI)", key=f"explain_{selected_idx}"):
            with st.spinner("Asking the explanation agent..."):
                resp = requests.get(f"{API}/agent/explain/{selected_idx}", timeout=120).json()
            st.info(resp["explanation"])

        # ---- Triage agent (medium risk only) ----
        if 40 <= score < 70:
            if st.button("Run triage agent", key=f"triage_{selected_idx}"):
                with st.spinner("Triage agent reasoning..."):
                    resp = requests.get(f"{API}/agent/triage/{selected_idx}", timeout=120).json()
                st.warning(f"Triage decision: **{resp['action']}** — {resp['reason']}")

        # ---- Case summary (high risk only) ----
        if score >= 70:
            if st.button("Generate case summary (AI)", key=f"summary_{selected_idx}"):
                with st.spinner("Drafting case summary..."):
                    resp = requests.get(f"{API}/agent/case_summary/{selected_idx}", timeout=120).json()
                st.text_area("Case summary", resp["summary"], height=150)

        st.markdown("---")
        st.markdown("**Account context**")
        ac1, ac2 = st.columns(2)

        with ac1:
            st.caption(f"Sender history: {txn['nameOrig']}")
            hist = requests.get(f"{API}/account/{txn['nameOrig']}/history", params={"limit": 10}).json()
            st.dataframe(hist, use_container_width=True, height=200)

        with ac2:
            st.caption(f"Linked accounts: {txn['nameOrig']}")
            linked = requests.get(f"{API}/account/{txn['nameOrig']}/linked").json()
            st.write("Sent to:", linked["sent_to"][:5])
            st.write("Received from:", linked["received_from"][:5])

        st.markdown("---")
        st.markdown("**Investigator decision**")
        notes = st.text_input("Notes", key=f"notes_{selected_idx}")
        b1, b2, b3 = st.columns(3)

        def log_decision(action):
            requests.post(f"{API}/decision", json={
                "txn_index": selected_idx,
                "action": action,
                "notes": notes,
            })
            st.success(f"Logged: {action}")

        if b1.button("Confirm block", type="primary"):
            log_decision("confirm_block")
        if b2.button("False positive"):
            log_decision("false_positive")
        if b3.button("Escalate"):
            log_decision("escalate")

        # ---- Investigator copilot ----
        st.markdown("---")
        st.markdown("**Ask the copilot**")
        question = st.text_input(
            "Ask about this transaction (e.g. 'has this sender done this before?')",
            key=f"copilot_q_{selected_idx}",
        )
        if st.button("Ask", key=f"copilot_btn_{selected_idx}") and question:
            with st.spinner("Copilot thinking..."):
                resp = requests.post(
                    f"{API}/agent/copilot",
                    json={"txn_index": selected_idx, "question": question},
                    timeout=120,
                ).json()
            st.markdown(f"**Copilot:** {resp['answer']}")

st.divider()
with st.expander("Decision log"):
    decisions = requests.get(f"{API}/decisions").json()
    st.dataframe(decisions, use_container_width=True)
