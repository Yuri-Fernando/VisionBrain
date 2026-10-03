from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from visionbrain.dashboard.data import flatten_event, load_events, summarize_events  # noqa: E402


st.set_page_config(page_title="VisionBrain Dashboard", page_icon="👁️", layout="wide")
st.title("VisionBrain Operations")
st.caption("Dashboard local e somente leitura para eventos e evidências do edge engine.")

default_path = PROJECT_ROOT / "outputs" / "events" / "events.jsonl"
event_path = Path(st.sidebar.text_input("Arquivo JSONL", str(default_path))).expanduser()
events, malformed = load_events(event_path)

event_types = sorted({str(event.get("event_type", "unknown")) for event in events})
selected_types = st.sidebar.multiselect("Tipos de evento", event_types, default=event_types)
severities = sorted({str(event.get("severity", "info")) for event in events})
selected_severities = st.sidebar.multiselect("Severidades", severities, default=severities)
filtered = [
    event
    for event in events
    if str(event.get("event_type", "unknown")) in selected_types
    and str(event.get("severity", "info")) in selected_severities
]

summary = summarize_events(filtered)
cols = st.columns(4)
cols[0].metric("Eventos", summary["total"])
cols[1].metric("Tipos", len(summary["types"]))
cols[2].metric("Evidências", summary["snapshots"])
cols[3].metric("Linhas inválidas", malformed)

if not events:
    st.info(
        "Nenhum evento encontrado. Execute `visionbrain demo --output outputs/demo` "
        "e aponte a barra lateral para `outputs/demo/events.jsonl`, ou execute o motor real."
    )
    st.stop()

rows = [flatten_event(event) for event in filtered]
data = pd.DataFrame(rows)
left, right = st.columns([1, 2])
with left:
    st.subheader("Distribuição")
    counts = pd.Series(summary["types"], name="eventos")
    st.bar_chart(counts)
with right:
    st.subheader("Linha do tempo")
    timeline = data.copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce", utc=True)
    timeline = timeline.dropna(subset=["timestamp"])
    if timeline.empty:
        st.caption("Sem timestamps válidos.")
    else:
        chart = timeline.groupby([pd.Grouper(key="timestamp", freq="1min"), "event_type"]).size().unstack(fill_value=0)
        st.line_chart(chart)

st.subheader("Eventos")
st.dataframe(data.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

evidence = [row for row in rows if row.get("snapshot")]
if evidence:
    st.subheader("Evidências")
    labels = [f"{row['event_type']} · frame {row['frame_index']} · {row['timestamp']}" for row in evidence]
    selected = st.selectbox("Evento com snapshot", range(len(evidence)), format_func=lambda index: labels[index])
    snapshot = Path(str(evidence[selected]["snapshot"]))
    if not snapshot.is_absolute():
        snapshot = PROJECT_ROOT / snapshot
    if snapshot.exists():
        st.image(str(snapshot), caption=str(snapshot), use_container_width=True)
    else:
        st.warning(f"Snapshot não encontrado: {snapshot}")
