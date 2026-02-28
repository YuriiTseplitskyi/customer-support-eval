import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from analyze.llm_council import ChatEvaluator

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "datasets"
RESULTS_DIR = ROOT_DIR / "results"
GEN_RESULTS_DIR = RESULTS_DIR / "generation"
RESULTS_FILE = RESULTS_DIR / "analysis_results.jsonl"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
GEN_RESULTS_DIR.mkdir(exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    base = uploaded_file.name or "dataset.jsonl"
    dest = DATA_DIR / base
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        counter = 1
        while dest.exists():
            dest = DATA_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def list_datasets() -> List[Path]:
    files = list(DATA_DIR.glob("*.json")) + list(DATA_DIR.glob("*.jsonl"))
    return sorted(files, key=lambda p: p.name.lower())


def load_dataset(path: Path) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported JSON format.")


def preview_dataset(path: Path, max_chars: int = 5000) -> str:
    try:
        if path.suffix.lower() == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            preview = "\n".join(lines[:20])
            if len(preview) > max_chars:
                return preview[:max_chars] + "\n... (truncated)"
            if len(lines) > 20:
                return preview + "\n... (truncated)"
            return preview

        pretty = json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
        if len(pretty) > max_chars:
            return pretty[:max_chars] + "\n... (truncated)"
        return pretty
    except Exception as exc:  # noqa: BLE001
        return f"Read error: {exc}"


def normalize_role(role: str) -> str:
    role_up = (role or "").strip().upper()
    if role_up in {"CLIENT", "CUSTOMER", "USER"}:
        return "CLIENT"
    if role_up in {"AGENT", "ASSISTANT", "SUPPORT"}:
        return "AGENT"
    return role_up


def extract_turns(record: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_turns = record.get("turns")
    if not isinstance(raw_turns, list):
        raw_turns = record.get("messages")
    if not isinstance(raw_turns, list):
        return []

    turns: List[Dict[str, str]] = []
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        role = normalize_role(str(item.get("role", "")))
        text = str(item.get("text", "")).strip()
        if role not in {"CLIENT", "AGENT"} or not text:
            continue
        turns.append({"role": role, "text": text})
    return turns


def record_label(record: Dict[str, Any], index: int) -> str:
    return str(record.get("id") or record.get("dialogue_id") or f"row-{index}")


def record_scenario(record: Dict[str, Any]) -> str:
    return str(record.get("scenario") or record.get("sub_scenario") or "unknown")


def render_chat(turns: List[Dict[str, str]]) -> None:
    for turn in turns:
        st.markdown(f"**{turn.get('role', '')}:** {turn.get('text', '')}")


def summarize_result(result: Dict[str, Any], label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mistakes = result.get("mistakes", {}).get("agent_mistakes", []) or []
    summary = {
        "chat": label,
        "intent": result["intent"]["intent"],
        "quality_score": result["quality"]["quality_score"],
        "satisfaction": result["satisfaction"]["satisfaction"],
        "mistakes": ", ".join(mistakes) if mistakes else "-",
    }
    reasons = {
        "intent": result["intent"]["reason"],
        "quality": result["quality"]["reason"],
        "satisfaction": result["satisfaction"]["reason"],
        "mistakes": result.get("mistakes", {}).get("mistake_reasons", {}),
    }
    return summary, reasons


def append_result(dataset: str, mode: str, summary: Dict[str, Any], reasons: Dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "mode": mode,
        **summary,
        "reasons": reasons,
    }
    with RESULTS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_results() -> List[Dict[str, Any]]:
    if not RESULTS_FILE.exists():
        return []
    return [
        json.loads(line)
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_generation(
    n: int,
    seed: int,
    out_path: Path,
    manifest_path: Path,
    model: str,
    temperature: float,
    max_retries: int,
    dummy_text: bool,
    allow_partial: bool,
) -> subprocess.CompletedProcess[str]:
    out_rel = out_path.relative_to(ROOT_DIR).as_posix()
    manifest_rel = manifest_path.relative_to(ROOT_DIR).as_posix()
    cmd = [
        sys.executable,
        "-m",
        "generate.run",
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--out",
        out_rel,
        "--manifest",
        manifest_rel,
        "--max_retries",
        str(max_retries),
    ]
    if dummy_text:
        cmd.append("--dummy_text")
    else:
        cmd.extend(["--model", model, "--temperature", str(temperature)])
    if allow_partial:
        cmd.append("--allow_partial")

    return subprocess.run(
        cmd,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )


def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


st.set_page_config(page_title="Dataset Studio", layout="wide")
st.title("Dataset Studio")

tab_gen, tab_analyst, tab_datasets, tab_results = st.tabs(
    ["Dataset Generation", "Dataset Analyze", "Datasets", "Results"]
)

with tab_gen:
    st.header("Dataset Generation")
    col1, col2 = st.columns(2)
    with col1:
        dataset_name = st.text_input("Dataset name", value=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        n_records = st.number_input("Number of records", min_value=1, max_value=10000, value=100, step=1)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        max_retries = st.number_input("Max retries", min_value=0, max_value=20, value=3, step=1)
    with col2:
        mode = st.radio("Generation mode", ["LLM", "Dummy"], horizontal=True)
        model_name = st.text_input("Model", value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        allow_partial = st.checkbox("Allow partial generation on failures", value=True)

    out_path = DATA_DIR / f"{dataset_name}.jsonl"
    manifest_path = GEN_RESULTS_DIR / f"{dataset_name}_manifest.json"

    st.caption(f"Output: `{out_path}`")
    st.caption(f"Manifest: `{manifest_path}`")

    if st.button("Generate Dataset", type="primary"):
        if mode == "LLM" and not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is missing.")
        else:
            with st.spinner("Generating dataset..."):
                proc = run_generation(
                    n=int(n_records),
                    seed=int(seed),
                    out_path=out_path,
                    manifest_path=manifest_path,
                    model=model_name,
                    temperature=float(temperature),
                    max_retries=int(max_retries),
                    dummy_text=(mode == "Dummy"),
                    allow_partial=allow_partial,
                )

            if proc.returncode != 0:
                st.error("Generation failed.")
                if proc.stderr.strip():
                    st.code(proc.stderr, language="text")
                elif proc.stdout.strip():
                    st.code(proc.stdout, language="text")
            else:
                rows = count_jsonl_records(out_path)
                st.success(f"Done. Generated {rows} records.")
                if proc.stdout.strip():
                    with st.expander("Generation log"):
                        st.code(proc.stdout[-10000:], language="text")

with tab_analyst:
    st.header("Dataset Analyze")
    files = list_datasets()
    if not files:
        st.warning("No datasets found. Generate or upload one first.")
    else:
        ds_name = st.selectbox("Select dataset", [path.name for path in files])
        selected_path = next(path for path in files if path.name == ds_name)
        dataset = load_dataset(selected_path)

        mode = st.radio("Analyze mode", ["Single chat", "Full dataset"], horizontal=True)
        model_name = st.text_input("Analyzer model", value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY is missing.")
        else:
            evaluator = ChatEvaluator(api_key=api_key, model_name=model_name)

            if mode == "Single chat":
                options = [
                    f"{idx}: {record_label(item, idx)} ({record_scenario(item)})"
                    for idx, item in enumerate(dataset)
                ]
                selected = st.selectbox("Select chat", options)
                idx = int(selected.split(":", 1)[0])
                chat_record = dataset[idx]
                turns = extract_turns(chat_record)

                if not turns:
                    st.warning("Selected record has no valid dialogue turns.")
                else:
                    st.subheader("Chat Preview")
                    render_chat(turns)
                    if st.button("Analyze Chat", type="primary"):
                        with st.spinner("Running evaluation..."):
                            try:
                                result = evaluator.evaluate(turns)
                                summary, reasons = summarize_result(result, record_label(chat_record, idx))
                                append_result(ds_name, "single", summary, reasons)
                            except Exception as exc:  # noqa: BLE001
                                st.error(f"Evaluation failed: {exc}")
                            else:
                                st.success("Completed")
                                st.write(f"intent: **{summary['intent']}**")
                                st.write(f"quality_score: **{summary['quality_score']}**")
                                st.write(f"satisfaction: **{summary['satisfaction']}**")
                                st.write(f"mistakes: **{summary['mistakes']}**")
                                with st.expander("Reasons"):
                                    st.write(f"intent: {reasons['intent']}")
                                    st.write(f"quality: {reasons['quality']}")
                                    st.write(f"satisfaction: {reasons['satisfaction']}")
                                    if reasons["mistakes"]:
                                        st.write("mistakes:")
                                        for mistake, reason in reasons["mistakes"].items():
                                            st.write(f"- {mistake}: {reason}")
            else:
                st.info("Runs analysis for all records in selected dataset.")
                if st.button("Analyze Full Dataset", type="primary"):
                    summaries: List[Dict[str, Any]] = []
                    reasons_list: List[Tuple[str, Dict[str, Any]]] = []
                    errors: List[str] = []

                    with st.spinner("Analyzing full dataset..."):
                        for idx, chat_record in enumerate(dataset):
                            label = record_label(chat_record, idx)
                            turns = extract_turns(chat_record)
                            if not turns:
                                errors.append(f"{label}: no valid turns")
                                continue
                            try:
                                result = evaluator.evaluate(turns)
                                summary, reasons = summarize_result(result, label)
                                summaries.append(summary)
                                reasons_list.append((label, reasons))
                                append_result(ds_name, "dataset", summary, reasons)
                            except Exception as exc:  # noqa: BLE001
                                errors.append(f"{label}: {exc}")

                    if errors:
                        st.error("Some records failed:\n" + "\n".join(errors))
                    if summaries:
                        st.success(f"Completed. Processed {len(summaries)} records.")
                        st.dataframe(summaries, hide_index=True)
                        with st.expander("Per-chat reasons"):
                            for label, reasons in reasons_list:
                                st.markdown(f"**{label}**")
                                st.write(f"- intent: {reasons['intent']}")
                                st.write(f"- quality: {reasons['quality']}")
                                st.write(f"- satisfaction: {reasons['satisfaction']}")
                                if reasons["mistakes"]:
                                    st.write("- mistakes:")
                                    for mistake, reason in reasons["mistakes"].items():
                                        st.write(f"  - {mistake}: {reason}")
                                st.markdown("---")

with tab_datasets:
    st.header("Datasets")
    uploads = st.file_uploader(
        "Upload .json / .jsonl files",
        type=["json", "jsonl"],
        accept_multiple_files=True,
    )
    if uploads:
        for file in uploads:
            saved = save_uploaded_file(file)
            st.success(f"Saved: {saved.name}")

    files = list_datasets()
    if not files:
        st.info("No datasets yet.")
    else:
        st.subheader("Available datasets")
        for path in files:
            with st.expander(f"{path.name} ({path.stat().st_size} bytes)"):
                st.code(preview_dataset(path), language="json")

with tab_results:
    st.header("Results")
    results = load_results()
    if not results:
        st.info("No saved analysis results yet.")
    else:
        dataset_options = ["All"] + sorted({record["dataset"] for record in results})
        selected_dataset = st.selectbox("Filter by dataset", dataset_options)
        filtered = results if selected_dataset == "All" else [r for r in results if r["dataset"] == selected_dataset]

        table = [
            {
                "timestamp": record["timestamp"],
                "dataset": record["dataset"],
                "chat": record["chat"],
                "intent": record["intent"],
                "quality_score": record["quality_score"],
                "satisfaction": record["satisfaction"],
                "mistakes": record["mistakes"],
            }
            for record in filtered
        ]
        st.dataframe(table, hide_index=True)

        with st.expander("Detailed reasons"):
            for record in filtered:
                st.markdown(f"**{record['chat']} - {record['dataset']}**")
                st.write(f"- intent: {record['reasons']['intent']}")
                st.write(f"- quality: {record['reasons']['quality']}")
                st.write(f"- satisfaction: {record['reasons']['satisfaction']}")
                mistake_reasons = record["reasons"].get("mistakes") or {}
                if mistake_reasons:
                    st.write("- mistakes:")
                    for mistake, reason in mistake_reasons.items():
                        st.write(f"  - {mistake}: {reason}")
                st.markdown("---")
