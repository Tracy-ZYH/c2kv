"""Audit exact BFCL overlap against one prepared B/C training corpus.

The audit reads only prepared-corpus metadata, BFCL task IDs/questions, and
checkpoint metadata.  Exact prompt matches use only selected training prefixes;
the audit never reads model weights or BFCL ground truth.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUDIT_SCHEMA = "c2kv-b-training-overlap-v1"
PREPARED_SCHEMA = "history-memory-paired-v1"
PACKING_VERSION = "history-event-v1"
RAW_LAYOUT_PROFILE = "event-native-evidence-v1"
EVIDENCE_VERSION = "history-evidence-v1"
TRAINING_PROFILE = "history-event-base-query-v1"
SESSIONS_FILE = "sessions.jsonl"
DECISIONS_FILE = "paired_decisions.jsonl"

_WS_RE = re.compile(r"\s+")
_NON_ID_RE = re.compile(r"[^0-9a-z]+")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    return {
        "path": path.resolve().as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": size,
    }


def _normalize_prompt(text: str) -> str:
    """Match the frozen overlap_1088_v1 exact-prompt normalization."""
    return _WS_RE.sub(" ", text.lower()).strip()


def _prompt_sha256(text: str) -> str:
    return _sha256_bytes(_normalize_prompt(text).encode("utf-8"))


def _normalize_task_id(text: str) -> str:
    """Canonicalize a complete ID without stripping namespaces or prefixes."""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return _NON_ID_RE.sub("_", normalized).strip("_")


def _decision_id(session: Mapping[str, Any], source_message_index: int) -> str:
    identity = json.dumps(
        [
            session["source"],
            session["session_id"],
            session["task_id"],
            session["template_id"],
            source_message_index,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "decision-" + _sha256_bytes(identity.encode("utf-8"))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _read_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    return _require_mapping(value, str(path)), payload


def _read_jsonl(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    rows: list[tuple[int, Mapping[str, Any]]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            rows.append((line_number, _require_mapping(row, f"{path}:{line_number}")))
    return rows


def _validated_ids(values: Iterable[str], label: str) -> list[str]:
    result = list(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{label} must contain only nonempty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate task IDs")
    canonical: dict[str, str] = {}
    for value in result:
        normalized = _normalize_task_id(value)
        if not normalized:
            raise ValueError(f"{label} contains an empty canonical task ID")
        previous = canonical.setdefault(normalized, value)
        if previous != value:
            raise ValueError(
                f"{label} has a canonical task-ID collision: {previous!r} and {value!r}"
            )
    return result


def _bfcl_prompt_units(
    path: Path, requested_ids: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, Any]]:
    units: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in requested_ids}
    canonical_to_id: dict[str, str] = {}
    rows_seen = 0
    requested_rows = 0
    for line_number, row in _read_jsonl(path):
        rows_seen += 1
        task_id = _require_string(row.get("id"), f"{path}:{line_number}.id")
        canonical = _normalize_task_id(task_id)
        previous = canonical_to_id.setdefault(canonical, task_id)
        if previous != task_id:
            raise ValueError(
                f"BFCL canonical task-ID collision: {previous!r} and {task_id!r}"
            )
        if task_id not in requested_ids:
            continue
        requested_rows += 1
        question = row.get("question")
        if not isinstance(question, list):
            raise ValueError(f"{path}:{line_number}.question must be a list")
        for turn_index, turn in enumerate(question):
            messages = turn if isinstance(turn, list) else [turn]
            for message_index, message in enumerate(messages):
                item = _require_mapping(
                    message,
                    f"{path}:{line_number}.question[{turn_index}][{message_index}]",
                )
                if item.get("role") != "user":
                    continue
                content = item.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"{path}:{line_number} has a non-text or empty BFCL user prompt"
                    )
                units[task_id].append(
                    {
                        "task_id": task_id,
                        "source_line": line_number,
                        "turn_index": turn_index,
                        "message_index": message_index,
                        "prompt_sha256": _prompt_sha256(content),
                    }
                )
    missing = sorted(requested_ids - {task_id for task_id, values in units.items() if values})
    if missing:
        raise ValueError(f"BFCL task IDs are missing or have no user prompts: {missing!r}")
    return units, canonical_to_id, {
        **_file_identity(path),
        "rows_seen": rows_seen,
        "requested_rows": requested_rows,
        "requested_task_ids": len(requested_ids),
    }


def _validate_manifest(
    corpus_dir: Path,
) -> tuple[Mapping[str, Any], dict[str, Any], Path, Path]:
    manifest_path = corpus_dir / "manifest.json"
    manifest, payload = _read_json(manifest_path)
    expected_fields = {
        "schema_version": PREPARED_SCHEMA,
        "packing_version": PACKING_VERSION,
        "raw_layout_profile": RAW_LAYOUT_PROFILE,
        "evidence_version": EVIDENCE_VERSION,
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Prepared manifest {field} is {manifest.get(field)!r}; expected {expected!r}"
            )
    if manifest.get("arms") != ["C", "B"]:
        raise ValueError("Prepared manifest arms must be exactly ['C', 'B']")
    if manifest.get("allow_unchanged_b") is not False:
        raise ValueError("Formal prepared corpus must set allow_unchanged_b=false")
    tokenizer = _require_mapping(manifest.get("tokenizer"), "manifest.tokenizer")
    _require_string(tokenizer.get("sha256"), "manifest.tokenizer.sha256")
    packing = _require_mapping(manifest.get("packing"), "manifest.packing")
    ratios = packing.get("ratios")
    if (
        not isinstance(ratios, list)
        or not ratios
        or any(not isinstance(ratio, int) or isinstance(ratio, bool) for ratio in ratios)
    ):
        raise ValueError("manifest.packing.ratios must be a nonempty integer list")
    _require_mapping(manifest.get("policy"), "manifest.policy")
    counts = _require_mapping(manifest.get("counts"), "manifest.counts")

    files = _require_mapping(manifest.get("files"), "manifest.files")
    if files != {"sessions": SESSIONS_FILE, "paired_decisions": DECISIONS_FILE}:
        raise ValueError("Prepared manifest files do not match the formal B/C contract")
    integrity = _require_mapping(manifest.get("file_integrity"), "manifest.file_integrity")
    sessions_path = corpus_dir / SESSIONS_FILE
    decisions_path = corpus_dir / DECISIONS_FILE
    file_identities: dict[str, Any] = {}
    for path in (sessions_path, decisions_path):
        actual = _file_identity(path)
        expected = _require_mapping(
            integrity.get(path.name), f"manifest.file_integrity[{path.name!r}]"
        )
        if expected.get("sha256") != actual["sha256"] or expected.get("bytes") != actual["bytes"]:
            raise ValueError(f"Prepared file integrity check failed: {path.name}")
        file_identities[path.name] = actual

    return manifest, {
        "manifest": {
            "path": manifest_path.resolve().as_posix(),
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
        },
        "files": file_identities,
        "contract": {
            **expected_fields,
            "arms": list(manifest["arms"]),
            "allow_unchanged_b": manifest["allow_unchanged_b"],
            "tokenizer_sha256": tokenizer["sha256"],
            "ratios": list(ratios),
            "counts": {
                key: counts.get(key)
                for key in (
                    "sessions_written",
                    "base_decisions_written",
                    "paired_exposures_written",
                )
            },
        },
    }, sessions_path, decisions_path


def _validate_corpus_rows(
    manifest: Mapping[str, Any],
    sessions_path: Path,
    decisions_path: Path,
) -> tuple[
    list[tuple[int, Mapping[str, Any]]],
    dict[tuple[int, int], dict[str, Any]],
]:
    sessions = _read_jsonl(sessions_path)
    decisions = _read_jsonl(decisions_path)
    if not sessions:
        raise ValueError("Prepared sessions.jsonl is empty")
    if not decisions:
        raise ValueError("Prepared paired_decisions.jsonl is empty")

    session_keys: list[str] = []
    seen_session_keys: set[str] = set()
    for expected_index, (line_number, session) in enumerate(sessions):
        label = f"{sessions_path}:{line_number}"
        if session.get("schema_version") != PREPARED_SCHEMA:
            raise ValueError(f"{label} has an incompatible schema_version")
        if session.get("session_index") != expected_index:
            raise ValueError(f"{label} session_index must be dense and ordered")
        source = _require_string(session.get("source"), f"{label}.source")
        session_id = _require_string(session.get("session_id"), f"{label}.session_id")
        _require_string(session.get("task_id"), f"{label}.task_id")
        _require_string(session.get("template_id"), f"{label}.template_id")
        if session.get("split") != "train":
            raise ValueError(f"{label}.split must be 'train'")
        messages = session.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{label}.messages must be a nonempty list")
        if "source_metadata" in session and not isinstance(session["source_metadata"], Mapping):
            raise ValueError(f"{label}.source_metadata must be an object when present")
        session_key = json.dumps([source, session_id], ensure_ascii=False, separators=(",", ":"))
        if session_key in seen_session_keys:
            raise ValueError(f"Duplicate canonical session identity: {session_key}")
        seen_session_keys.add(session_key)
        session_keys.append(session_key)

    valid_ratios = set(_require_mapping(manifest["packing"], "manifest.packing")["ratios"])
    prefixes: dict[tuple[int, int], dict[str, Any]] = {}
    exposure_keys: set[tuple[str, int, int]] = set()
    for line_number, record in decisions:
        label = f"{decisions_path}:{line_number}"
        if record.get("schema_version") != PREPARED_SCHEMA:
            raise ValueError(f"{label} has an incompatible schema_version")
        session_index = record.get("session_index")
        if (
            not isinstance(session_index, int)
            or isinstance(session_index, bool)
            or not 0 <= session_index < len(sessions)
        ):
            raise ValueError(f"{label} references an unknown session_index")
        session = sessions[session_index][1]
        source_message_index = record.get("source_message_index")
        messages = session["messages"]
        if (
            not isinstance(source_message_index, int)
            or isinstance(source_message_index, bool)
            or not 0 <= source_message_index < len(messages)
        ):
            raise ValueError(f"{label} has an invalid source_message_index")
        target = _require_mapping(
            messages[source_message_index],
            f"{sessions_path}:{sessions[session_index][0]}.messages[{source_message_index}]",
        )
        if target.get("role") != "assistant":
            raise ValueError(f"{label} source_message_index does not point to an assistant target")
        if record.get("session_key") != session_keys[session_index]:
            raise ValueError(f"{label}.session_key differs from its session")
        if record.get("source") != session["source"] or record.get("split") != "train":
            raise ValueError(f"{label} source/split differs from its train session")
        if set(_require_mapping(record.get("arms"), f"{label}.arms")) != {"C", "B"}:
            raise ValueError(f"{label}.arms must contain exactly C and B")
        if record.get("ratio") not in valid_ratios:
            raise ValueError(f"{label}.ratio is outside manifest.packing.ratios")
        decision_id = _require_string(record.get("decision_id"), f"{label}.decision_id")
        if decision_id != _decision_id(session, source_message_index):
            raise ValueError(f"{label}.decision_id differs from its deterministic session identity")
        repetition_index = record.get("repetition_index")
        if not isinstance(repetition_index, int) or isinstance(repetition_index, bool) or repetition_index < 0:
            raise ValueError(f"{label}.repetition_index must be a nonnegative integer")
        exposure_key = (decision_id, int(record["ratio"]), repetition_index)
        if exposure_key in exposure_keys:
            raise ValueError(f"{label} duplicates a prepared exposure")
        exposure_keys.add(exposure_key)
        prefix_key = (session_index, source_message_index)
        prefix = prefixes.setdefault(
            prefix_key,
            {
                "session_index": session_index,
                "session_line": sessions[session_index][0],
                "source_message_index": source_message_index,
                "decision_id": decision_id,
                "paired_record_lines": [],
            },
        )
        if prefix["decision_id"] != decision_id:
            raise ValueError(f"{label} assigns multiple decision IDs to one source prefix")
        prefix["paired_record_lines"].append(line_number)

    referenced_sessions = {session_index for session_index, _ in prefixes}
    if referenced_sessions != set(range(len(sessions))):
        raise ValueError("sessions.jsonl contains sessions unused by paired_decisions.jsonl")

    counts = _require_mapping(manifest["counts"], "manifest.counts")
    expected_counts = {
        "sessions_written": len(sessions),
        "base_decisions_written": len({record[1]["decision_id"] for record in decisions}),
        "paired_exposures_written": len(decisions),
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise ValueError(f"manifest.counts.{key}={counts.get(key)!r}; expected {expected}")
    return sessions, prefixes


def _training_inventory(
    sessions: Sequence[tuple[int, Mapping[str, Any]]],
    prefixes: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    prompt_locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_id_locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[tuple[str, str, str | None]] = Counter()
    source_prefix_counts: Counter[tuple[str, str, str | None]] = Counter()

    prefixes_by_session: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for (session_index, _), prefix in prefixes.items():
        prefixes_by_session[session_index].append(prefix)

    prompt_occurrences = 0
    for session_index, (session_line, session) in enumerate(sessions):
        metadata = session.get("source_metadata")
        benchmark = metadata.get("benchmark") if isinstance(metadata, Mapping) else None
        if benchmark is not None and not isinstance(benchmark, str):
            raise ValueError(f"sessions.jsonl:{session_line}.source_metadata.benchmark must be a string")
        source_key = (session["source"], session["template_id"], benchmark)
        source_counts[source_key] += 1
        source_prefix_counts[source_key] += len(prefixes_by_session[session_index])
        prefix_indices = sorted(
            int(prefix["source_message_index"]) for prefix in prefixes_by_session[session_index]
        )
        for field in ("task_id", "session_id"):
            raw_id = _require_string(session.get(field), f"sessions.jsonl:{session_line}.{field}")
            task_id_locations[_normalize_task_id(raw_id)].append(
                {
                    "field": field,
                    "session_index": session_index,
                    "session_line": session_line,
                    "source": session["source"],
                    "task_id": session["task_id"],
                    "session_id": session["session_id"],
                    "selected_source_message_indices": prefix_indices,
                }
            )

        messages = session["messages"]
        ordered_prefixes = sorted(
            prefixes_by_session[session_index], key=lambda value: value["source_message_index"]
        )
        for message_index, message in enumerate(messages):
            item = _require_mapping(
                message, f"sessions.jsonl:{session_line}.messages[{message_index}]"
            )
            if item.get("role") != "user":
                continue
            supporting_prefix = next(
                (
                    prefix
                    for prefix in ordered_prefixes
                    if int(prefix["source_message_index"]) > message_index
                ),
                None,
            )
            if supporting_prefix is None:
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"sessions.jsonl:{session_line}.messages[{message_index}] has a non-text or empty user prompt"
                )
            digest = _prompt_sha256(content)
            prompt_occurrences += 1
            prompt_locations[digest].append(
                {
                    "session_index": session_index,
                    "session_line": session_line,
                    "source": session["source"],
                    "task_id": session["task_id"],
                    "session_id": session["session_id"],
                    "user_message_index": message_index,
                    "supporting_source_message_index": supporting_prefix[
                        "source_message_index"
                    ],
                    "supporting_decision_id": supporting_prefix["decision_id"],
                    "supporting_paired_record_line": supporting_prefix[
                        "paired_record_lines"
                    ][0],
                }
            )

    source_inventory = [
        {
            "source": source,
            "template_id": template_id,
            "source_metadata_benchmark": benchmark,
            "sessions": count,
            "selected_prefixes": source_prefix_counts[(source, template_id, benchmark)],
        }
        for (source, template_id, benchmark), count in sorted(
            source_counts.items(), key=lambda item: tuple(value or "" for value in item[0])
        )
    ]
    training_summary = {
        "selected_unique_prefixes": len(prefixes),
        "selected_user_message_occurrences": prompt_occurrences,
        "user_prompt_occurrence_unit": (
            "unique (session_index, user_message_index) visible before at least one "
            "selected source_message_index"
        ),
        "unique_normalized_user_prompts": len(prompt_locations),
        "future_turns_scanned": False,
    }
    prefix_inventory = [
        {
            "session_index": prefix["session_index"],
            "session_line": prefix["session_line"],
            "source_message_index": prefix["source_message_index"],
            "decision_id": prefix["decision_id"],
            "paired_record_lines": list(prefix["paired_record_lines"]),
        }
        for _, prefix in sorted(prefixes.items())
    ]
    return prompt_locations, task_id_locations, source_inventory, {
        **training_summary,
        "prefix_provenance": prefix_inventory,
    }


def _overlap_for_ids(
    task_ids: Sequence[str],
    bfcl_units: Mapping[str, Sequence[Mapping[str, Any]]],
    training_prompt_locations: Mapping[str, Sequence[Mapping[str, Any]]],
    training_task_id_locations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    canonical_matches: list[dict[str, Any]] = []
    prompt_matches: list[dict[str, Any]] = []
    prompt_matched_tasks: set[str] = set()
    for task_id in task_ids:
        canonical = _normalize_task_id(task_id)
        id_locations = training_task_id_locations.get(canonical, ())
        if id_locations:
            canonical_matches.append(
                {
                    "task_id": task_id,
                    "canonical_task_id": canonical,
                    "training_sources": list(id_locations),
                }
            )
        for unit in bfcl_units[task_id]:
            digest = unit["prompt_sha256"]
            prompt_locations = training_prompt_locations.get(digest, ())
            if prompt_locations:
                prompt_matched_tasks.add(task_id)
                prompt_matches.append(
                    {
                        **dict(unit),
                        "training_prefix_hits": list(prompt_locations),
                    }
                )
    id_matched_tasks = [item["task_id"] for item in canonical_matches]
    return {
        "interpretation": (
            "exact_match_found" if canonical_matches or prompt_matches else "no_exact_match"
        ),
        "canonical_exact_task_ids": {
            "matched_task_count": len(id_matched_tasks),
            "matched_task_ids": sorted(id_matched_tasks),
            "matches": canonical_matches,
        },
        "normalized_exact_user_prompts": {
            "matched_task_count": len(prompt_matched_tasks),
            "matched_task_ids": sorted(prompt_matched_tasks),
            "matched_prompt_units": len(prompt_matches),
            "matches": prompt_matches,
        },
    }


def _audit_checkpoints(
    checkpoint_dirs: Iterable[str | Path], corpus_identity: str
) -> dict[str, Any]:
    paths = [Path(path) for path in checkpoint_dirs]
    if not paths:
        return {
            "status": "not_provided",
            "corpus_identity": corpus_identity,
            "checkpoints": [],
        }
    results: list[dict[str, Any]] = []
    for checkpoint_dir in paths:
        config_path = checkpoint_dir / "config.json"
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        config, _ = _read_json(config_path)
        trainer_state, _ = _read_json(trainer_state_path)
        contract = _require_mapping(
            trainer_state.get("contract"), f"{trainer_state_path}.contract"
        )
        arm = contract.get("arm")
        if arm not in {"C", "B"}:
            raise ValueError(f"{trainer_state_path}.contract.arm must be C or B")
        if config.get("history_memory_arm") != arm:
            raise ValueError(f"{config_path} and trainer_state.json disagree on arm")
        if config.get("history_memory_seed") != contract.get("seed"):
            raise ValueError(f"{config_path} and trainer_state.json disagree on seed")
        profiles = {
            config.get("history_memory_training_profile"),
            trainer_state.get("training_profile"),
            contract.get("profile"),
        }
        if profiles != {TRAINING_PROFILE}:
            raise ValueError(f"{checkpoint_dir} is not an event-native B/C checkpoint")
        identities = {
            config.get("history_memory_corpus_identity"),
            contract.get("corpus_identity"),
        }
        if identities != {corpus_identity}:
            raise ValueError(f"{checkpoint_dir} is not bound to the supplied corpus manifest")
        if config.get("history_memory_packing_version") != PACKING_VERSION:
            raise ValueError(f"{checkpoint_dir} has an incompatible packing version")
        if config.get("history_memory_raw_layout") != RAW_LAYOUT_PROFILE:
            raise ValueError(f"{checkpoint_dir} has an incompatible raw layout profile")
        if config.get("history_memory_evidence_version") != EVIDENCE_VERSION:
            raise ValueError(f"{checkpoint_dir} has an incompatible evidence version")
        results.append(
            {
                "checkpoint_dir": checkpoint_dir.resolve().as_posix(),
                "config": _file_identity(config_path),
                "trainer_state": _file_identity(trainer_state_path),
                "arm": arm,
                "seed": contract.get("seed"),
                "global_step": trainer_state.get("global_step"),
                "profile": TRAINING_PROFILE,
                "corpus_identity": corpus_identity,
            }
        )
    return {
        "status": "bound",
        "corpus_identity": corpus_identity,
        "checkpoints": results,
    }


def audit_formal_corpus(
    corpus_dir: str | Path,
    *,
    bfcl_tasks_path: str | Path,
    candidate_ids: Iterable[str],
    dev_task_ids: Iterable[str],
    checkpoint_dirs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Validate a formal prepared corpus and report bounded exact overlaps.

    Empty exact-match sets mean only that no match exists under the recorded
    normalization.  This function never freezes or labels a formal split clean.
    """
    corpus_root = Path(corpus_dir)
    if not corpus_root.is_dir():
        raise ValueError(f"corpus_dir is not a directory: {corpus_root}")
    candidates = _validated_ids(candidate_ids, "candidate_ids")
    development = _validated_ids(dev_task_ids, "dev_task_ids")
    requested_ids = set(candidates) | set(development)
    if not requested_ids:
        raise ValueError("candidate_ids and dev_task_ids are both empty")

    manifest, corpus_identity, sessions_path, decisions_path = _validate_manifest(corpus_root)
    sessions, prefixes = _validate_corpus_rows(manifest, sessions_path, decisions_path)
    prompt_locations, task_id_locations, source_inventory, training_prefixes = (
        _training_inventory(sessions, prefixes)
    )
    bfcl_units, _, bfcl_source = _bfcl_prompt_units(Path(bfcl_tasks_path), requested_ids)
    candidate_overlap = _overlap_for_ids(
        candidates, bfcl_units, prompt_locations, task_id_locations
    )
    development_overlap = _overlap_for_ids(
        development, bfcl_units, prompt_locations, task_id_locations
    )
    checkpoint_result = _audit_checkpoints(
        checkpoint_dirs, corpus_identity["manifest"]["sha256"]
    )

    dev_set = set(development)
    id_matched = set(candidate_overlap["canonical_exact_task_ids"]["matched_task_ids"])
    prompt_matched = set(
        candidate_overlap["normalized_exact_user_prompts"]["matched_task_ids"]
    )
    exclusions = (set(candidates) & dev_set) | id_matched | prompt_matched
    status = "checkpoint_bound" if checkpoint_result["status"] == "bound" else "corpus_only"
    exact_match_result = (
        "exact_match_found"
        if candidate_overlap["interpretation"] == "exact_match_found"
        else "no_exact_match"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "status": status,
        "formal_split_frozen": False,
        "interpretation": exact_match_result,
        "validation": {
            "status": "passed",
            "corpus_contract": "passed",
            "checkpoint_contract": (
                "passed" if checkpoint_result["status"] == "bound" else "not_provided"
            ),
        },
        "normalization": {
            "user_prompt_exact": "lowercase; collapse regex \\s+ to one ASCII space; strip",
            "canonical_task_id_exact": (
                "Unicode NFKC; lowercase; replace each non-[0-9a-z] run with underscore; "
                "strip underscores; compare the complete normalized string without alias stripping"
            ),
            "near_or_semantic_overlap": "not_computed",
        },
        "scope": {
            "formal_readiness": "not_determined",
            "near_or_semantic_overlap": "not_computed",
            "raw_prompts_or_tokens_stored": False,
            "population_interpretation": (
                "Prepared training population upper bound; checkpoint metadata binding "
                "is recorded separately; "
                "not an actual trained minibatch or exposure census"
            ),
        },
        "corpus": corpus_identity,
        "checkpoint_binding": checkpoint_result,
        "bfcl_source": bfcl_source,
        "training_prefixes": training_prefixes,
        "source_alias_diagnostic": {
            "scope": "bounded_observed_labels_only",
            "fields": ["source", "template_id", "source_metadata.benchmark"],
            "used_as_overlap_filter": False,
            "inventory": source_inventory,
        },
        "overlap": {
            "candidate": candidate_overlap,
            "development": development_overlap,
        },
        "excluded_candidate_ids": sorted(exclusions),
        "eligible_candidate_ids": sorted(set(candidates) - exclusions),
    }


__all__ = ["audit_formal_corpus"]
