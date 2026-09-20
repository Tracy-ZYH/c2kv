
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_history_multisource_lib import (
    build_decision_points,
    extract_tool_call,
    has_tool_call_text,
    iter_agent_llm,
    iter_hermes,
    iter_toucan,
    parquet_file_summary,
)


def inspect_source(name, path, iterator_factory, max_preview=2):
    stats = Counter()
    files = parquet_file_summary(path)
    trajectories = 0
    decision_points = 0
    roles = Counter()
    tool_shapes = Counter()
    subsets = Counter()
    preview = []
    tool_call_examples = []
    for traj in iterator_factory(path, stats):
        trajectories += 1
        subsets.update([str(traj.metadata.get("benchmark") or traj.metadata.get("subset_name") or traj.metadata.get("category") or "unknown")])
        tool_shapes.update(["list" if isinstance(traj.tools, list) else type(traj.tools).__name__])
        for turn in traj.turns:
            for msg in turn.get("messages", []):
                roles.update([msg.get("role", "unknown")])
        dps = build_decision_points(traj, min_completed_history_turns=2, max_history_docs=16)
        decision_points += len(dps)
        for sample in dps:
            if len(preview) < max_preview:
                preview.append(sample)
            if len(tool_call_examples) < max_preview and has_tool_call_text(sample.get("target", "")):
                tool_call_examples.append(extract_tool_call(sample.get("target", "")))
    return {
        "name": name,
        "files": files,
        "raw_or_span_trajectories_after_adapter": trajectories,
        "decision_points_min_history_2": decision_points,
        "stats": dict(stats),
        "roles": dict(roles),
        "tool_definition_format": dict(tool_shapes),
        "subset_or_domain_distribution": subsets.most_common(30),
        "sample_preview": preview[:1],
        "tool_call_preview": tool_call_examples[:2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default="datasets/agent-llm-traces-v2")
    parser.add_argument("--hermes_path", default="datasets/hermes-agent-reasoning-traces")
    parser.add_argument("--toucan_path", default="datasets/toucan-1.5m")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = {
        "agent_llm": inspect_source("agent_llm", args.agent_path, lambda p, s: iter_agent_llm(p, s)),
        "toucan": inspect_source("toucan", args.toucan_path, lambda p, s: iter_toucan(p, s, allow_irrelevant=True)),
        "hermes": inspect_source("hermes", args.hermes_path, lambda p, s: iter_hermes(p, s, strip_reasoning=True)),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
