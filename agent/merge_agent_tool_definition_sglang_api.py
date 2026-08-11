from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().parent / "api" / "merge_agent_tool_definition_sglang_api.py"),
    run_name="__main__",
)
