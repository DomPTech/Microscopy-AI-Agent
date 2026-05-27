from pathlib import Path
from typing import Optional

from smolagents import Tool

from atomonous.config import settings


class ExperimentSearchTool(Tool):
    name = "search_past_experiments"
    description = (
        "Searches past experiment session memories stored in the artifacts directory for a text query. "
        "Matches against session folder names and text-based artifacts (JSON/YAML/TXT)."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Text to search for in past session memories."
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of sessions to return.",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, query: str, max_results: Optional[int] = 5) -> str:
        if query is None or not str(query).strip():
            return "Query must be a non-empty string."

        query_text = str(query).strip()
        query_lower = query_text.lower()

        base_dir = Path(settings.artifacts_dir).expanduser().resolve()
        if not base_dir.exists():
            return f"Artifacts directory not found: {base_dir}"

        try:
            max_sessions = int(max_results) if max_results is not None else 5
        except ValueError:
            return "max_results must be an integer."

        if max_sessions <= 0:
            return "max_results must be greater than zero."

        def _read_match_snippet(path: Path) -> Optional[str]:
            try:
                if path.stat().st_size > 1_000_000:
                    return None
                content = path.read_text(errors="ignore")
            except Exception:
                return None

            idx = content.lower().find(query_lower)
            if idx == -1:
                return None
            start = max(0, idx - 80)
            end = min(len(content), idx + 80)
            snippet = content[start:end].replace("\n", " ").replace("\r", " ")
            return snippet.strip()

        rows = []
        session_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
        session_dirs.sort(key=lambda p: p.name, reverse=True)

        sessions_added = 0
        for session_dir in session_dirs:
            session_matches = []

            if query_lower in session_dir.name.lower():
                session_matches.append({
                    "session": session_dir.name,
                    "file": "(session name)",
                    "snippet": session_dir.name
                })

            for item in session_dir.rglob("*"):
                if not item.is_file():
                    continue
                if item.suffix.lower() not in {".json", ".yaml", ".yml", ".txt"}:
                    continue

                snippet = _read_match_snippet(item)
                if snippet:
                    session_matches.append({
                        "session": session_dir.name,
                        "file": str(item.relative_to(session_dir)),
                        "snippet": snippet
                    })

            if session_matches:
                rows.extend(session_matches)
                sessions_added += 1

            if sessions_added >= max_sessions:
                break

        if not rows:
            return f"No matches for '{query_text}' in {base_dir}"

        header = f"Found {sessions_added} session(s) matching '{query_text}':"
        table = ["", "| Session | File | Snippet |", "| --- | --- | --- |"]
        for row in rows:
            table.append(f"| {row['session']} | {row['file']} | {row['snippet']} |")

        return header + "\n" + "\n".join(table)


class ExperimentArtifactReadTool(Tool):
    name = "read_experiment_artifact"
    description = (
        "Reads a text-based artifact file from the session memories folder. "
        "The path must be within the artifacts directory."
    )
    inputs = {
        "artifact_path": {
            "type": "string",
            "description": "Relative or absolute path to a file inside the artifacts directory."
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum number of characters to return from the file.",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, artifact_path: str, max_chars: Optional[int] = 8000) -> str:
        if artifact_path is None or not str(artifact_path).strip():
            return "artifact_path must be a non-empty string."

        base_dir = Path(settings.artifacts_dir).expanduser().resolve()
        if not base_dir.exists():
            return f"Artifacts directory not found: {base_dir}"

        try:
            limit = int(max_chars) if max_chars is not None else 8000
        except ValueError:
            return "max_chars must be an integer."

        if limit <= 0:
            return "max_chars must be greater than zero."

        raw_path = Path(str(artifact_path)).expanduser()
        resolved_path = raw_path.resolve() if raw_path.is_absolute() else (base_dir / raw_path).resolve()

        try:
            resolved_path.relative_to(base_dir)
        except ValueError:
            return "artifact_path must be within the artifacts directory."

        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Artifact file not found: {resolved_path}"

        try:
            content = resolved_path.read_text(errors="ignore")
        except Exception as e:
            return f"Failed to read artifact file: {e}"

        if len(content) > limit:
            content = content[:limit] + "\n...[truncated]"

        return content
