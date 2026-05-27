import pytest

from atomonous.config import settings
from atomonous.tools.experiment_tools import ExperimentSearchTool, ExperimentArtifactReadTool
from atomonous.tools import symbolic_regression_tool


@pytest.fixture
def tools_sandbox(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "artifacts_dir", str(artifacts_dir))
    return artifacts_dir


def test_search_past_experiments_finds_sessions_and_files(tools_sandbox):
    session_by_name = tools_sandbox / "2025-01-01_00-00-00_beam-calibration"
    session_by_name.mkdir()

    session_by_file = tools_sandbox / "2025-01-02_00-00-00"
    session_by_file.mkdir()
    notes_path = session_by_file / "notes.txt"
    notes_path.write_text("Calibration results look good.")

    tool = ExperimentSearchTool()
    result = tool.forward(query="calibration", max_results=5)

    assert "Found 2 session(s) matching" in result
    assert "beam-calibration" in result
    assert "notes.txt" in result


def test_read_experiment_artifact_reads_file(tools_sandbox):
    session_dir = tools_sandbox / "2025-01-03_00-00-00"
    session_dir.mkdir()
    artifact_path = session_dir / "summary.json"
    artifact_path.write_text('{"status": "ok"}')

    tool = ExperimentArtifactReadTool()
    result = tool.forward(artifact_path=str(artifact_path))

    assert '"status": "ok"' in result


def test_symbolic_regression_tool_runs_with_stubbed_regressor(monkeypatch):
    class FakeRegressor:
        def __init__(self, *args, **kwargs):
            self._program = "x0 + 1"

        def fit(self, X, y):
            return self

        def score(self, X, y):
            return 0.99

    monkeypatch.setattr(symbolic_regression_tool, "SymbolicRegressor", FakeRegressor)

    tool = symbolic_regression_tool.SymbolicRegressionTool()
    result = tool.forward(x_features=[[1.0], [2.0], [3.0]], y_target=[2.0, 3.0, 4.0], feature_names=["x0"])

    assert "Symbolic Regression Complete" in result
    assert "x0 + 1" in result
    assert "0.9900" in result
