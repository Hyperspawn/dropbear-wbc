import json
from pathlib import Path

from benchmarks.g1.validate import PATH, validate


def test_tasks_valid():
    assert validate(json.loads(Path(PATH).read_text())) == []
