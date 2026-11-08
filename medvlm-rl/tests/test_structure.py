from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]

def test_core_sources_exist_and_parse():
    for path in (ROOT / "rewards/medical.py", ROOT / "evaluation/eval.py", ROOT / "evaluation/utils.py", ROOT / "scripts/model_merger.py"):
        assert path.exists(), path
        ast.parse(path.read_text(encoding="utf-8"))

def test_2b_configuration_references():
    cfg = (ROOT / "configs/medvlm_2b_dapo.yaml").read_text()
    assert "Qwen/Qwen3-VL-2B-Instruct" in cfg
    assert "rewards/medical.py:compute_score" in cfg
    assert "prompts/medical_format.jinja" in cfg
    assert "hiyouga/math12k" not in cfg
    assert not (ROOT / "verl").exists()
