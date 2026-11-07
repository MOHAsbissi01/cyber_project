"""UI tests with Streamlit's AppTest: every page renders without exception and its main interaction works against the real backend."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _need_artifacts(has_artifacts):
    if not has_artifacts:
        pytest.skip("artifacts not built")


def page(name):
    return AppTest.from_file(str(ROOT / "app" / "views" / name), default_timeout=300)


@pytest.mark.parametrize("name", ["dashboard.py", "graph_explorer.py", "evaluation.py", "dataset_explorer.py", "incident_analysis.py",
                                  "question_answering.py"])
def test_page_renders_without_exception(name):
    at = page(name).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_incident_analysis_flow_and_empty_input():
    at = page("incident_analysis.py").run()
    at.button[0].click().run()
    assert not at.exception and "Predicted topic" in [s.value for s in at.subheader]
    at.text_area[0].set_value("   ").run()
    at.button[0].click().run()
    assert any("empty" in w.value.lower() for w in at.warning)


def test_question_answering_flow_with_answer_and_abstention():
    at = page("question_answering.py").run()
    at.chat_input[0].set_value("Which vulnerabilities affect log4j?").run()
    assert not at.exception and "CVE-2021-44228" in at.chat_message[1].markdown[0].value
    at.chat_input[0].set_value("What is the capital of France?").run()
    assert any("cannot answer" in m.markdown[0].value for m in at.chat_message[3:4] if m.markdown) or any(
        "cannot answer" in w.value for w in at.warning)


def test_graph_explorer_entity_selection():
    at = page("graph_explorer.py").run()
    assert not at.exception
    at.text_input[0].set_value("windows").run()                   # search -> the entity profile of the best match is rendered
    assert not at.exception
    assert any("Relations" in m.value for m in at.markdown)
    at.text_input[0].set_value("zzzz-no-such-entity").run()       # empty state: informative message, no crash
    assert not at.exception and any("No entity matches" in i.value for i in at.info)


def test_dataset_explorer_filters():
    at = page("dataset_explorer.py").run()
    at.selectbox[0].select("xss").run()
    assert not at.exception
