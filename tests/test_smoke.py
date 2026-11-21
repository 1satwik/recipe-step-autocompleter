"""These pass on day one. They check the plumbing so you notice quickly if you break it."""
import os

from app import llm
from app.storage import Store


def test_the_store_keeps_what_you_put_in_it(tmp_path):
    store = Store(str(tmp_path / "app.json"))
    store.put("hello", {"n": 1})
    assert store.get("hello") == {"n": 1}
    assert store.get("missing", "fallback") == "fallback"
    assert store.all() == {"hello": {"n": 1}}


def test_model_settings_default_to_a_local_ollama(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                  # no .env here
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    base, key, model = llm.settings()
    assert base == "http://localhost:11434/v1" and key == "" and model == ""


def test_the_command_line_runs(capsys):
    from app.cli import main
    main(["hello"])
    assert "is ready" in capsys.readouterr().out


def test_suggest_command_displays_numbered_suggestions(capsys):
    from app.cli import main

    main(["suggest", "mix", "the", "flour", "with", "--count", "2"])
    output = capsys.readouterr().out
    assert "Suggested continuations:" in output
    assert "1." in output


def test_interactive_clear_and_exit(monkeypatch, capsys):
    from app.cli import interactive_mode

    answers = iter(["clear", "", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    interactive_mode()
    output = capsys.readouterr().out
    assert "Input cleared" in output
    assert "Please enter" in output
    assert "Goodbye!" in output


def test_suggest_count_must_be_positive(capsys):
    from app.cli import main

    try:
        main(["suggest", "mix", "the", "flour", "--count", "0"])
    except SystemExit as error:
        assert error.code == 2

