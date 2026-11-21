"""Tests for the project milestones."""
import pytest

# Milestone 1: Compile a corpus of cooking recipe instructions.
def test_clean_text_lowercase_and_removes_punctuation():
    from app.core import clean_text

    assert clean_text("  Add_The CHOPPED onions!  ") == "add the chopped onions"


def test_tokenize_returns_clean_words():
    from app.core import tokenize

    assert tokenize("Mix 2 cups of flour, please.") == ["mix", "2", "cups", "of", "flour", "please"]
    assert tokenize("") == []


def test_load_save_and_milestone_1(tmp_path):
    from app.core import load_corpus, milestone_1, save_processed_corpus

    source = tmp_path / "source.txt"
    destination = tmp_path / "processed.txt"
    source.write_text("Add the chopped onions to the pan.\nStir gently until soft!\n", encoding="utf-8")

    corpus = load_corpus(source)
    assert corpus == [
        ["add", "the", "chopped", "onions", "to", "the", "pan"],
        ["stir", "gently", "until", "soft"],
    ]
    assert save_processed_corpus(corpus, destination).exists()
    assert destination.read_text(encoding="utf-8").strip().splitlines()[0] == "add the chopped onions to the pan"

    project_source = tmp_path / "recipe.txt"
    project_destination = tmp_path / "output.txt"
    project_source.write_text("Cook the rice.\n", encoding="utf-8")
    assert milestone_1(project_source, project_destination).exists()


def test_preprocessing_rejects_invalid_inputs(tmp_path):
    from app.core import clean_text, load_corpus, preprocess_corpus

    with pytest.raises(TypeError):
        clean_text(None)
    with pytest.raises(TypeError):
        preprocess_corpus(None)
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing.txt")


# Milestone 2: Implement n-gram counting and maximum-likelihood probabilities with smoothing.
def test_ngram_counts_include_unigrams_bigrams_and_trigrams():
    from app.core import count_bigrams, count_trigrams, count_unigrams

    corpus = [["add", "the", "oil"], ["add", "the", "salt"]]
    assert count_unigrams(corpus)["add"] == 2
    assert count_bigrams(corpus)[(("add",), "the")] == 2
    assert count_trigrams(corpus)[(("add", "the"), "oil")] == 1


def test_mle_and_laplace_probabilities():
    from app.core import mle_probability

    counts = {
        (("add",), "the"): 3,
        (("add",), "salt"): 1,
    }
    assert mle_probability("the", ("add",), counts) == 0.75
    assert mle_probability("the", ("add",), counts, vocabulary={"the", "salt", "oil"}, alpha=1) == 4 / 7
    assert mle_probability("oil", ("add",), counts, vocabulary={"the", "salt", "oil"}, alpha=1) == 1 / 7


def test_model_smoothing_and_unknown_words():
    from app.core import NGramLanguageModel

    model = NGramLanguageModel([["add", "the", "oil"], ["add", "the", "salt"]])
    assert model.probability("the", ()) > 0
    assert model.probability("unknownword", ("add",)) > 0
    probabilities = model.next_word_probabilities(("not", "in", "corpus"))
    assert all(probability > 0 for probability in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-12


def test_model_validation_and_empty_corpus():
    from app.core import NGramLanguageModel

    with pytest.raises(ValueError):
        NGramLanguageModel([])
    with pytest.raises(ValueError):
        NGramLanguageModel([["add"]], smoothing=-1)


def test_milestone_2_loads_project_corpus():
    from app.core import NGramLanguageModel, milestone_2

    model = milestone_2()
    assert isinstance(model, NGramLanguageModel)
    assert model.vocabulary


# Milestone 3: Create a function to generate recipe step continuations.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build a CLI for users to input partial steps and get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")

