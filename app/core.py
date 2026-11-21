"""Core functions for the Recipe Step Autocompleter."""

import re
from pathlib import Path


# --- Milestone 1: Compile a corpus of cooking recipe instructions.
def clean_text(text: str) -> str:
    """Convert text to lowercase and remove punctuation or unwanted characters."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Clean text and split it into words."""
    if not text or not isinstance(text, str):
        return []
    return clean_text(text).split()


def preprocess_corpus(text: str) -> list[list[str]]:
    """Split a corpus into sentences and tokenize each sentence."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [tokenize(line) for line in text.splitlines() if tokenize(line)]


def load_corpus(path: str | Path) -> list[list[str]]:
    """Read and preprocess a line-based cooking-instruction corpus."""
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    return preprocess_corpus(corpus_path.read_text(encoding="utf-8"))


def save_processed_corpus(corpus: list[list[str]], path: str | Path) -> Path:
    """Save a processed corpus as one space-separated instruction per line."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(words) for words in corpus if words]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def milestone_1(
    source: str | Path = "corpus/recipe_instructions.txt",
    destination: str | Path = "corpus/processed_recipe_instructions.txt",
) -> Path:
    """Compile and preprocess the cooking instruction corpus."""
    corpus = load_corpus(source)
    if not corpus:
        raise ValueError("The source corpus is empty")
    return save_processed_corpus(corpus, destination)


# --- Milestone 2: Implement n-gram counting and maximum-likelihood probabilities with smoothing.
# Hint: Apply n-gram counting and calculate probabilities with smoothing techniques.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Create a function to generate recipe step continuations.
# Hint: Develop a function to predict the next word based on the n-gram model.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Build a CLI for users to input partial steps and get suggestions.
# Hint: Design a command-line interface to interact with your prediction function.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
