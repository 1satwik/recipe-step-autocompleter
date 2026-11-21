"""Core functions for the Recipe Step Autocompleter."""

import re
from collections import Counter
import math
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


# --- Milestone 2: Implement n-gram counting and probabilities with smoothing.
def count_ngrams(corpus: list[list[str]] | str, n: int) -> Counter:
    """Count n-grams as (context tuple, next word) pairs."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")

    if isinstance(corpus, str):
        corpus = preprocess_corpus(corpus)
    if not isinstance(corpus, list):
        raise TypeError("corpus must be a string or a list of word lists")

    counts = Counter()
    for sentence in corpus:
        if not isinstance(sentence, list):
            raise TypeError("each sentence must be a list of words")
        words = [str(word).lower() for word in sentence if str(word).strip()]
        for index in range(len(words) - n + 1):
            context = tuple(words[index:index + n - 1])
            counts[(context, words[index + n - 1])] += 1
    return counts


def count_unigrams(corpus: list[list[str]] | str) -> Counter:
    """Count individual words."""
    return Counter(word for sentence in (
        preprocess_corpus(corpus) if isinstance(corpus, str) else corpus
    ) for word in sentence)


def count_bigrams(corpus: list[list[str]] | str) -> Counter:
    """Count bigrams."""
    return count_ngrams(corpus, 2)


def count_trigrams(corpus: list[list[str]] | str) -> Counter:
    """Count trigrams."""
    return count_ngrams(corpus, 3)


def mle_probability(
    next_word: str,
    context: tuple[str, ...] | list[str] | str,
    counts: Counter,
    vocabulary: set[str] | list[str] | tuple[str, ...] | None = None,
    alpha: float = 0.0,
) -> float:
    """Calculate MLE probability, optionally applying Laplace smoothing."""
    if not isinstance(next_word, str):
        raise TypeError("next_word must be a string")
    if isinstance(context, str):
        context = tuple(context.lower().split())
    else:
        context = tuple(str(word).lower() for word in context)

    next_word = next_word.lower()
    context_count = sum(
        count for (ngram_context, _), count in counts.items()
        if ngram_context == context
    )
    matching_count = counts.get((context, next_word), 0)

    if alpha == 0:
        return matching_count / context_count if context_count else 0.0

    if alpha < 0:
        raise ValueError("alpha must be zero or positive")
    vocabulary_size = len(vocabulary) if vocabulary is not None else len({
        word for (_, word) in counts
    })
    if vocabulary_size == 0:
        return 0.0
    return (matching_count + alpha) / (context_count + alpha * vocabulary_size)


class NGramLanguageModel:
    """A unigram, bigram and trigram language model using MLE."""

    UNK = "<UNK>"

    def __init__(
        self,
        corpus: list[list[str]] | str,
        smoothing: float = 1.0,
    ) -> None:
        if smoothing < 0:
            raise ValueError("smoothing must be zero or positive")
        if isinstance(corpus, str):
            if "\n" in corpus:
                corpus = preprocess_corpus(corpus)
            else:
                corpus = [tokenize(corpus)]
        if not isinstance(corpus, list):
            raise TypeError("corpus must be a string or a list of word lists")

        clean_corpus = []
        for sentence in corpus:
            if not isinstance(sentence, (list, tuple)):
                raise TypeError("each sentence must be a list or tuple of words")
            words = [
                str(word).lower()
                for word in sentence
                if str(word).strip()
            ]
            if words:
                clean_corpus.append(words)
        if not clean_corpus:
            raise ValueError("corpus must contain at least one word")

        self.corpus = clean_corpus
        self.smoothing = float(smoothing)
        self.vocabulary = set(
            word for sentence in clean_corpus for word in sentence
        )
        self.vocabulary.add(self.UNK)
        self.unigram_counts = count_unigrams(clean_corpus)
        self.bigram_counts = count_bigrams(clean_corpus)
        self.trigram_counts = count_trigrams(clean_corpus)

    def normalize_word(self, word: str) -> str:
        """Map an unknown word to the model's unknown-word token."""
        if not isinstance(word, str):
            raise TypeError("word must be a string")
        word = word.lower().strip()
        return word if word in self.vocabulary else self.UNK

    def probability(
        self,
        next_word: str,
        context: tuple[str, ...] | list[str] | str = (),
    ) -> float:
        """Return a Laplace-smoothed next-word probability."""
        if isinstance(context, str):
            context_words = tokenize(context)
        else:
            context_words = [
                str(word).lower()
                for word in context
                if str(word).strip()
            ]
        context_words = tuple(self.normalize_word(word) for word in context_words)
        next_word = self.normalize_word(next_word)

        if len(context_words) >= 2:
            counts = self.trigram_counts
            context_key = context_words[-2:]
        elif len(context_words) == 1:
            counts = self.bigram_counts
            context_key = context_words
        else:
            counts = {
                ((), word): count
                for word, count in self.unigram_counts.items()
            }
            context_key = ()

        return mle_probability(
            next_word,
            context_key,
            counts,
            vocabulary=self.vocabulary,
            alpha=self.smoothing,
        )

    def next_word_probabilities(
        self,
        context: tuple[str, ...] | list[str] | str = (),
    ) -> dict[str, float]:
        """Calculate probabilities for every known word and <UNK>."""
        return {
            word: self.probability(word, context)
            for word in sorted(self.vocabulary)
        }

    def most_likely_next_words(
        self,
        context: tuple[str, ...] | list[str] | str = (),
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """Return next words ranked by probability."""
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        ranked = sorted(
            self.next_word_probabilities(context).items(),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:limit]


def milestone_2(
    source: str | Path = "corpus/processed_recipe_instructions.txt",
    smoothing: float = 1.0,
) -> NGramLanguageModel:
    """Train a reusable unigram, bigram and trigram model."""
    corpus = load_corpus(source)
    if not corpus:
        raise ValueError("The source corpus is empty")
    return NGramLanguageModel(corpus, smoothing=smoothing)


# --- Milestone 3: Create a function to generate recipe step continuations.
def generate_continuations(
    partial_instruction: str,
    model: NGramLanguageModel,
    suggestions: int = 3,
    max_words: int = 12,
) -> list[str]:
    """Generate ranked continuation phrases using trigram, bigram and unigram backoff."""
    if not isinstance(partial_instruction, str):
        raise TypeError("partial_instruction must be a string")
    if not isinstance(model, NGramLanguageModel):
        raise TypeError("model must be an NGramLanguageModel")
    if not isinstance(suggestions, int) or suggestions < 1:
        raise ValueError("suggestions must be a positive integer")
    if not isinstance(max_words, int) or not 1 <= max_words <= 50:
        raise ValueError("max_words must be an integer from 1 to 50")

    partial_words = tokenize(partial_instruction)
    if not partial_words:
        return []

    beam = [(partial_words, 0.0)]
    beam_width = max(suggestions * 3, 9)

    def candidates(context_words: list[str]) -> list[tuple[str, float]]:
        normalized = [model.normalize_word(word) for word in context_words]

        if len(normalized) >= 2:
            context = tuple(normalized[-2:])
            matching = [
                (word, count)
                for (ngram_context, word), count in model.trigram_counts.items()
                if ngram_context == context
            ]
            if matching:
                return sorted(
                    ((word, model.probability(word, context)) for word, _ in matching),
                    key=lambda item: (-item[1], item[0]),
                )[:beam_width]

        if normalized:
            context = tuple(normalized[-1:])
            matching = [
                (word, count)
                for (ngram_context, word), count in model.bigram_counts.items()
                if ngram_context == context
            ]
            if matching:
                return sorted(
                    ((word, model.probability(word, context)) for word, _ in matching),
                    key=lambda item: (-item[1], item[0]),
                )[:beam_width]

        ranked = model.most_likely_next_words((), beam_width)
        return [
            (word, probability)
            for word, probability in ranked
            if word != model.UNK and word not in partial_words
        ]

    for _ in range(max_words):
        expanded = []
        for words, score in beam:
            for word, probability in candidates(words):
                generated_words = words[len(partial_words):]
                if word == model.UNK or word in generated_words[-3:]:
                    continue
                safe_probability = max(probability, 1e-12)
                expanded.append((words + [word], score + math.log(safe_probability)))
        if not expanded:
            break
        beam = sorted(
            expanded,
            key=lambda item: (-item[1] / len(item[0]), item[0]),
        )[:beam_width]

    ranked_results = []
    seen = set()
    for words, score in sorted(
        beam,
        key=lambda item: (-item[1] / len(item[0]), item[0]),
    ):
        continuation = " ".join(words[len(partial_words):]).strip()
        if continuation and continuation not in seen:
            seen.add(continuation)
            ranked_results.append(continuation)
        if len(ranked_results) >= suggestions:
            break
    return ranked_results


def milestone_3(
    source: str | Path = "corpus/processed_recipe_instructions.txt",
    partial_instruction: str = "add the tomatoes to",
    suggestions: int = 3,
) -> list[str]:
    """Train the model and generate continuations for an incomplete instruction."""
    model = milestone_2(source)
    return generate_continuations(partial_instruction, model, suggestions=suggestions)


# --- Milestone 4: Build a CLI for users to input partial steps and get suggestions.
# Hint: Design a command-line interface to interact with your prediction function.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")
