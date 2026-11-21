# Recipe Step Autocompleter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/8ux2stkxr4/badge.svg)](https://ask.thili.ai/b/8ux2stkxr4)

The Recipe Step Autocompleter is a command-line tool designed to assist home cooks. It helps complete incomplete recipe instructions, ensuring no steps are missed and recipes are followed smoothly. This project leverages next-word prediction and text generation capabilities to suggest logical continuations for recipe instructions.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that autocompletes incomplete recipe instructions, helping home cooks follow recipes smoothly.

**Done when.** The CLI tool successfully suggests logical continuations for partial recipe instructions, using a next-word prediction model trained on a corpus of cooking instructions.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Compile a corpus of cooking recipe instructions.**
      Hint: Gather and preprocess text data for your corpus.
- [x] **2. Implement n-gram counting and maximum-likelihood probabilities with smoothing.**
      Hint: Apply n-gram counting and calculate probabilities with smoothing techniques.
- [x] **3. Create a function to generate recipe step continuations.**
      Hint: Develop a function to predict the next word based on the n-gram model.
- [ ] **4. Build a CLI for users to input partial steps and get suggestions.**  
      Hint: Design a command-line interface to interact with your prediction function.

## Milestone 1: Cooking corpus

The original representative cooking instructions are in
`corpus/recipe_instructions.txt`. `app.core.clean_text`, `app.core.tokenize`,
and `app.core.load_corpus` clean the text, lowercase it, remove unwanted
characters, and split it into words. The processed line-based corpus is stored
at `corpus/processed_recipe_instructions.txt`.

Regenerate the processed corpus with:

```bash
python -c "from app.core import milestone_1; milestone_1()"
```

## Milestone 2: N-gram probabilities

`app.core.NGramLanguageModel` trains unigram, bigram, and trigram counts from
the processed corpus. It uses Maximum Likelihood Estimation and Laplace
add-one smoothing so unknown words receive nonzero probabilities. Build or
train the model with:

```bash
python -c "from app.core import milestone_2; model = milestone_2(); print(model.most_likely_next_words('add the'))"
```

## Milestone 3: Continuation generation

`app.core.generate_continuations` ranks short phrases with a bounded probability
beam. It prefers trigram contexts, falls back to bigrams, and finally uses
unigrams when the context has not appeared in the corpus. Generated phrases
cannot repeat a word in the recent three-word window and are capped by
`max_words`.

```bash
python -c "from app.core import milestone_3; print(milestone_3())"
```

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify how many next-word suggestions they want to see.
- Implement a feature to suggest multiple alternative next words, not just the most probable one.
- Add a command to clear the current partial step and start a new one.
