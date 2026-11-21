"""Command-line entry point for the Recipe Step Autocompleter."""
import argparse

from app.core import generate_continuations, milestone_2


def print_suggestions(partial_instruction: str, suggestions: int = 3) -> list[str]:
    """Print ranked continuations and return them for testing."""
    model = milestone_2()
    continuations = generate_continuations(
        partial_instruction,
        model,
        suggestions=suggestions,
    )
    if not continuations:
        print("No suggestions could be generated for that instruction.")
        return []
    print("\nSuggested continuations:")
    for number, continuation in enumerate(continuations, start=1):
        print(f"{number}. {partial_instruction.strip()} {continuation}")
    return continuations


def interactive_mode(default_suggestions: int = 3) -> None:
    """Run a simple prompt until the user exits."""
    print("Recipe Step Autocompleter")
    print("Type 'clear' to start over, or 'exit' or 'quit' to leave.")

    while True:
        try:
            partial_instruction = input(
                "\nEnter an incomplete recipe instruction:\n> "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if partial_instruction.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return
        if partial_instruction.lower() == "clear":
            print("Input cleared. Enter a new recipe instruction.")
            continue
        if not partial_instruction:
            print("Please enter an incomplete recipe instruction.")
            continue

        print_suggestions(partial_instruction, default_suggestions)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="recipe-step-autocompleter",
        description="Autocomplete incomplete cooking recipe instructions.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("hello", help="check that everything is wired up")

    suggest_parser = sub.add_parser("suggest", help="suggest continuations")
    suggest_parser.add_argument("instruction", nargs="+", help="partial recipe instruction")
    suggest_parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=3,
        help="number of suggestions (default: 3)",
    )

    sub.add_parser("interactive", help="start an interactive autocomplete session")

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Recipe Step Autocompleter is ready. Open README.md for your milestones.")
    elif args.command == "suggest":
        if args.count < 1:
            parser.error("--count must be a positive integer")
        print_suggestions(" ".join(args.instruction), args.count)
    elif args.command == "interactive":
        interactive_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
