"""Command-line chat with a locally trained JakeLLM checkpoint."""

import argparse

from inference.generate import generate_text, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description="JakeLLM local text inference")
    parser.add_argument("--checkpoint", default="checkpoints/jakellm.pt")
    parser.add_argument("--tokenizer-dir", default="tokenizer")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt", help="Generate once and exit (otherwise interactive)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    model, tokenizer = load_model(args.checkpoint, args.tokenizer_dir, args.device)

    def respond(prompt):
        return generate_text(model, tokenizer, prompt, args.max_new_tokens,
                             args.temperature, args.top_k)

    if args.prompt is not None:
        print(respond(args.prompt))
        return

    print("JakeLLM pronto! Digite 'sair' para encerrar.")
    while True:
        try:
            prompt = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"sair", "exit", "quit"}:
            return
        if not prompt:
            continue
        try:
            print("JakeLLM:", respond(prompt))
        except ValueError as exc:
            print(f"Entrada inválida: {exc}")


if __name__ == "__main__":
    main()
