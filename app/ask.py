"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --temperature 0.2 "LoRAとは何ですか？"
    uv run python -m app.ask --interactive
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    if temperature is None:
        response = client.models.generate_content(model=MODEL, contents=question)
    else:
        response = client.models.generate_content(
            model=MODEL,
            contents=question,
            config=types.GenerateContentConfig(temperature=temperature),
        )

    print_usage(response)

    return response.text


def print_usage(response) -> None:
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[tokens] input={usage.prompt_token_count} "
            f"output={usage.candidates_token_count}"
        )


def chat_interactively(temperature: float | None = None) -> None:
    client = genai.Client(api_key=get_api_key())
    config = (
        types.GenerateContentConfig(temperature=temperature)
        if temperature is not None
        else None
    )
    if config is None:
        chat = client.chats.create(model=MODEL)
    else:
        chat = client.chats.create(model=MODEL, config=config)

    print("会話を開始します。終了するには /exit または /quit を入力してください。")
    while True:
        try:
            question = input("質問> ").strip()
        except EOFError:
            print()
            break
        if question.lower() in {"/exit", "/quit"}:
            break
        if not question:
            continue

        response = chat.send_message(question)
        print(response.text)
        print_usage(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--temperature", type=float, help="生成温度（例: 0.2、1.0、1.8）")
    parser.add_argument(
        "--interactive", action="store_true", help="会話履歴を保って続けて質問する"
    )
    args = parser.parse_args()

    if args.interactive and args.question is not None:
        parser.error("--interactive と question は同時に指定できません")

    print(f"[model] {MODEL}")
    if args.temperature is not None:
        print(f"[temperature] {args.temperature}")
    try:
        if args.interactive:
            chat_interactively(temperature=args.temperature)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question, temperature=args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n会話を終了します。")


if __name__ == "__main__":
    main()
