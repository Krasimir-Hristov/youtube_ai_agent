import os
import json
from pathlib import Path
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)


MODEL = "deepseek/deepseek-v3.2"
MAX_TOKENS = 1024
TEMPERATURE = 0.7
SYSTEM_PROMPT = "Ти си полезен ИИ асистент. Отговаряй само на български, освен ако потребителят не поиска друго."
MAX_HISTORY = 10
HISTORY_FILE = Path("conversation_history.json")

conversation_history: list[ChatCompletionMessageParam] = []


def load_history() -> None:
    if not HISTORY_FILE.exists():
        return


try:
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversation_history.extend(data)
    print(f" Loaded {len(data)} messages from history!")


except Exception as e:
    print(f"Cannot load the history: {e}")


def save_history() -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(list(conversation_history), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Cannot save the history: {e}")


def ask(user_message: str) -> None:
    conversation_history.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    try:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *conversation_history,
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )

        full_reply = ""

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_reply += content

        print()

        conversation_history.append(
            {
                "role": "assistant",
                "content": full_reply,
            }
        )

        if len(conversation_history) > MAX_HISTORY:
            del conversation_history[:-MAX_HISTORY]

        save_history()
    except Exception as e:
        conversation_history.pop()
        print(f"Error: {e}")


def main():

    load_history()

    print("\n=== AI Assitant with OpenRouter ===")
    print(f"Model: {MODEL}")
    print("Ask or write 'exit' to exit")

    while True:

        user_input = input("user: ").strip()

        if user_input == "exit":
            print("Chao!!!")
            break
        if not user_input:
            continue

        if user_input.lower() == "clear":
            conversation_history.clear()
            HISTORY_FILE.unlink(missing_ok=True)
            print("The history is clear!!!")
            continue

        print("\n Assistent:")
        print("-" * 40)
        ask(user_input)
        print("-" * 40)


if __name__ == "__main__":
    main()
