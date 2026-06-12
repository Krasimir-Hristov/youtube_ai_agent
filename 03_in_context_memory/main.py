#  in context memory

import os
from openai import OpenAI
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

conversation_history = []


def ask(user_message: str) -> None:
    conversation_history.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
            + conversation_history,
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

    except Exception as e:
        conversation_history.pop()
        print(f"Error: {e}")


def main():

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
            print("The history is clear!!!")

        print("\n Assistent:")
        print("-" * 40)
        ask(user_input)
        print("-" * 40)


if __name__ == "__main__":
    main()
