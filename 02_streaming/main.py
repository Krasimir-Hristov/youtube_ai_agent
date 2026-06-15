import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# OpenRouter Client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)


# All model settings in one place

MODEL = "deepseek/deepseek-v3.2"
MAX_TOKENS = 1024
TEMPERATURE = 0.7

# The system prompt defines the models role and behaviour

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Reply in Bulgarian unless the user asks otherwise."
)


# Core function


def ask_stream(user_message: str) -> None:
    """
    Send a single message and stream the model's reply token by token.

    Args:
        user_message: The text input from the user.
    """

    try:

        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)

        print()  # newline after streaming ends

    except Exception as e:
        print(f"Error: {e}")


# Entry point


def main() -> None:
    """Run the interactive streaming chat loop in the terminal."""

    print("\n=== AI Chatbot (Streaming) - Openrouter ===")
    print(f"Model: {MODEL}")
    print("Type your question or type 'exit' to quit.")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("\nAssistant:")
        print("-" * 40)
        ask_stream(user_input)
        print("-" * 40)


if __name__ == "__main__":
    main()
