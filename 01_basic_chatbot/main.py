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

# The system promt defines the models role and behaviour

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Reply in Bulgarian unless the user asks otherwise."
)


# Core function


def ask(user_message: str) -> str | None:
    """
    Send a single message and return the models reply.


    Args:
        user_message: The text input from the user.

    Returns:
        The models reply as a plain string.
    """

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        return response.choices[0].message.content or ""

    except Exception as e:
        return f"Error: {e}"


# Entry point


def main() -> None:
    """Run the interactive chat loop in the terminal."""

    print("\n=== AI Chatbot - Openrouter ===")
    print(f"Model: {MODEL}")
    print("Type tour question or type 'exit' to quit.")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("\n Assitant")
        print("-" * 40)
        answer = ask(user_input)
        print(answer)
        print("-" * 40)


if __name__ == "__main__":
    main()
