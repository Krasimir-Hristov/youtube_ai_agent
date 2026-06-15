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
# ask() now uses streaming - the reply is printed token by token as it arrives instead of waiting for the full response.
# Return type changes from str to None becouse we print directly inside the function instead of returning a string.


def ask(user_message: str) -> None:
    """
    Send a single message and return the models reply.

    Instead of waiting for the full response and returning it,
    this function prints each token as it arrives from the API.


    Args:
        user_message: The text input from the user.

    Returns:
        None output goes directly to stdout via print().0
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
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content

            if content:
                print(content, end="", flush=True)

        print()

    except Exception as e:
        print(f"Error: {e}")
        return


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
        ask(user_input)
        print("-" * 40)


if __name__ == "__main__":
    main()
