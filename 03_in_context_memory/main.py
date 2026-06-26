import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

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

MAX_HISTORY = 10

conversations_history: List[ChatCompletionMessageParam] = []


def ask(user_message: str) -> None:
    """
    Send a single message and return the models reply.

    Instead of waiting for the full response and returning it,
    this function prints each token as it arrives from the API.
    Appends both the user message and assistant reply to conversation_history
    so the model has full context on the next call.


    Args:
        user_message: The text input from the user.

    Returns:
        None output goes directly to stdout via print().0
    """

    conversations_history.append(
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
                *conversations_history,
            ],
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

        conversations_history.append(
            {
                "role": "assistant",
                "content": full_reply,
            }
        )

        if len(conversations_history) > MAX_HISTORY:
            del conversations_history[:-MAX_HISTORY]

    except Exception as e:
        conversations_history.pop()
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

        if user_input.lower() == "clear":
            conversations_history.clear()
            print("---History cleared. Starting fresh. ---\n")
            continue

        print("\n Assitant")
        print("-" * 40)
        ask(user_input)
        print("-" * 40)


if __name__ == "__main__":
    main()
