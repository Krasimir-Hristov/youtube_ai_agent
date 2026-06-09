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


def ask(user_message: str) -> str | None:

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

        return response.choices[0].message.content

    except Exception as e:
        return f"Error message: {e}"


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

        print("\n Assistent:")
        print("-" * 40)
        answer = ask(user_input)
        print(answer)
        print("-" * 40)


if __name__ == "__main__":
    main()
