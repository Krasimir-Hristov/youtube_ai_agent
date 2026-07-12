import json
import os
from typing import TypedDict, List
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

load_dotenv()


# OpenRouter Client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)


# All model settings in one place

MODEL = "deepseek/deepseek-v3.2"

SYSTEM_PROMPT = (
    "You are a helpful AI assitant."
    "Reply in Bulgarian unless the user asks otherwise."
    "When the user asks about movie, use the get_movie_info tool."
    "Always include ALL fields: title, year, genre, director, rating, and summary"
)
MAX_TOKENS = 1024
TEMPERATURE = 0.7
MAX_HISTORY = 10
HISTORY_FILE = Path("conversation_history.json")

# Tools constants
MOVIE_MODEL = "deepseek/deepseek-v3.2"
MOVIE_TEMPERATURE = 0.3
MOVIE_MAX_TOKENS = 512

MOVIE_SYSTEM_PROMPT = """You are a movie information assistant.
You have ONE job: return structured data about movies.
 
When given a movie name, respond with ONLY this JSON structure:
{
  "title": "the movie title in the original language",
  "year": 1994,
  "genre": ["genre1", "genre2"],
  "director": "director full name",
  "rating": 8.5,
  "summary": "two or three sentence summary in Bulgarian"
}
 
If the movie is not found or you are not sure, return:
{
  "error": "Movie not found"
}
 
If the user asks about ANYTHING other than movies, return:
{
  "error": "I only handle movie lookups"
}
 
No prose. No explanation. No markdown. Raw JSON only."""


class MovieInfo(TypedDict):
    title: str
    year: int
    genre: list[str]
    director: str
    rating: float
    summary: str


class MovieError(TypedDict):
    error: str


TOOL: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_movie_info",
            "description": (
                "Look um information about a movie."
                "Use this when the user asks about a specific movie"
                "Its director, genre, rating, or plot"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_name": {
                        "type": "string",
                        "description": "The name of the movie to look up.",
                    }
                },
                "required": ["movie_name"],
            },
        },
    }
]


conversations_history: List[ChatCompletionMessageParam] = []


def load_history() -> None:
    """Load conversation history from JSON file if ti exist"""

    if not HISTORY_FILE.exists():
        return

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        conversations_history.extend(data)
        print(f"Loaded {len(data)} massages from history.")

        # [1,2].append([3,4]) -> [1,2,[3,4]]
        # [1,2].extend([3,4]) -> [1,2,3,4]
    except Exception as e:
        print(f"Cloude not load history: {e}")


def save_history() -> None:
    """Save current conversatons history to JSON file."""

    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:

            json.dump(list(conversations_history), f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Could not save history: {e}")


def get_movie_info(movie_name: str) -> MovieInfo | MovieError:
    """Look up a movie and return structured data.

    This function only handles movie lookups.

    Args:
    movie_name : The name of the movie to look up.

    Returns:
    A MovieInfo TypeDict on success, MovieError otherwise.
    """
    try:
        response = client.chat.completions.create(
            model=MOVIE_MODEL,
            messages=[
                {"role": "system", "content": MOVIE_SYSTEM_PROMPT},
                {"role": "user", "content": movie_name},
            ],
            max_tokens=MOVIE_MAX_TOKENS,
            temperature=MOVIE_TEMPERATURE,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""

        return json.loads(raw)

    except json.JSONDecodeError:
        return {"error": "Model returned invalid JSON"}

    except Exception as e:
        return {"error": f"API error: {e}"}


def ask(user_message: str) -> None:
    """Send a message, handle tool calls, stream the final reply.


    1. Send request with tools available.
    2. If model requests a tool, execute it.
    3. Sen tool result back to model.
    4. Stream the final response.


    Args:
        user_message: The text input from the user.

    Returns:
        None the output is directly printed.
    """

    conversations_history.append({"role": "user", "content": user_message})

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversations_history,
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            tools=TOOL,
        )

        message = response.choices[0].message

        if response.choices[0].finish_reason == "tool_calls":

            if not message.tool_calls:
                return

            tool_call = message.tool_calls[0]

            if not isinstance(tool_call, ChatCompletionMessageToolCall):
                return

            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            conversations_history.append(message.model_dump())  # type: ignore[arg-type]

            if tool_name == "get_movie_info":
                print(f"Tool calling with argument: {tool_args['movie_name']}")

                tool_result = get_movie_info(tool_args["movie_name"])
                print(f" Tool Result {json.dumps(tool_result, ensure_ascii=False)}")
            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            conversations_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                },
            )

            final_response = client.chat.completions.create(
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

            for chunk in final_response:
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

        else:
            full_reply = message.content or ""
            print(full_reply)

            conversations_history.append(
                {
                    "role": "assistant",
                    "content": full_reply,
                }
            )

            if len(conversations_history) > MAX_HISTORY:
                del conversations_history[:-MAX_HISTORY]

            save_history()

    except Exception as e:
        conversations_history.pop()
        print(f"Error: {e}")


def main() -> None:
    """Run the interactive chat loop in the terminal."""

    load_history()

    print("\n=== AI Chatbot with tool calling ===")
    print(f"Model: {MODEL}")
    print(f"Tools: get_movie_info")
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
            HISTORY_FILE.unlink(missing_ok=True)
            print("---History cleared. Starting fresh. ---\n")
            continue

        print("\n Assitant")
        print("-" * 40)
        ask(user_input)
        print("-" * 40)


if __name__ == "__main__":
    main()
