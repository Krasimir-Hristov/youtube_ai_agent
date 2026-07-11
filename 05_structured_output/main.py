import json
import os
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# OpenRouter Client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)


# All model settings in one place

MODEL = "deepseek/deepseek-v3.2"
MAX_TOKENS = 512
TEMPERATURE = 0.3

# The system promt defines the models role and behaviour

SYSTEM_PROMPT = """You are a movie information assistant.
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
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": movie_name},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""

        return json.loads(raw)

    except json.JSONDecodeError:
        return {"error": "Model returned invalid JSON"}

    except Exception as e:
        return {"error": f"API error: {e}"}


def print_result(result: MovieInfo | MovieError) -> None:
    """Pretty-print a structured movie result to the terminal."""

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    print(f"  Title    : {result['title']}  ({result['year']})")
    print(f"  Director    : {result['director']}")
    print(f"  Genre    : {', '.join(result['genre'])}")
    print(f"  Rating    : {result['rating']}/10")
    print(f"  Summary    : {result['summary']}")


def main() -> None:
    """Demo: test the movie lookup with a few examples."""

    print("\n=== Movie info - Structured output ===\n")

    test_queries = [
        "The Matrix",
        "Inception",
        "Qdsasad123",
        "What is the weather today?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print("_" * 40)
        result = get_movie_info(query)
        print_result(result)
        print()


if __name__ == "__main__":
    main()
