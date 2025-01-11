import os
import json
import openai
import requests

################################################################################
# 1) Setup:
#    - Make sure you install openai and requests (e.g., pip install openai requests).
#    - Set environment variable OPENAI_API_KEY with your OpenAI key.
#    - Replace <your_perplexity_token> and <your_firecrawl_token> below with your
#      actual tokens.
################################################################################

# For illustration only; in a production system, store tokens securely (e.g., environment variables or vault).
PERPLEXITY_API_TOKEN = "<your_perplexity_token>"
FIRECRAWL_API_TOKEN = "<your_firecrawl_token>"

openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...")

################################################################################
# 2) Define Python functions that call external APIs
#
#    These will be exposed (via function calling) to the main ChatCompletion call,
#    so the model can decide if it wants to invoke them to gather data.
################################################################################

def call_perplexity(query: str) -> str:
    """
    Calls the Perplexity AI API with the given query.
    Returns the text content from the model’s answer.
    """
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "search_recency_filter": "month",
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"


def call_firecrawl_scrape(url: str, wait_for: int = 0) -> str:
    """
    Calls Firecrawl's /v1/scrape endpoint to scrape a single URL.
    Returns the resulting markdown content if successful.
    """
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "waitFor": wait_for,
        "location": {"country": "US", "languages": ["en"]},
        "removeBase64Images": True,
    }
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("success"):
            return data["data"].get("markdown", "No markdown content was returned.")
        else:
            return f"Firecrawl error: {data.get('warning', 'Unknown error')}"
    except Exception as e:
        return f"Error calling Firecrawl API: {str(e)}"


def call_firecrawl_search(query: str, limit: int = 5) -> str:
    """
    Calls Firecrawl's /search endpoint to perform a SERP search with optional scraping.
    Returns a JSON-string of the combined search results.
    By default, it returns up to 5 results.
    """
    endpoint = "https://api.firecrawl.dev/v1/search"
    payload = {
        "query": query,
        "limit": limit,
        # If you'd like to also fetch full markdown each time, you can add "scrapeOptions"
        # with "formats": ["markdown"]. For large queries, you might want to do carefully.
        #
        # "scrapeOptions": {
        #     "formats": ["markdown"],
        #     "onlyMainContent": True,
        #     "removeBase64Images": True
        # },
    }
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        # Return as JSON for clarity
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error calling Firecrawl Search API: {str(e)}"


def call_openai_o1(prompt: str) -> str:
    """
    Calls OpenAI with model='o1' to handle more advanced reasoning or sub-queries.
    """
    try:
        sub_response = openai.ChatCompletion.create(
            model="o1",
            messages=[{"role": "user", "content": prompt}],
        )
        return sub_response.choices[0].message["content"]
    except Exception as e:
        return f"Error calling OpenAI model='o1': {str(e)}"


def call_openai_gpt4o(prompt: str) -> str:
    """
    Calls OpenAI with model='gpt-4o' for advanced GPT-4 style reasoning or sub-queries.
    """
    try:
        sub_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return sub_response.choices[0].message["content"]
    except Exception as e:
        return f"Error calling OpenAI model='gpt-4o': {str(e)}"


################################################################################
# 3) Define the JSON schema for each function (tool).
#    We provide these definitions to the main ChatCompletion endpoint so the model
#    can choose to call them if it decides they are relevant.
################################################################################

tools = [
    {
        "type": "function",
        "function": {
            "name": "call_perplexity",
            "description": (
                "Use this to do a broader web-based query via Perplexity. Provide a 'query' "
                "that you want to research or get an answer for."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A question or search query to be sent to Perplexity",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_firecrawl_scrape",
            "description": (
                "Use this to scrape a single webpage with Firecrawl. Provide the full URL and an "
                "optional wait time in ms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to scrape"},
                    "wait_for": {
                        "type": "number",
                        "description": "Wait in ms before scraping (0 by default)."
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_firecrawl_search",
            "description": (
                "Use this to search the web (SERP) with Firecrawl. Provide a 'query' and optionally "
                "a 'limit' for the number of results. Returns JSON data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A search query, e.g. 'best pizza in NYC'"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Max number of results to retrieve; default=5."
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_openai_o1",
            "description": (
                "Use this if you want to sub-question a more advanced reasoning model named 'o1'. "
                "Provide a 'prompt' that you want to analyze with model='o1'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A prompt or question to the 'o1' model"
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_openai_gpt4o",
            "description": (
                "Use this if you want to sub-question a GPT-4 style advanced model called 'gpt-4o'. "
                "Provide a 'prompt' that you want to analyze with this model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A prompt or question to the 'gpt-4o' model"
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    },
]

################################################################################
# 4) Main conversation loop
#
#    The user asks a question. We pass the conversation (including function defs)
#    to the "o1" model. If the model decides a function call is appropriate, we
#    detect it, run the function, feed results back, letting the model produce an
#    informed final answer. 
#
################################################################################

def main():
    print("Welcome! Ask any question. The system will use model='o1' to answer.")
    user_question = input("User: ").strip()

    # Start conversation with system + user messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are an advanced assistant using model='o1'. You can call the following functions if needed "
                "to gather additional information or handle sub-queries. Think carefully about when to call them. "
                "When you have enough info, produce the best possible final answer for the user."
            ),
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    # We'll do up to a few iterations. The assistant can request function calls.
    for _ in range(5):
        response = openai.ChatCompletion.create(
            model="o1",
            messages=messages,
            tools=tools,
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        tool_calls = msg.get("tool_calls")
        assistant_content = msg.get("content")

        if tool_calls:
            # The model is requesting one or more function calls
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                arguments_json = tc["function"]["arguments"]

                # Attempt to parse the function arguments:
                try:
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError:
                    arguments = {}

                # Execute the matching function:
                if func_name == "call_perplexity":
                    query = arguments.get("query", "")
                    result = call_perplexity(query)

                elif func_name == "call_firecrawl_scrape":
                    url = arguments.get("url", "")
                    wait_for = arguments.get("wait_for", 0)
                    result = call_firecrawl_scrape(url, wait_for)

                elif func_name == "call_firecrawl_search":
                    query = arguments.get("query", "")
                    limit = arguments.get("limit", 5)
                    result = call_firecrawl_search(query, limit)

                elif func_name == "call_openai_o1":
                    subprompt = arguments.get("prompt", "")
                    result = call_openai_o1(subprompt)

                elif func_name == "call_openai_gpt4o":
                    subprompt = arguments.get("prompt", "")
                    result = call_openai_gpt4o(subprompt)

                else:
                    result = f"Tool {func_name} is not implemented."

                # Now feed the result back as a "tool" role
                tool_result_message = {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc["id"],
                }
                messages.append(msg)  # The assistant's request
                messages.append(tool_result_message)

        elif finish_reason == "stop":
            # The model gave a final answer
            if assistant_content:
                print("\nAssistant:\n" + assistant_content)
            else:
                print("\nAssistant provided no content.")
            break

        elif finish_reason == "length":
            # The conversation got cut off or is too long
            print("The model's response got cut off. Stopping...")
            break

        elif finish_reason == "tool_calls":
            # The model ended after generating function calls. Continue in loop 
            # so we can feed back the tool results as messages.
            pass

        else:
            # Possibly an unrecognized reason
            if assistant_content:
                print("\nAssistant (final):\n" + assistant_content)
            else:
                print("\nAssistant provided no content.")
            break

    print("\n--- End of conversation ---")


if __name__ == "__main__":
    main()
