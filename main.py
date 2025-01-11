
import os
import json
import openai
import requests

################################################################################
# 1) Setup:
#    - Make sure you install openai and requests (e.g., pip install openai requests).
#    - Set environment variable OPENAI_API_KEY with your OpenAI key
#    - Replace <your_perplexity_token> and <your_firecrawl_token> below.
################################################################################

# For illustration only; in a production system, store tokens securely (e.g., an environment variable).
PERPLEXITY_API_TOKEN = "<your_perplexity_token>"
FIRECRAWL_API_TOKEN = "<your_firecrawl_token>"

openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...")

################################################################################
# 2) Define Python functions that call external APIs
#
#    We will expose these (via function calling) to the OpenAI model. The model
#    can decide to call them by providing JSON arguments that match the schema.
#
################################################################################

def call_perplexity(query: str) -> str:
    """
    Calls the Perplexity AI API with the given query.
    Returns the text content from the model’s answer.
    """
    # You will need a valid Perplexity token. 
    # This example just merges the arguments with minimal parameters.
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
        # Parse the first answer from the Perplexity response.
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"


def call_firecrawl_scrape(url: str, wait_for: int = 0) -> str:
    """
    Calls Firecrawl's /v1/scrape endpoint to scrape a URL.
    Returns the resulting markdown content if successful.
    """
    endpoint = "https://api.firecrawl.dev/v1/scrape"

    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "waitFor": wait_for,  # how many ms to wait before scraping
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


def call_openai_subfunction(prompt: str) -> str:
    """
    Calls the OpenAI API (could be a sub-request) using model o1 with the user-provided prompt.
    Returns the model's response text.
    """
    try:
        sub_response = openai.ChatCompletion.create(
            model="o1",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return sub_response.choices[0].message["content"]
    except Exception as e:
        return f"Error calling OpenAI subfunction: {str(e)}"

################################################################################
# 3) Define the JSON schema for each function (tool). These definitions will be
#    provided to the main ChatCompletion endpoint so that the model can choose
#    to call them.
################################################################################

tools = [
    {
        "type": "function",
        "function": {
            "name": "call_perplexity",
            "description": (
                "Use this to research or summarize information from the web. "
                "Takes a string 'query' that you want to ask Perplexity about."
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
                "Use this to scrape a website with Firecrawl. Provide the full URL and optionally a wait time in ms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to scrape"},
                    "wait_for": {
                        "type": "number",
                        "description": "Wait in ms before scraping (optional). Defaults to 0",
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
            "name": "call_openai_subfunction",
            "description": (
                "Use this to reason about a sub-question with OpenAI using model='o1' inside the overall conversation. "
                "Provide a prompt that the subfunction will send to the model. E.g. to refine or summarize something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A prompt or question for the subfunction call to openai again (model='o1')",
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
#    The user asks a question. We pass the conversation history (including these
#    function definitions) to the "o1" model. If the model decides a function
#    call is appropriate, we detect it, execute the function, and feed the result
#    back. Finally, we display the final answer.
#
################################################################################

def main():
    print("Welcome! Ask any question. The system will use model='o1' to answer.")
    user_question = input("User: ").strip()

    # Start conversation with system and user messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant with deep reasoning abilities. "
                "You can call the following functions if needed to gather more information. "
                "First, consider carefully if external data or sub-queries are needed for thorough reasoning. "
                "When you are confident you have enough information, provide the best possible answer. "
                "If you need more info, use the provided tool calls."
            ),
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    # We'll do up to a few iterations: the assistant can request tool calls,
    # we call them outside, feed results back as a "tool" role, and so forth.
    for _ in range(5):  
        # We supply tools in the request
        response = openai.ChatCompletion.create(
            model="o1",
            messages=messages,
            tools=tools,
        )

        msg = response.choices[0].message

        finish_reason = response.choices[0].finish_reason
        tool_calls = msg.get("tool_calls")  # If the model decided to call a function
        assistant_content = msg.get("content")

        if tool_calls:
            # The model is requesting one or more function calls
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                arguments_json = tc["function"]["arguments"]

                # Parse arguments
                try:
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError:
                    # Provide a fallback if the model's JSON is invalid
                    arguments = {}

                # Execute the corresponding Python function
                if func_name == "call_perplexity":
                    query = arguments.get("query", "")
                    result = call_perplexity(query)

                elif func_name == "call_firecrawl_scrape":
                    url = arguments.get("url", "")
                    wait_for = arguments.get("wait_for", 0)
                    result = call_firecrawl_scrape(url, wait_for)

                elif func_name == "call_openai_subfunction":
                    subprompt = arguments.get("prompt", "")
                    result = call_openai_subfunction(subprompt)

                else:
                    result = f"Tool {func_name} is not implemented."

                # Now we feed the result back as a tool message
                tool_result_message = {
                    "role": "tool",
                    "content": result,
                    # tie back to the call ID so the model can handle multi-step reasoning
                    "tool_call_id": tc["id"],
                }
                messages.append(msg)  # The assistant's message that requested the tool
                messages.append(tool_result_message)

        elif finish_reason == "stop":
            # The model is providing a final answer (no function call).
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
            # The model ended after giving function calls but didn't produce final text
            # We feed back the tool calls and keep going in the loop
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
