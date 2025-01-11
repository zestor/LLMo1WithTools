import os
import json
import openai
import requests
import pytest
from datetime import datetime

################################################################################
# 1) Setup:
#    - Make sure to install openai and requests packages (e.g., pip install openai requests).
#    - Set the environment variable OPENAI_API_KEY with your OpenAI key.
#    - Tokens for other APIs should be retrieved from environment variables for security.
################################################################################

# For demonstration purposes only; in production, securely store tokens using environment variables or a vault.
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "...")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "...")
openai.api_key = os.getenv("OPENAI_API_KEY", "...")

################################################################################
# 2) Define Python functions to interface with external APIs
#
#    These functions are available for the main ChatCompletion call, allowing
#    the model to choose to invoke them to obtain data.
################################################################################

def get_current_datetime() -> str:
    """Returns the current date and time formatted for display."""
    now = datetime.now()
    formatted_time = now.strftime("%A, %B %d, %Y, %H:%M:%S")
    return f"Current date and time: {formatted_time}"

def call_research_assistant(query: str) -> str:
    """
    Uses the Perplexity AI API to perform a research query.
    Returns the text response from the model's answer.
    """
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "top_p": 0.9,
        "search_recency_filter": "month",
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"

def call_web_search_assistant(url: str, wait_for: int = 0) -> str:
    """
    Uses Firecrawl API to scrape the content from a given URL.
    Returns the content as markdown if successful.
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
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
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

def get_critical_feedback(question: str, assistant_content: str) -> str:
    """
    Provides a critical evaluation of the Q&A to identify important questions.
    Invokes a helper to parse the content and list top 10 critical questions.
    """
    critical_feedback = call_helper(f"""
    Critically evaluate the following question and answer. 
    Your task is to provide a list of the top 10 most important questions about the solution that need to be answered.
    Respond only with one question per line.
    
    ``` Question
    {question}
    ```

    ``` Answer
    {assistant_content}
    ```
    """)
    print(f"\nCritical Feedback\n\n{critical_feedback}")
    return critical_feedback

def call_research_professional(question: str, prompt: str) -> str:
    """
    Interfaces with OpenAI's "o1" model for advanced reasoning tasks.
    Executes a conversation loop, analyzing when to apply function calls.
    """
    messages = [{"role": "user", "content": get_current_datetime() + '\n' + prompt}]
    refactor_count = 1

    for _ in range(500):
        try:
            response = openai.chat.completions.create(
                model="o1",
                messages=messages,
                response_format={"type": "text"},
                reasoning_effort="high",
                tools=tools
            )

            msg = response.choices[0].message
            print(f"Response\n")
            print("=" * 80)
            print(f"\n{msg}\n\n")

            finish_reason = response.choices[0].finish_reason
            tool_calls = msg.tool_calls
            assistant_content = msg.content

            if tool_calls:
                # Processing any requested function calls
                for tc in tool_calls:
                    func_name = tc.function.name
                    arguments_json = tc.function.arguments

                    # Parsing function arguments
                    try:
                        arguments = json.loads(arguments_json)
                    except json.JSONDecodeError:
                        arguments = {}

                    # Executing relevant function
                    if func_name == "call_research_assistant":
                        query = arguments.get("query", "")
                        result = call_research_assistant(query)
                    elif func_name == "call_web_search_assistant":
                        url = arguments.get("url", "")
                        wait_for = arguments.get("wait_for", 0)
                        result = call_web_search_assistant(url, wait_for)
                    elif func_name == "call_research_professional":
                        subprompt = arguments.get("prompt", "")
                        result = call_research_professional(question, subprompt)
                    elif func_name == "call_helper":
                        subprompt = arguments.get("prompt", "")
                        result = call_helper(subprompt)
                    else:
                        result = f"Tool {func_name} is not implemented."

                    # Feed function result back as tool role
                    tool_result_message = {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.id,
                    }
                    messages.append(msg)  # Add assistant's request
                    messages.append(tool_result_message)

            elif finish_reason == "stop":
                # If refactor_count is less than 5, provide critical feedback
                if refactor_count < 5:
                    refactor_count += 1
                    messages.append({
                        "role": "developer",
                        "content": get_critical_feedback(question, assistant_content),
                    })                 
                elif refactor_count == 5:
                    refactor_count += 1
                    messages.append({
                        "role": "developer",
                        "content": "provide your final answer as if writing it for the first time",
                    }) 
                else:
                    # Final assistant answer handling
                    if assistant_content:
                        print("\nAssistant:\n" + assistant_content)
                        return assistant_content
                    else:
                        print("\nAssistant provided no content.")
                    break

            elif finish_reason == "length":
                # Handling response cut-off due to length
                print("The model's response got cut off. Stopping...")
                break

            elif finish_reason == "tool_calls":
                # Continue loop to handle generating function calls
                pass
            else:
                # Handle any unexpected finish reasons
                if assistant_content:
                    print("\nAssistant (final):\n" + assistant_content)
                else:
                    print("\nAssistant provided no content.")
                break
        except Exception as e:
            return f"Error calling OpenAI model='o1': {str(e)}"
    return ""

def call_helper(prompt: str) -> str:
    """
    Interfaces with OpenAI's "gpt-4o" model for complex reasoning tasks.
    Provides support for advanced queries and intermediate task solving.
    """
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": get_current_datetime() + '\n' + prompt}],
            temperature=0.9,
            max_completion_tokens=1600
        )

        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI model='gpt-4o': {str(e)}"

################################################################################
# 3) Define JSON schema for each function (tool).
#    These definitions are provided to ChatCompletion to enable the model to choose
#    appropriate functions to call.
################################################################################

tools = [
    {
        "type": "function",
        "function": {
            "name": "call_research_assistant",
            "description": (
                "Allows the model to request assistance from a virtual research assistant. "
                "Provide detailed information for conducting research on one topic at a time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query or question for the research assistant",
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
            "name": "call_web_search_assistant",
            "description": (
                "Uses a virtual research assistant to perform web URL research. "
                "Provide a URL to extract content and generate a markdown report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to scrape"},
                    "wait_for": {
                        "type": "number",
                        "description": "Milliseconds to wait before scraping (default 0)",
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
            "name": "call_research_professional",
            "description": (
                "Calls a professional external researcher for in-depth analysis. "
                "Provide a detailed prompt for tackling sophisticated inquiries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Search inquiry, e.g., 'best pizza in NYC'",
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
            "name": "call_helper",
            "description": (
                "Uses an assistant to support tasks that do not involve research directly, "
                "assisting with intermediate question parsing and logical tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Prompt or question for assistant support",
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
#    Processes user inquiries by utilizing OpenAI's "o1" model to evaluate the
#    necessity of function calls. Executes functions and iteratively supplies
#    feedback until receiving a conclusive answer.
#
################################################################################

def main():
    # Attempt to read user question from a file
    try:
        with open('o1wtools-input.txt', 'r') as input_file:
            user_question = input_file.read()
            print("File has been successfully read.")
    except FileNotFoundError:
        print("The file 'input.txt' does not exist.")
    except IOError:
        print("An error occurred while reading the file.")

    # Conduct research based on user's question
    final_answer = call_research_professional(user_question, user_question)

    # Write conclusion to a file
    try:
        with open('o1wtools-ouput.txt', 'w') as output_file:
            output_file.write(final_answer)
            print("Content written successfully to 'output.txt'.")
    except IOError:
        print("An error occurred while writing to the file.")

    print("\n--- End of conversation ---")

class TestFunctions:
    """
    A suite of tests interacting with real APIs.
    Ensure tokens are valid and be aware of potential usage on credits.
    """

    def test_call_research_assistant(self):
        # Perform a basic query via Perplexity
        result = call_research_assistant("What is the capital of France?")
        # Verify there are no error messages
        assert "Error " not in result, f"Failed Perplexity call: {result}"
        # Check non-emptiness for content relevance
        assert len(result.strip()) > 0, "Perplexity returned no content."

    def test_call_web_search_assistant(self):
        # Attempt to scrape a specified URL
        result = call_web_search_assistant("http://www.chrisclark.com", 0)
        assert "Error " not in result, f"Failed Firecrawl scrape: {result}"
        # Ensure returned content is non-empty
        assert len(result.strip()) > 0, "Firecrawl returned no content."

    def test_call_research_professional(self):
        # Queries the 'o1' model for a simple prompt
        prompt = "Give me a short greeting in Spanish."
        result = call_research_professional(prompt, prompt)
        assert "Error " not in result, f"Failed openai o1 call: {result}"
        # Validate response non-emptiness
        assert len(result.strip()) > 0, "Empty response from o1."

    def test_call_helper(self):
        # Queries the 'gpt-4o' model for assistance
        prompt = "Explain quantum computing in one sentence."
        result = call_helper(prompt)
        assert "Error " not in result, f"Failed openai gpt-4o call: {result}"
        # Confirm helper provides substantial output
        assert len(result.strip()) > 0, "Empty response from gpt-4o."

if __name__ == "__main__":
    main()
