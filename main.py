import os
import json
import openai
import requests
import pytest
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from firecrawl import FirecrawlApp


lock = threading.Lock()

# For illustration only; in a production system, store tokens securely (e.g., environment variables or vault).
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "...")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "...")
openai.api_key = os.getenv("OPENAI_API_KEY", "...")


def get_current_datetime() -> str:
    now = datetime.now()
    formatted_time = now.strftime("%A, %B %d, %Y, %H:%M:%S")
    return f"Current date and time:{formatted_time}"

def _call_research_assistant(query: str) -> str:

    response1 = _call_research_assistant("find official, " + query, "month")
    print("Program will wait for 2 seconds.")
    time.sleep(2)
    response2 = _call_research_assistant("retrieve official, " + query, "day")

    prompt = f"""
Write your best response to the question based on this independent research.

``` Question
{query}
```

``` Research Article
{response1}
```

``` Research Article
{response2}
```
"""
    return call_helper(prompt)

def call_research_assistant(query: str, recency: str = "month") -> str:
    """
    Calls the Perplexity AI API with the given query.
    Returns the text content from the model’s answer.
    """
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": [
            {"role": "user", "content": query},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "search_recency_filter": recency,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        retval = data["choices"][0]["message"]["content"]
        joined_citations = "\n".join(data["citations"])
        citations = f"\n\nCitations:\n{joined_citations}"
        retval = retval + citations

        # don't summarize source code
        #if not is_valid_query("Does query contain source code?", retval):
            # Chain of Density Summarization
            #retval = COD_summarization(retval)

        print(f"* * *  Research Assistant Response  * * *\n\n{retval}\n\n")
        return retval
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"


def call_web_search_assistant(url: str) -> str:
    retval = ""
    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        retval = app.scrape_url(url, params={'formats': ['markdown', 'html']})
    except Exception as e:
        retval = f"Error returning markdown data from {url}: {str(e)}"
    return retval


def get_critical_feedback(question: str, assistant_content: str) -> str:
    critical_feedback = call_helper(f"""
    Critically evaluate the following question and answer. 
    Your task is provide a list of top 3 most important questions about the solution which need to be answered.
    Respond only with one question per line.
    
    ``` Question
    {question}
    ```

    ``` Answer
    {assistant_content}
    ```
    """)
    response = f"""
Optimize your response to the objective by focusing on addressing the critical feedback. 
I would encourage your to use your tools. 
Multiple tool calls in a single request are supported.
The tools have no context regarding our prior conversation, you must provide the tool all the details including context.
    
```Objective
{question}
```
    
```Critical feedback
{critical_feedback}
```
    """
    print(f"{response}\n\n")
    return response

def is_valid_query(criteria: str, query: str) -> bool:
    retval = False
    result = call_helper(f"""
Evaluate if this query meets the criteria. 
Respond with only YES or NO.
                         
# Criteria
{criteria}

# Query
{query} 
""")
    # Convert the result to uppercase and check if it contains "YES"
    if "YES" in result.upper():
        retval = True
    #print("-" * 80)
    #print(f"Query: {query}")
    #print(f"A valid task requiring a complex LLM call = {retval}")
    #print("-" * 80)
    return retval


def call_helper(prompt: str, model: str = "gpt-4o", messages: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Calls OpenAI with model='gpt-4o' for advanced GPT-4 style reasoning or sub-queries.
    """
    helper_messages = []

    if messages is None:
        helper_messages = [
            {'role': 'user', 'content': get_current_datetime() + '\n' + prompt}
        ]
    else:
        helper_messages = messages.copy()
        # Append the user message if messages were provided
        helper_messages.append({'role': 'user', 'content': prompt})
    
    try:
        completion = openai.chat.completions.create(
            model=model,
            messages=helper_messages
        )

        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI model='{model}': {str(e)}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "call_research_assistant",
            "description": (
                "Use this to utilize a PhD grad student to perform research, "
                "they can only research one single intent question at a time, "
                "they have no context or prior knowledge of this conversation, "
                "you must give them the context and a single intention query. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A straight to the point concise succint question or search query to be sent to research assistant",
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
                "Use this to utilize a PhD grad student to perform web url research, "
                "provide them the url they will do the research "
                "and they will provide a markdown report summary of the web content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to scrape"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]

"""
    {
        "type": "function",
        "function": {
            "name": "call_research_professional",
            "description": (
                "Use this to utilize a professional 3rd party researcher, "
                "provide them the details of what to search for, "
                "they can only research one topic at a time, "
                "provide all details they have no prior knowledge or context to your query. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A search query, e.g. 'best pizza in NYC'"
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
                "This PhD student can't do research for you "
                "but can assist you with intermediate tasks, "
                "provide all details they have no prior knowledge or context to your query. "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A prompt or question to PhD student helper"
                    }
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    },
"""


def build_chain_of_density_prompt(article_text: str, prior_json: str) -> str:
    return f"""
``` Article
{article_text}
```

``` Previously generated JSON
{prior_json}
```

You will generate increasingly entity-rich denser summaries of the above Article. Repeat the following step 7 times. If there is previously generated JSON, this is a continuation.

**Step 1:** Identify up to 5 informative Entities (";" delimited) from the Article which are missing from the previously generated summary.

**Step 2:** Write a new, entity-rich denser summary which covers every entity and detail from the previous summary plus the Missing Entities.

A Missing Entity is:
- **Relevant:** to the main story.
- **Specific:** descriptive yet concise (5 words or fewer).
- **Novel:** not in the previous summary.
- **Faithful:** present in the Article.
- **Anywhere:** located anywhere in the Article.

**Guidelines:**
- Make every word count: enrich the previous summary to improve flow and make space for missing entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
- The summaries should become more informative with less words, yet self-contained, e.g., easily understood without the Article without missing any detail from the original.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
- Always mention at least once the full text for any acronyms in the summary

Answer in JSON. The JSON should be a list (length 7) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".

"""

def remove_code_blocks(text):
    # Split the text into lines
    lines = text.splitlines()
    
    # Filter out lines that start with ```
    filtered_lines = [line for line in lines if not line.lstrip().startswith('```')]
    
    # Join the lines back together
    cleaned_text = '\n'.join(filtered_lines)
    
    return cleaned_text

def COD_summarization(
    article_text: str,
    max_passes: int = 1
) -> str:
    """
    Runs multiple passes. Each pass calls GPT-4o once:
      1. The model returns 5 updated summaries.
      2. The last summary is examined for newly introduced entities.
      3. If no newly introduced entities are detected, we end.
    Returns the final JSON structure from the last iteration that adds new entities.
    """
    retval = ""
    prior_json = "[]"

    for pass_index in range(max_passes):
        prompt = build_chain_of_density_prompt(article_text, prior_json)

        #print ("**" * 40)
        #print ("\n")
        #print(prompt)

        response = call_helper(prompt, model = "gpt-4o-mini")

        # remove lines starting with ```
        response = remove_code_blocks(response)

        #print ("**" * 40)
        #print ("\n")
        #print(response)
        #print ("**" * 40)
        #print ("\n")

        try:
            chain_data = json.loads(response)
            #print(json.dumps(chain_data, indent=4, ensure_ascii=False))
        except ValueError as e:
            print(f"[Error] JSON parse failed at iteration {pass_index+1}: {e}")
            break

        # The last summary in the chain_data is the densest summary for this iteration
        final_summary_obj = chain_data[-1]
        final_summary_text = final_summary_obj.get("Denser_Summary", "")
        prior_json = f"[{json.dumps(final_summary_obj, ensure_ascii=False)}]"

        # Get last Denser_Summary from the JSON
        retval = final_summary_text

    return retval


def parse_tool_calls_from_text(assistant_content: str) -> List[Dict[str, Any]]:
    """
    Looks for any ```json ...``` blocks in the assistant content.
    Tries to parse them as JSON with "name" and "arguments".
    Returns a list of tool call dicts in the format:
       [
         {
           "id": <some_id>,
           "name": <tool_name>,
           "arguments": <dict_of_args>
         }, ...
       ]
    If none found, returns an empty list.
    """
    # Find all JSON blocks demarcated by triple backticks (```json ... ```).
    pattern = r'```json\s*(.*?)\s*```'
    blocks = re.findall(pattern, assistant_content, flags=re.DOTALL)
    tool_calls = []
    for block in blocks:
        try:
            data = json.loads(block)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                # Craft a minimal structure similar to how we handle official tool calls
                # For consistency, let's just set id to an incremental or a time stamp
                tool_calls.append({
                    "function": {
                        "name": data["name"],
                        "arguments": json.dumps(data["arguments"])
                    }
                })
        except:
            # If parsing fails, ignore that block
            pass
    return tool_calls

def call_research_professional(question: str, prompt: str, model_version: str = "o1-mini") -> str:
    """
    Calls either openai model='o1' (with official function calling) or
    openai model='o1-mini' (ReAct style, parse JSON from text).
    Minimizes code duplication by handling both flows in a single loop.
    """

    # For logging / iterative improvement
    refactor_count = 1

    # Prepare the conversation messages
    if model_version == "o1-mini":
        # For o1-mini, include a system message giving instructions on how to produce JSON tool calls
        system_message = (
            "You are a helpful AI that can use the following tools by producing JSON in your message. "
            "To call multiple tools, output multiple JSON blocks (in triple backticks, with a line ```json) in a single response."
            "When you want to call a tool, output EXACTLY a JSON block (in triple backticks, with a line ```json) "
            "of the form:\n\n"
            "{\n"
            '  "name": "<tool_name>",\n'
            '  "arguments": { "name" : "value", "name":"value", ... }\n'
            "}\n\n"
            "The valid tools are:\n"
            f"{json.dumps(tools, indent=4, ensure_ascii=False)}"
            "Return your final answer in plain text (no JSON) once you have all information you need. "
            "Do not output extraneous text outside or after a JSON block if calling a tool."
        )
        #    + "\n".join([f"- {t['function']['name']}" for t in tools])
        #    + "\n\n"
        messages = [
            {'role': 'user', 'content': system_message},
            {'role': 'user', 'content': get_current_datetime() + '\n' + prompt},
        ]
    else:
        # For o1, we simply start with a user message
        messages = [
            {"role": 'user', 'content': get_current_datetime() + '\n' + prompt},
        ]

    # Main loop for back-and-forth with the model
    for _ in range(500):
        # For debugging/logging
        print("~" * 80)
        print("\nMessage Stack Before:\n")
        try:
            print(json.dumps(messages, indent=4))
        except:
            try:
                json_obj = json.loads(messages)
                print(json.dumps(json_obj, indent=4))
            except:
                print(str(messages).replace("{'role'", "\n\n\n{'role'"))
        print("\n" + "~" * 80 + "\n")

        # change the model to o1 for the last summary 
        #if refactor_count == 3 and model_version == 'o1-mini':
        #    model_version = "o1"
        #    messages = messages[1:]

        # Call the appropriate model
        base_args = {
            "messages": messages,
            "response_format": {"type": "text"},
        }

        # Define model-specific settings
        if model_version == "o1":
            model_args = {
                "model": "o1",
                "tools": tools,
                "reasoning_effort": "high",
                "max_completion_tokens": 100000
            }
        elif model_version == "o1-mini":
            model_args = {
                "model": "o1-mini",
                "max_completion_tokens": 65536
            }
        else:
            raise ValueError(f"Unsupported model version: {model_version}")

        # Merge common and model-specific settings
        args = {**base_args, **model_args}

        # Call OpenAI API with merged arguments
        response = openai.chat.completions.create(**args)

        print(f"Message Received")
        try:
            print(json.dumps(response, indent=4))
        except:
            try:
                response_json_obj = json.loads(response)
                print(json.dumps(response_json_obj, indent=4))
            except:
                print(str(response).replace("{'role'", "\n\n\n{'role'"))

        msg = response.choices[0].message

        try:
            print(json.dumps(msg, indent=4))
        except:
            try:
                response_json_obj = json.loads(msg)
                print(json.dumps(response_json_obj, indent=4))
            except:
                print(str(msg).replace("{'role'", "\n\n\n{'role'"))

        assistant_content = msg.content
        finish_reason = response.choices[0].finish_reason

        # Determine tool calls based on model version
        if model_version == "o1":
            tool_calls = getattr(msg, "tool_calls", None)
        elif model_version == "o1-mini":
            tool_calls = parse_tool_calls_from_text(assistant_content)

        # Append the assistant's text to the conversation
        if model_version == "o1":
            messages.append(msg)
        elif model_version == "o1-mini":
            messages.append({'role':'assistant', 'content':assistant_content})

        # For debugging/logging
        print("~" * 80)
        print("\nMessage Stack After:\n")
        try:
            print(json.dumps(messages, indent=4))
        except:
            try:
                json_obj = json.loads(messages)
                print(json.dumps(json_obj, indent=4))
            except:
                print(str(messages).replace("{'role'", "\n\n\n{'role'"))
        print("\n" + "~" * 80 + "\n")
        
        # Log to a file
        try:
            with open('o1wtools-intermediate.txt', 'a') as output_file:
                if finish_reason is not None:
                    output_file.write(f"{finish_reason}\n")
                if assistant_content is not None:
                    output_file.write(f"{assistant_content}\n")
                if tool_calls:
                    output_file.write(f"{tool_calls}\n")
                output_file.write("=" * 80 + "\n" + "=" * 80 + "\n")
        except IOError:
            print("An error occurred while writing to the file.")

        # If there are tool calls, handle them
        if tool_calls:
            # “o1” tool_calls is a list of tool call objects with .function.name, .function.arguments
            # “o1-mini” parse_tool_calls_from_text returns a list of dicts with the same structure:
            #    { "id": <some_id>, "function": { "name": <tool_name>, "arguments": <json_string> } }
            for tc in tool_calls:
                # In “o1” this might be tc.function.name/arguments
                # In “o1-mini”, we structured the parse results similarly
                func_name = tc["function"]["name"] if isinstance(tc, dict) else tc.function.name
                arguments_json = tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments

                print("^" * 80)
                print("\nTool Request")
                print(f"Tool name: {func_name}\nArguments: {arguments_json}\n")
                print("^" * 80 + "\n")

                # Attempt to parse arguments JSON
                try:
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError:
                    arguments = {}

                # Dispatch to the correct tool
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

                # Feed the result back as a "tool" role message (for o1),
                # or just an extra message (for o1-mini). Either way, the model sees it next iteration.
                tool_role = "user" if model_version == 'o1-mini' else "tool"

                tool_result_message = {'role': tool_role, 'content': result}

                if model_version == "o1-mini":
                    tool_result_message["tool_response"] = func_name

                # “o1” includes a "tool_call_id"; optional for “o1-mini.” We can unify it:
                if isinstance(tc, dict) and "id" in tc:
                    tool_result_message["tool_call_id"] = tc["id"]
                else:
                    # For official function calls: tool_call_id = tc.id
                    possible_id = getattr(tc, "id", None)
                    if possible_id:
                        tool_result_message["tool_call_id"] = possible_id

                messages.append(tool_result_message)

                # Log the tool result
                try:
                    with open('o1wtools-intermediate.txt', 'a') as output_file:
                        output_file.write(f"{result}\n")
                        output_file.write("=" * 80 + "\n")
                except IOError:
                    print("An error occurred while writing to the file.")

            # After tool calls, continue loop so the model sees the new tool outputs
            continue

        # If no tool calls, check finish_reason
        if finish_reason == "stop":

            if refactor_count <= 1:

                refactor_count = refactor_count + 1

                feedback_questions = get_critical_feedback(question, assistant_content)
                messages.append({'role': 'user', 'content': feedback_questions})

                continue

            elif refactor_count == 2:

                refactor_count = refactor_count + 1

                feedback = "provide your final answer as if writing it for the first time"
                messages.append({'role': 'user','content': feedback,})
                # Writing to a file
                try:
                    with open('o1wtools-intermediate.txt', 'a') as output_file:
                        output_file.write(f"{feedback}\n")
                        output_file.write("=" * 80)
                        output_file.write("\n")
                except IOError:
                    print("An error occurred while writing to the file.") 

                continue 

            else:
                # The model gave a final answer
                if assistant_content:
                    print("\nAssistant:\n" + assistant_content)
                    return assistant_content
                else:
                    print("\nAssistant provided no content.")
                break

        elif finish_reason in ["length", "max_tokens", "content_filter"]:
            # The conversation got cut off or other forced stop
            print("The model's response ended due to finish_reason =", finish_reason)
            break

        # If we get here with no tool calls and not “stop,”
        # we can guess the model simply produced final text or there's no more to do
        if assistant_content.strip():
            print("\nAssistant:\n" + assistant_content)
            return assistant_content

    return "Lacked sufficient details to complete request."


def main():
    # Reading from a file
    with open('o1wtools-input.txt', 'r') as input_file:
        user_question = input_file.read()

    with open('o1wtools-intermediate.txt', 'w') as output_file:
        output_file.write("")
        
    # Modify model_version here if you want to switch (e.g. "o1-mini")
    final_answer = call_research_professional(user_question, user_question)

    # Writing to a file
    with open('o1wtools-ouput.txt', 'w') as output_file:
        output_file.write(final_answer)

    print("\n--- End of conversation ---")


class TestFunctions:
    """
    A class with tests that actually call the real APIs.
    Make sure your tokens are valid and be aware this might consume credits/tokens.
    """

    def test_call_research_assistant(self):
        # Simple query to Perplexity
        result = call_research_assistant("What is the capital of France?")
        # Check that we didn't get an immediate error
        assert "Error " not in result, f"Failed Perplexity call: {result}"
        # Expect to see "Paris" or non-empty text
        assert len(result.strip()) > 0, "Perplexity return is empty."

    def test_call_web_search_assistant(self):
        # Attempt to scrape example.com (simple site)
        result = call_web_search_assistant("http://www.chrisclark.com", 0)
        assert "Error " not in result, f"Failed Firecrawl scrape: {result}"
        # We expect some text content or markdown
        assert len(result.strip()) > 0, "Firecrawl returned empty."

    def test_call_research_professional(self):
        # By default tries 'o1' model
        prompt = "Give me a short greeting in Spanish."
        result = call_research_professional(prompt, prompt, model_version="o1")
        assert "Error " not in result, f"Failed openai o1 call: {result}"
        assert len(result.strip()) > 0, "Empty response from o1."

    def test_call_research_professional_o1_mini(self):
        # Test the new ReAct approach with "o1-mini"
        prompt = "Give me a short greeting in Spanish."
        result = call_research_professional(prompt, prompt, model_version="o1-mini")
        # We can't guarantee the LLM's output, but at least ensure no immediate error:
        assert "Error " not in result, f"Error result: {result}"
        assert len(result.strip()) > 0, "Empty response from o1-mini."

    def test_call_helper(self):
        # Queries the 'gpt-4o' model
        prompt = "Explain quantum computing in one sentence."
        result = call_helper(prompt)
        assert "Error " not in result, f"Failed openai gpt-4o call: {result}"
        assert len(result.strip()) > 0, "Empty response from gpt-4o."

if __name__ == "__main__":
    main()
