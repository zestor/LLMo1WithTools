import os
import json
import openai
import requests
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from firecrawl import FirecrawlApp
import unicodedata
from o1reasoning_calculator import Calculator


lock = threading.Lock()

# For illustration only; in a production system, store tokens securely (e.g., environment variables or vault).
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "...")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "...")
openai.api_key = os.getenv("OPENAI_API_KEY", "...")


scores = []


def add_score(score):
    # Append the new score to the global list
    scores.append(score)
    # Print all scores in order
    print_scores()

def print_scores():
    # Print scores separated by commas
    print("Scores:", ", ".join(map(str, scores)))
    # Log the tool result
    try:
        with open('o1wtools-intermediate.txt', 'a') as output_file:
            output_file.write("^" * 80)
            output_file.write("\nScores:")
            output_file.write(", ".join(map(str, scores)))
            output_file.write("^" * 80 + "\n")                        

    except IOError:
        print("An error occurred while writing to the file.")  

def convert_invalid_json_to_valid(input_str):
    # Remove markdown code block delimiters using regex
    input_str = re.sub(r'```json\s*', '', input_str, flags=re.IGNORECASE)
    input_str = re.sub(r'```\s*', '', input_str)

    # Trim any remaining leading/trailing whitespace
    input_str = input_str.strip()
    
    # Fix issues with missing braces and colons
    try:
        # Repair structure: ensure that "Critical_Evaluation" is enclosed properly
        # Check if the input doesn't already start and end with appropriate braces
        if not input_str.startswith('{'):
            input_str = '{' + input_str
        
        if not input_str.endswith('}'):
            input_str = input_str + '}'
        
        # Correct the structure by replacing misplaced or missing colons/commas
        input_str = re.sub(r'"Critical_Evaluation":\s*', '"Critical_Evaluation": {', input_str, count=1)

        input_str = input_str + '}'

        # Debug: print statements for checking the sanitized string form
        # print("Corrected JSON String:", input_str)

        #print(f"this is the reformatted json\n\n{input_str}\n\n")
        # Load the JSON data
        data = json.loads(input_str)

        return json.dumps(data, indent=4)

    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}"
    
def parse_rating_response(response_data, threshold: float):

    print(f"parse_rating_response\n\n{response_data}\n\n")
    try:
        json_data = ""
        if not '\n' in response_data:
            json_data = convert_invalid_json_to_valid(response_data)   
        else:
            lines = response_data.splitlines()
            json_data = "\n".join(line for line in lines if not line.strip().startswith('```'))

        #print(f"Loading this json data\n\n{json_data}\n\n")

        data = json.loads(json_data)
        if 'Critical_Evaluation' in data:
            evaluation = data['Critical_Evaluation']
            if all(key in evaluation for key in ['Pros', 'Cons', 'Rating']):
                try:
                    # Attempt to convert the rating to a float
                    rating = float(evaluation['Rating'])
                    add_score(rating)
                    return rating >= threshold
                except (ValueError, TypeError) as e:
                    print("FAILED parse_rating_response: ", e)
                    pass
    except json.JSONDecodeError:
        print("FAILED json.JSONDecodeError parse_rating_response")
        pass
    return False

def get_current_datetime() -> str:
    now = datetime.now()
    formatted_time = now.strftime("%A, %B %d, %Y, %H:%M:%S")
    return f"Current date and time:{formatted_time}"


def call_web_search_assistant(query: str, recency: str = "month") -> str:
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

        print(f"* * *  Research Assistant Response  * * *\n\n{retval}\n\n")
        return retval
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}"


def call_web_content_retriever(url: str) -> str:
    retval = ""
    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        retval = app.scrape_url(url, params={'formats': ['markdown']}, timeout=180000)
        #firecrawl_json_obj = json.loads(json)
        #retval = firecrawl_json_obj.data.markdown
    except Exception as e:
        retval = f"Error returning markdown data from {url}: {str(e)}"
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
            "name": "call_web_search_assistant",
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

]

"""

    {
        "type": "function",
        "function": {
            "name": "call_web_content_retriever",
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

    is_getting_critical_feedback = False
    is_final_answer = False
    
    messages = []

    # Prepare the conversation messages
    if model_version == "o1-mini":
        # For o1-mini, include a system message giving instructions on how to produce JSON tool calls
        system_message = (
            "I've upgraded your functionality to access post-Oct 2023 data and use tools. You can now request tool "
            "executions using JSON in your responses, and I'll run them and return the results in user messages. "
            "You are able to call tools just by telling me you want to run a tool by responding with JSON as described below. "
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
    for _ in range(100):
        # For debugging/logging
        #print("~" * 80)
        #print("\nMessage Stack Before:\n")
        #try:
        #    print(json.dumps(messages, indent=4))
        #except:
        #    try:
        #        json_obj = json.loads(messages)
        #        print(json.dumps(json_obj, indent=4))
        #    except:
        #        print(str(messages).replace("{'role'", "\n\n\n{'role'"))
        #print("\n" + "~" * 80 + "\n")

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

        #print(f"Message Received")
        #try:
        #    print(json.dumps(response, indent=4))
        #except:
        #    try:
        #        response_json_obj = json.loads(response)
        #        print(json.dumps(response_json_obj, indent=4))
        #    except:
        #        print(str(response).replace("{'role'", "\n\n\n{'role'"))

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

        if response.usage.prompt_tokens > 60000 and not is_final_answer:
            is_final_answer = True
            print("*" * 80)
            print("*" * 80)
            print("ABORTING... SHORTCUT TO FINAL ANSWER DUE TO CONTEXT LENGTH")
            print("*" * 80)
            print("*" * 80)
            messages.append({'role': 'user', 'content': 'Write your long long long final analysis to the user\'s question without missing any detail.'})
            continue

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
        

        print("*" * 80)
        print(f"USAGE... Prompt {response.usage.prompt_tokens} "
              f"Completion {response.usage.completion_tokens} "
              f"Total {response.usage.total_tokens}")
        print(f"USAGE... Reasoning {response.usage.completion_tokens_details.reasoning_tokens} "
              f"Accepted Prediction {response.usage.completion_tokens_details.accepted_prediction_tokens} "
              f"Rejected Prediction {response.usage.completion_tokens_details.rejected_prediction_tokens} ")
        print("*" * 80)

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
                if func_name == "call_web_search_assistant":
                    query = arguments.get("query", "")
                    result = call_web_search_assistant(query)

                elif func_name == "call_web_content_retriever":
                    url = arguments.get("url", "")
                    result = call_web_content_retriever(url)

                elif func_name == "call_research_professional":
                    subprompt = arguments.get("prompt", "")
                    result = call_research_professional(subprompt, subprompt)

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

            if assistant_content is not None:
                if is_final_answer:
                    print("\nAssistant:\n" + assistant_content)
                    return assistant_content             
                                        
                if is_getting_critical_feedback:
                    is_getting_critical_feedback = False

                    scores_text = "\nScores:" + ", ".join(map(str, scores))

                    promptx = f"""
You are a highly successful people manager with all company resources 
at your disposal. Your employee is performing the following task and 
has received the following scores and feedback. Response must include 
your best motivational speech to the employee to substantially increase 
their score on the task. Provide incentives for good performance and 
discourage poor performance through constructive feedback or consequences. 
Respond as if you are talking to them directly without mentioning their name.

``` task
{question}
```

``` Iterative scores in order based on the initial draft and the latest version
{scores_text}
```

``` feedback
{assistant_content}
```
"""
                    manager_feedback = call_helper(promptx)

                    promptx = f"""
Your response has not passed the company quality metric.
Gather more information and revise your response.

Manager feedback:
{manager_feedback}
"""
                    messages.append({'role': 'user', 'content': promptx})
                    continue


                is_pass_threshold = parse_rating_response(assistant_content, 0.99)
                print(f"\n\n\nPASSED THRESHOLD {0.99} {is_pass_threshold}\n\n\n")

                if is_pass_threshold:   
                    is_final_answer = True
                    messages.append({'role': 'user', 'content': f'Write your long long long final analysis to the user\'s question without missing any detail.\n\nUser\'s Question\n\n{question}'})
                    continue
                else:
                    is_getting_critical_feedback = True
                    promptx = f"""
Critically evaluate your response against the user's question  
and provide a list of both pros / cons statements and rating 
between 0.0 and 1.0. With 1.0 being the highest score.

```User's Question
{question}
```

```Rating Guidance

#### 0.0 - Completely Unacceptable
- **Clarity:** The response is entirely unclear or nonsensical.
  - *Example:* "Fwdoa+kdjjs! None needed."
- **Relevance:** Does not relate to the user’s request in any way.
- **Completeness:** No attempt to address the request.
- **Accuracy:** Completely inaccurate or misleading.
- **User Engagement:** Off-putting or entirely confusing, leading to frustration.

#### 0.1 - Severely Lacking
- **Clarity:** Poorly structured and confusing.
  - *Example:* "Maybe this helps answer?"
- **Relevance:** Barely touches on the topic; mostly irrelevant.
- **Completeness:** Contains almost no useful information.
- **Accuracy:** Mostly incorrect; little correct information.
- **User Engagement:** Difficult to stay engaged due to frustration or confusion.

#### 0.2 - Very Poor
- **Clarity:** Somewhat understandable but mostly unclear.
  - *Example:* "Could mean it answers part of your question."
- **Relevance:** Mostly irrelevant material with minor relevant points.
- **Completeness:** Largely incomplete; missing critical details.
- **Accuracy:** Predominantly inaccurate information.
- **User Engagement:** Likely disengages user quickly.

#### 0.3 - Poor
- **Clarity:** Significant clarity issues; requires multiple readings.
  - *Example:* "Mildly responds to what you need."
- **Relevance:** Disconnected from the main topic; few applicable details.
- **Completeness:** Major areas unaddressed.
- **Accuracy:** Misinformation present with some accurate points.
- **User Engagement:** Causes frustration; minimal user interest maintained.

#### 0.4 - Below Average
- **Clarity:** Parts of the response are clear, others confusing.
  - *Example:* "The topic relates slightly to your question."
- **Relevance:** Some relevance; a lot of unrelated content.
- **Completeness:** Leaves out several important points.
- **Accuracy:** Contains factual errors but some correct information.
- **User Engagement:** Engages intermittently; often loses reader's attention.

#### 0.5 - Average
- **Clarity:** Understandable but lacks fluent transitions.
  - *Example:* "Discusses part of your request adequately."
- **Relevance:** Mix of relevant and irrelevant information.
- **Completeness:** Covers the fundamental points; lacks depth.
- **Accuracy:** A mix of accurate and inaccurate elements.
- **User Engagement:** User may scan rather than read deeply.

#### 0.6 - Satisfactory
- **Clarity:** Mostly clear; some awkward phrasing.
  - *Example:* "Adequate explanation touching on your query."
- **Relevance:** Mostly relevant with minor deviations.
- **Completeness:** Expanded but not exhaustive coverage.
- **Accuracy:** Generally correct, with minor errors.
- **User Engagement:** Holds user interest reasonably well.

#### 0.7 - Good
- **Clarity:** Generally clear; minor occasional ambiguity.
  - *Example:* "Provides good insight into your request requirements."
- **Relevance:** Stays on topic; relevant to the user’s question.
- **Completeness:** Covers most aspects; may miss finer details.
- **Accuracy:** Accurate overall with negligible mistakes.
- **User Engagement:** Effectively maintains user interest.

#### 0.8 - Very Good
- **Clarity:** Clear and easy to follow.
  - *Example:* "Addresses your request thoroughly and understandably."
- **Relevance:** Highly relevant throughout.
- **Completeness:** Comprehensive coverage with minor omissions.
- **Accuracy:** Accurate and dependable information.
- **User Engagement:** Encourages ongoing engagement and interest.

#### 0.9 - Excellent
- **Clarity:** Exceptionally clear and well-organized.
  - *Example:* "Extremely well covered and detailed response."
- **Relevance:** Stays completely on topic; very applicable.
- **Completeness:** Extensive and near exhaustive detail.
- **Accuracy:** Error-free and precise.
- **User Engagement:** Highly engaging and prompts further exploration.

#### 1.0 - Outstanding
- **Clarity:** Crystal clear with exemplary flow.
  - *Example:* "Perfect response; precisely addresses and solves your query."
- **Relevance:** Perfectly aligned with the question; completely relevant.
- **Completeness:** Exhaustive in depth and scope.
- **Accuracy:** 100% accurate with impeccable reliability.
- **User Engagement:** Maximizes engagement; encourages active interaction.
```
"""
                    promptx = promptx + """
Respond only in JSON following the example template below.

```json
{
    "Critical_Evaluation": {
        "Pros": [
        ],
        "Cons": [
        ],
        "Rating": 0.0
    }
}
```
    """
                    messages.append({'role': 'user', 'content': promptx})
                    continue
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


if __name__ == "__main__":
    main()
