import os
import json
from openai import OpenAI
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
USE_DEEPSEEK = True

if USE_DEEPSEEK:
    client = OpenAI(base_url="https://api.deepseek.com")
    client.api_key = os.getenv("DEEPSEEK_API_KEY", "...")
else:
    client = OpenAI(base_url="https://api.openai.com")
    client.api_key = os.getenv("OPENAI_API_KEY", "...")


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
    Calls LLM for advanced reasoning or sub-queries.
    """
    helper_messages = []

    if USE_DEEPSEEK:
        model = "deepseek-reasoner"

    if messages is None:
        helper_messages = [
            {'role': 'user', 'content': get_current_datetime() + '\n' + prompt}
        ]
    else:
        helper_messages = messages.copy()
        # Append the user message if messages were provided
        helper_messages.append({'role': 'user', 'content': prompt})
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=helper_messages
        )

        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM model='{model}': {str(e)}"


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

# for DeepSeek they don't support multiple messages
def get_compressed_messages(messages) -> Dict[str,str]:
    formatted_output = ""
    for message in messages:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        formatted_output += f"\n=====\n[{role.upper()}]:\n=====\n{content}\n\n"
    return [{"role":"user", "content":formatted_output}]

def call_research_professional(question: str, prompt: str, model_version: str = "o1-mini") -> str:
    """
    Calls reasoning LLM o1, o1-mini, deepseek-reasoner
    """

    if USE_DEEPSEEK:
        model_version = "deepseek-reasoner"

    is_getting_critical_feedback = False
    is_final_answer = False
    
    messages = []

    # Prepare the conversation messages
    if model_version == "o1-mini" or model_version == "deepseek-reasoner":
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

        if USE_DEEPSEEK:
            base_args = {
                "messages": get_compressed_messages(messages),
                "response_format": {"type": "text"},
            }
        else:
            base_args = {
                "messages": messages,
                "response_format": {"type": "text"},
            }

        # Define model-specific settings
        if model_version == "o1":
            MAX_PROMPT_TOKENS = 60000
            model_args = {
                "model": model_version,
                "tools": tools,
                "reasoning_effort": "high",
                "max_completion_tokens": 100000
            }
        elif model_version == "o1-mini":
            MAX_PROMPT_TOKENS = 60000
            model_args = {
                "model": model_version,
                "max_completion_tokens": 65536
            }
        elif model_version == "deepseek-reasoner":
            MAX_PROMPT_TOKENS = 32000
            model_args = {
                "model": model_version,
                "max_completion_tokens": 8096,
                "stream": False
            }
        else:
            raise ValueError(f"Unsupported model version: {model_version}")

        # Merge common and model-specific settings
        args = {**base_args, **model_args}

        # Call LLM API with merged arguments
        response = client.chat.completions.create(**args)

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

        if response.usage.prompt_tokens > MAX_PROMPT_TOKENS and not is_final_answer:
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
            messages.append(msg)
        else:
            tool_calls = parse_tool_calls_from_text(assistant_content)
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

                if not model_version == "o1":
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

Feedback from your Manager:
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

#### 0.825 - Very Good Plus
- **Clarity:** Exceptionally clear with seamless flow.
- **Relevance:** Maintains high relevance with slight enhancements.
- **Completeness:** Nearly comprehensive; addresses almost all aspects.
- **Accuracy:** Highly accurate with minimal errors.
- **User Engagement:** Very engaging, sustaining interest effortlessly.

#### 0.85 - Excellent
- **Clarity:** Exceptionally clear and well-organized.
  - *Example:* "Extremely well covered and detailed response."
- **Relevance:** Stays completely on topic; very applicable.
- **Completeness:** Extensive and thorough detail, covering all key points.
- **Accuracy:** Error-free and precise information.
- **User Engagement:** Highly engaging and prompts further exploration.

#### 0.875 - Excellent Plus
- **Clarity:** Impeccably clear with flawless structure.
- **Relevance:** Perfectly aligned with the user’s intent.
- **Completeness:** Exhaustive coverage with insightful additions.
- **Accuracy:** Perfectly accurate with no discernible errors.
- **User Engagement:** Maximizes engagement; highly compelling and interactive.

#### 0.9 - Outstanding
- **Clarity:** Crystal clear with exemplary flow and readability.
  - *Example:* "Perfect response; precisely addresses and solves your query with exceptional clarity."
- **Relevance:** Perfectly aligned with the question; completely relevant in all aspects.
- **Completeness:** Exhaustive in depth and scope, leaving no aspect unaddressed.
- **Accuracy:** 100% accurate with impeccable reliability; all information is correct and verifiable.
- **User Engagement:** Maximizes engagement; encourages active interaction and sustained interest.
- **Additional Criteria:**
  - **Structure:** Logically organized with a coherent progression of ideas.
  - **Style:** Professional and appropriate tone tailored to the user's needs.
  - **Insightfulness:** Provides valuable insights or perspectives that enhance understanding.

#### 0.925 - Outstanding Plus
- **Clarity:** Flawless clarity with masterful organization and presentation.
- **Relevance:** Seamlessly integrates all aspects of the user's question with precise alignment to their intent.
- **Completeness:** Comprehensive and insightful, leaving no stone unturned and covering all possible dimensions.
- **Accuracy:** Impeccable accuracy with authoritative and reliable information supported by credible sources.
- **User Engagement:** Exceptionally engaging; fosters deep user interaction and maintains high levels of interest throughout.
- **Additional Criteria:**
  - **Depth of Analysis:** Demonstrates thorough analysis and critical thinking, providing nuanced explanations.
  - **Creativity:** Incorporates creative elements or unique approaches that add value to the response.
  - **Responsiveness:** Anticipates and addresses potential follow-up questions or related concerns effectively.

#### 0.95 - Superior
- **Clarity:** Perfectly articulated with exceptional readability and precision.
- **Relevance:** Utterly relevant, addressing every facet of the user's inquiry with exactitude.
- **Completeness:** Complete and thorough beyond expectations, covering all key and ancillary points comprehensively.
- **Accuracy:** Absolute accuracy with definitive authority; all statements are verifiable and error-free.
- **User Engagement:** Highly captivating; inspires user action, fosters deeper exploration, and maintains sustained interest.
- **Additional Criteria:**
  - **Depth of Content:** Provides in-depth coverage with rich, detailed information that enhances user understanding.
  - **Analytical Rigor:** Exhibits strong analytical skills, offering critical evaluations and well-supported conclusions.
  - **Adaptability:** Tailors responses dynamically to align with the user's knowledge level and specific needs.
  - **Resourcefulness:** Effectively incorporates relevant examples, analogies, or references that facilitate comprehension.

#### 0.96 - Superior Plus
- **Clarity:** Impeccable clarity with an elegant narrative structure that facilitates effortless understanding.
- **Relevance:** Intricately tailored to the user's needs with insightful relevance, ensuring every aspect directly addresses the inquiry.
- **Completeness:** Unmatched thoroughness, encompassing all possible angles and providing exhaustive information without redundancy.
- **Accuracy:** Flawlessly accurate with authoritative depth, presenting information that is not only correct but also enriched with expert knowledge.
- **User Engagement:** Exceptionally engaging; profoundly impactful and memorable, fostering a strong connection with the user.
- **Additional Criteria:**
  - **Innovative Thinking:** Introduces innovative concepts or approaches that offer fresh perspectives.
  - **Comprehensive Integration:** Skillfully integrates multiple relevant topics or ideas seamlessly.
  - **Exceptional Support:** Provides robust evidence, detailed examples, and comprehensive explanations that substantiate all claims.
  - **User-Centric Approach:** Demonstrates a deep understanding of the user's context and adapts the response to maximize relevance and utility.

#### 0.97 - Exemplary
- **Clarity:** Unmatched clarity with a sophisticated and nuanced presentation that ensures complete understanding.
- **Relevance:** Deeply resonates with the user's intent, enhancing their comprehension and addressing implicit needs.
- **Completeness:** Comprehensive beyond standard expectations, providing added value through extensive coverage and supplementary information.
- **Accuracy:** Perfectly accurate with insightful analysis, offering precise and well-supported information.
- **User Engagement:** Highly engaging; creates a meaningful and lasting impression, encouraging continuous interaction and exploration.
- **Additional Criteria:**
  - **Advanced Insight:** Delivers profound insights that significantly enhance the user's perspective.
  - **Holistic Approach:** Considers and integrates various relevant factors, providing a well-rounded and multifaceted response.
  - **Expert Tone:** Maintains an authoritative yet approachable tone that instills confidence and trust.
  - **Proactive Assistance:** Anticipates further user needs and proactively addresses potential questions or areas of interest.

#### 0.98 - Masterful
- **Clarity:** Flawlessly clear with masterful articulation that conveys complex ideas with ease.
- **Relevance:** Perfectly aligned and anticipates user needs seamlessly, ensuring every element of the response serves a purpose.
- **Completeness:** Exhaustive and insightful, offering profound depth and breadth that thoroughly satisfies the user's inquiry.
- **Accuracy:** Impeccably accurate with authoritative and reliable information, presenting data and facts with impeccable precision.
- **User Engagement:** Exceptionally engaging; inspires trust and admiration, maintaining user interest through compelling content and presentation.
- **Additional Criteria:**
  - **Strategic Depth:** Demonstrates strategic thinking by connecting concepts and providing actionable recommendations.
  - **Comprehensive Detailing:** Includes comprehensive details that leave no aspect unexplored, enhancing the richness of the response.
  - **Polished Presentation:** Exhibits a polished and professional presentation that reflects a high level of expertise and dedication.
  - **Empathetic Understanding:** Shows a deep empathetic understanding of the user's situation, tailoring the response to resonate personally.

#### 0.99 - Near Perfect
- **Clarity:** Crystal clear with impeccable expression, ensuring absolute understanding without ambiguity.
- **Relevance:** Precisely tailored to the user's question, leaving no room for ambiguity or misinterpretation.
- **Completeness:** Virtually exhaustive, covering every conceivable aspect with finesse and thoroughness.
- **Accuracy:** Absolute precision with no errors; authoritative and reliable, providing information that is both correct and insightful.
- **User Engagement:** Maximizes engagement; deeply resonates and encourages further exploration and interaction.
- **Additional Criteria:**
  - **Exemplary Insight:** Offers exceptional insights that provide significant added value and deepen user understanding.
  - **Seamless Integration:** Effortlessly integrates diverse elements into a cohesive and harmonious response.
  - **Innovative Excellence:** Showcases innovative excellence by introducing groundbreaking ideas or methodologies.
  - **Ultimate User Alignment:** Aligns perfectly with the user's goals and expectations, delivering a response that feels personalized and highly relevant.

#### 1.0 - Outstanding
- **Clarity:** Crystal clear with exemplary flow and precision, ensuring the response is effortlessly understandable.
  - *Example:* "Perfect response; precisely addresses and solves your query with exceptional clarity and coherence."
- **Relevance:** Perfectly aligned with the question; completely relevant in all aspects and anticipates implicit user needs.
- **Completeness:** Exhaustive in depth and scope, leaving no aspect unaddressed and providing comprehensive coverage.
- **Accuracy:** 100% accurate with impeccable reliability; all information is correct, verifiable, and articulated with authority.
- **User Engagement:** Maximizes engagement; encourages active interaction, sustained interest, and fosters a meaningful connection with the user.
- **Additional Criteria:**
  - **Mastery of Subject:** Demonstrates unparalleled expertise and mastery of the subject matter, providing authoritative and insightful content.
  - **Exceptional Innovation:** Introduces highly innovative concepts or solutions that significantly enhance the response's value.
  - **Flawless Structure:** Exhibits a flawless and logical structure that enhances the readability and effectiveness of the response.
  - **Inspirational Quality:** Possesses an inspirational quality that motivates and empowers the user, leaving a lasting positive impression.
  - **Comprehensive Support:** Provides extensive supporting evidence, detailed examples, and thorough explanations that reinforce all assertions.
  - **Adaptive Responsiveness:** Adapts dynamically to any nuances in the user's question, ensuring the response is precisely tailored and highly effective.
  - **Holistic Integration:** Seamlessly integrates multiple perspectives and dimensions, offering a well-rounded and multifaceted answer.
  - **Empathetic Connection:** Establishes a deep empathetic connection, demonstrating a profound understanding of the user's context and needs.
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
