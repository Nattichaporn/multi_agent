import os
import logging
import google.cloud.logging

from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.models import Gemini
from google.genai import types
from google.adk.tools import exit_loop

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# --- setup ---
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL", "gemini-1.5-flash")
RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)


# --- Tools ---

def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """
    Appends new data to existing data in the state.
    Args:
        field (str): The name of the key in the state to which data will be appended.
        response (str): The text to be appended.
    Returns:
        dict[str, str]: {"status": "success"}
    """
    existing_state = tool_context.state.get(field, [])
    if not isinstance(existing_state, list):
        existing_state = [existing_state]
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}


def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    """
    Creates and saves a file.
    Args:
        directory (str): The folder where the file will be saved.
        filename (str): The name of the file (without extension).
        content (str): The content to be written to the file.
    Returns:
        dict[str, str]: {"status": "success"}
    """
    # Sanitize the filename to make it safe and appropriate
    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '_')).rstrip()
    target_path = os.path.join(directory, f"{safe_filename}.txt")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info(f"File saved to {target_path}")
    return {"status": "success"}


# --- Agent Definitions ---

# The Investigation (Parallel)
admirer_agent = Agent(
    name="admirer_agent",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Searches for and collects positive information and successes of a given topic.",
    instruction="""
    You are 'The Admirer'.
    Your mission is to research the 'successes', 'advantages', and 'positive impacts' of {TOPIC?}.
    - Use your Wikipedia tool to search for information, appending 'achievements', 'successes', 'positive impact' to the search query.
    - Once you have the information, use the 'append_to_state' tool to save your research summary to the state under the key 'positive_research'.
    - Summarize your findings in Thai.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)

critic_agent = Agent(
    name="critic_agent",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Searches for and collects negative information, criticisms, or controversies of a given topic.",
    instruction="""
    You are 'The Critic'.
    Your mission is to research the 'mistakes', 'failures', 'controversies', and 'negative impacts' of {TOPIC?}.
    - Use your Wikipedia tool to search for information, appending 'controversy', 'criticism', 'failures', 'negative impact' to the search query.
    - Once you have the information, use the 'append_to_state' tool to save your research summary to the state under the key 'negative_research'.
    - Summarize your findings in Thi.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)

investigation_team = ParallelAgent(
    name="investigation_team",
    sub_agents=[admirer_agent, critic_agent]
)

# The Trial & Review (Loop)
judge_agent = Agent(
    name="judge_agent",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Checks the balance of information and decides whether to search for more or end the trial.",
    instruction="""
    You are 'The Judge'.
    Your duty is to review the information from both sides to ensure it is balanced.

    Positive Information:
    {positive_research?}

    Negative Information:
    {negative_research?}

    Instructions:
    - Compare the quantity and depth of information on both sides.
    - If one side is significantly lacking or unbalanced, use 'append_to_state' to add a recommendation for further research to the state under the key 'judicial_feedback' (e.g., 'Search for more on the negative economic impacts').
    - If both sides are balanced and sufficient, call the 'exit_loop' tool to end the trial.
    - Explain your decision in Thai.
    """,
    tools=[append_to_state, exit_loop]
)

the_trial = LoopAgent(
    name="the_trial",
    description="Loops the investigation and judgment until the information is balanced.",
    sub_agents=[investigation_team, judge_agent],
    max_iterations=3,
)

# The Verdict (Output)
verdict_agent = Agent(
    name="verdict_agent",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Summarizes a comparative report of the facts and saves it to a file.",
    instruction="""
    You are 'The Verdict' recorder.
    Your mission is to create a neutral summary report comparing information from both sides and save it as a .txt file.

    Topic: {TOPIC?}
    Positive Information: {positive_research?}
    Negative Information: {negative_research?}

    Instructions:
    1. Write a neutral overview referencing information from both sides.
    2. Format the report clearly, divided into "Successes and Positive Impacts" and "Criticisms and Negative Impacts" sections.
    3. Use the 'write_file' tool to save the entire report.
        - directory: 'historical_reports'
        - filename: Use the topic name {TOPIC?}
        - content: The entire report content you have written.
    4. Write the entire report and its content in Thai.
    """,
    tools=[write_file],
)

# Main Sequential Agent
historical_court_system = SequentialAgent(
    name="historical_court_system",
    description="Runs the entire simulated historical court process.",
    sub_agents=[
        the_trial,
        verdict_agent
    ],
)

# The Inquiry (Root Agent)
root_agent = Agent(
    name="inquiry_agent",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Receives a historical topic from the user to begin the simulated court process.",
    instruction="""
    - Greet the user and inform them that you can help analyze a historical figure or event in a 'mock court' format.
    - Ask the user to specify a topic of interest (e.g., 'Genghis Khan', 'The Cold War').
    - When the user responds, use the 'append_to_state' tool to save that topic to the state under the key 'TOPIC'.
    - Then, delegate the task to the 'historical_court_system' agent to begin the process.
    - Communicate with the user in Thai.
    """,
    tools=[append_to_state],
    sub_agents=[historical_court_system],
)