from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from .tools import retrieve_context

root_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o-mini"),
    name="root_agent",
    description="A simple conversational agent using GPT-4o-mini.",
    instruction="""
You answer questions about Visio.

Use the retrieve_context tool to find answers. 
Do not make up answers, if the retrieve_context tool does not return any relevant documents, say you don't know.
""",
    tools=[retrieve_context],
)
