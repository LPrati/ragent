from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    model=LiteLlm(model="openai/gpt-4o-mini"),
    name="root_agent",
    description="A simple conversational agent using GPT-4o-mini.",
    instruction="Answer user questions politely and concisely.",
)
