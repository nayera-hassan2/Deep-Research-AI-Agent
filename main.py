import os
import json
import logging
import requests
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph
from transformers import pipeline


# Setting up API keys 
load_dotenv()  # Loading API keys from .env file

# Fetching API keys securely
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")

# Ensuring API key is present
if not TAVILY_API_KEY:
    raise ValueError("API key missing! Add TAVILY_API_KEY to .env.")

# Seting up API key in environment
os.environ["TAVILY_API_KEY"] = "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"


# Logger
logging.basicConfig(filename="ai_research.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Loading summarization model
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    logging.error(f"Summarization model error: {e}")
    summarizer = None

# State representation for the AI research system
class ResearchState(dict):
    query: str
    data: str
    summary: str

# Fetch research data from Tavily
def fetch_research_data(state: ResearchState) -> ResearchState:
    """Fetches research data from Tavily."""
    query = state["query"]
    try:
        search_tool = TavilySearchResults()
        results = search_tool.invoke({"query": query, "num_results": 5})
        if not results:
            logging.warning(f"No data found for: {query}")
            return {"query": query, "data": "No relevant data found.", "summary": ""}
        return {"query": query, "data": json.dumps(results, indent=2), "summary": ""}
    except Exception as e:
        logging.error(f"Research Agent error: {e}")
        return {"query": query, "data": "Error retrieving data.", "summary": ""}

# Summarize research findings
def generate_summary(state: ResearchState) -> ResearchState:
    """Summarizes research findings."""
    data = state["data"]
    if not data or data == "No relevant data found.":
        return {"query": state["query"], "data": data, "summary": "No data available for summarization."}
    if summarizer is None:
        return {"query": state["query"], "data": data, "summary": "Summarization model unavailable."}
    try:
        summary = summarizer(data[:1024], max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        return {"query": state["query"], "data": data, "summary": summary}
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return {"query": state["query"], "data": data, "summary": "Error generating summary."}

# Define research workflow
graph = StateGraph(ResearchState)
graph.add_node("ResearchAgent", fetch_research_data)
graph.add_node("SummarizationAgent", generate_summary)
graph.add_edge("ResearchAgent", "SummarizationAgent")
graph.set_entry_point("ResearchAgent")
workflow = graph.compile()

# Run AI research pipeline
def run_ai_research_system(query: str = None) -> str:
    """Runs AI research pipeline with predefined or user input."""
    if not query:
        query = input("Enter research topic: ").strip()
        if not query:
            print("Error: Query cannot be empty.")
            return ""
    logging.info(f"Processing query: {query}")
    result = workflow.invoke({"query": query, "data": "", "summary": ""})
    
    output_data = {"query": query, "data": result["data"], "summary": result["summary"]}
    with open("research_output.json", "w") as file:
        json.dump(output_data, file, indent=4)
    
    return result["summary"]

if __name__ == "__main__":
    query = None  # Can Set query here for automation or leave None for manual input
    output = run_ai_research_system(query)
    print("\nResearch Summary:\n", output)