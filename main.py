import os
import json
import logging
import requests
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
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


# Setting up logging
logging.basicConfig(filename="ai_research.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Loading summarization model
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    logging.error(f"Summarization model error: {e}")
    summarizer = None

# State representation for the AI research system
class ResearchState(TypedDict):
    query: str
    data: str
    summary: str

# Fetching research data from Tavily
def fetch_research_data(state: ResearchState) -> ResearchState:
    """Fetches research data from Tavily."""
    query = state["query"]
    try:
        logging.info(f"Fetching data for query: {query}")
        search_tool = TavilySearchResults()
        results = search_tool.invoke({"query": query, "num_results": 5})
        
        if not results:
            logging.warning(f"No data found for: {query}")
            return {"query": query, "data": "No relevant data found.", "summary": ""}
        fetched_data = json.dumps(results, indent=2)
        logging.info(f"Fetched data: {fetched_data[:500]}...")  # Log first 500 chars

        return {"query": query, "data": fetched_data, "summary": ""}
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
        logging.info("Generating summary...")
        summary = summarizer(data[:1024], max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        logging.info(f"Summary generated: {summary[:200]}...")  # Log first 200 chars
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
    
    # Invoke LangGraph workflow
    try:
        result = workflow.invoke({"query": query, "data": "", "summary": ""})
    except Exception as e:
        logging.error(f"Workflow execution error: {e}")
        return "Error executing workflow."

    output_data = {"query": query, "data": result["data"], "summary": result["summary"]}
    
    with open("research_output.json", "w") as file:
        json.dump(output_data, file, indent=4)

    return result["summary"]


# Main execution
if __name__ == "__main__":
    print("\nWelcome to the AI Research System!")
    print("You can enter a topic, and the system will fetch relevant research data and summarize it.\n")
    
    query = input("Enter your research topic: ").strip()  # Prompting user for input
    
    if not query:
        print("\nError: Query cannot be empty. Please restart and enter a valid topic.")
    else:
        print("\nFetching research data, please wait...\n")
        output = run_ai_research_system(query)
        print("\nResearch Summary:\n", output)