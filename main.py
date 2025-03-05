import os
import json
import logging
import requests
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph
from transformers import pipeline


# ===========================
# 🔹 CONFIGURATION
# ===========================
# Setting up API keys 
load_dotenv()  # Loading API keys from .env file

# Fetching API keys securely
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")

# Ensuring API key is present
if not TAVILY_API_KEY:
    raise ValueError("API key missing! Ensure .env file contains TAVILY_API_KEY.")

# Seting up API key in environment
os.environ["TAVILY_API_KEY"] = "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"


# Initializing Logger/ log file
logging.basicConfig(filename="ai_research.log", level=logging.INFO, format="%(asctime)s - %(message)s")


# Initializing Summarization Model Globally
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Load BART once
    logging.info("✅ Summarization model loaded successfully.")
except Exception as e:
    logging.error(f"⚠️ Error loading summarization model: {e}")
    summarizer = None  # Prevent crashes

# =========================== 
# 🔹 STATE SCHEMA
# ===========================
class ResearchState(TypedDict):
    query: str
    data: str
    summary: str


# ===========================
# 🔹 RESEARCH AGENT (Web Crawler)
# ===========================
def fetch_research_data(state: ResearchState) -> ResearchState:
    """Fetches research data using Tavily Search API."""
    query = state["query"]

    try:
        search_tool = TavilySearchResults()
        results = search_tool.invoke({"query": query, "num_results": 5})  # Fetch top 5 search results

        if not results:
            logging.warning(f"⚠️ No relevant data found for query: {query}")
            return {"query": query, "data": "No relevant data found.", "summary": ""}

        logging.info(f"✅ Fetched {len(results)} search results for query: {query}")
        return {"query": query, "data": json.dumps(results, indent=2), "summary": ""}

    except Exception as e:
        logging.error(f"❌ Research Agent Error: {e}")
        return {"query": query, "data": "Error retrieving research data.", "summary": ""}


# ===========================
# 🔹 Local DRAFTING AGENT (Summarizes Data) (No API Needed)
# ===========================
def generate_summary(state: ResearchState) -> ResearchState:
    """Summarizes research findings using a local Transformer model (offline)."""
    data = state["data"]

    if not data or data == "No relevant data found.":
        return {"query": state["query"], "data": data, "summary": "⚠️ No data available for summarization."}

    if summarizer is None:
        return {"query": state["query"], "data": data, "summary": "⚠️ Summarization model is unavailable."}

    try:
        summary = summarizer(data[:1024], max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        logging.info("✅ Successfully generated summary using local model.")
        return {"query": state["query"], "data": data, "summary": summary}

    except Exception as e:
        logging.error(f"❌ Summarization Agent Error: {e}")
        return {"query": state["query"], "data": data, "summary": "⚠️ Error generating summary."}


# ===========================
# 🔹 BUILD THE GRAPH (Fix for SimpleSequentialChain)
# ===========================
graph = StateGraph(ResearchState)
graph.add_node("ResearchAgent", fetch_research_data)
graph.add_node("SummarizationAgent", generate_summary)

graph.add_edge("ResearchAgent", "SummarizationAgent")  # Linking nodes
graph.set_entry_point("ResearchAgent")  # Setting entry point

workflow = graph.compile()  # Compiling workflow


# ===========================
# 🔹 RUN THE SYSTEM (Pipeline)
# ===========================
def run_ai_research_system(query: str) -> str:
    """Runs the AI research pipeline locally and saves output."""
    logging.info(f"🔎 Processing query: {query}")
    result = workflow.invoke({"query": query, "data": "", "summary": ""})
    
    # Save output to file
    output_data = {"query": query, "summary": result["summary"]}
    with open("research_output.json", "w") as file:
        json.dump(output_data, file, indent=4)
    
    return result["summary"]


def run_interactive_mode():
    """Interactive mode for user input."""
    query = input("🔍 Enter a research topic: ").strip()

    if not query:
        print("❌ Error: Query cannot be empty.")
        return

    print("\n🚀 Researching... Please wait...")
    research_data = fetch_research_data({"query": query, "data": "", "summary": ""})["data"]

    if research_data and research_data != "No relevant data found.":
        print("\n📝 Drafting Summary...")
        summary = generate_summary({"query": query, "data": research_data, "summary": ""})["summary"]

        # Save output to file
        output_data = {"query": query, "summary": summary}
        with open("research_output.json", "w") as file:
            json.dump(output_data, file, indent=4)

        print("\n✅ Research Summary:\n", summary)
    else:
        print("\n⚠️ No research data found!")


# ===========================
# 🔹 MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    query = "Latest advancements in AI research"
    output = run_ai_research_system(query)
    print("\n📄 Final Research Summary:\n", output)