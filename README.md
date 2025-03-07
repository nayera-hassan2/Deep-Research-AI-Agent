# AI Research Agent System

Overview

The AI Research Agent System is a Deep Research AI Agentic System designed to crawl websites and gather online information using the Tavily Search API. This system leverages LangGraph and LangChain to implement a multi-agent approach:

1. Research Agent - Gathers relevant data from the web.

2. Answer Drafter Agent - Summarizes and structures the collected data into meaningful insights.

3. LangGraph Orchestration - Ensures seamless coordination between agents for efficient processing.

The system automates research tasks, making it a valuable tool for data gathering and summarization.

Features

* Web crawling using Tavily API
* AI-powered summarization using transformer models
* Multi-agent workflow with LangGraph
* Results saved in a structured JSON format
* Simple and interactive CLI-based input

Tech Stack

* Python

* LangGraph & LangChain

* Tavily API

* Transformers (Hugging Face)

* Logging & JSON Handling

Installation & Setup

1. Clone the Repository

git clone https://github.com/nayera-hassan2/Deep-Research-AI-Agent.git

cd Deep-Research-AI-Agent

2. Set Up a Virtual Environment

python -m venv venv

Activate the virtual environment based on your OS:

* PowerShell: venv\Scripts\Activate

* CMD: venv\Scripts\activate.bat

* Git Bash/Linux/macOS: source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Up Environment Variables

Create a .env file and add your Tavily API key:

TAVILY_API_KEY=your_api_key_here

Usage

Running the AI Research System

python main.py

You will be prompted to enter a research topic in the terminal.

Example Queries:

* Enter research topic: Latest advancements in AI  
* Enter research topic: Impact of quantum computing on cryptography  
* Enter research topic: Evolution of web development frameworks  

The system will fetch research data, summarize it, and save the output to research_output.json.

Future Improvements

* Enhanced data filtering for more relevant results
* Optimized summarization for better context understanding
* Web-based UI for better user interaction

License

This project is for research purposes. Modify and use it as needed.