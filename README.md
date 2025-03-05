# AI Research Agent System

Overview:

This project is a Deep Research AI Agentic System designed to crawl websites and gather online information using Tavily Search API. The system is built with LangGraph and LangChain, implementing a multi-agent approach where:

One agent focuses on research and data collection
1. Another agent works as an answer drafter by summarizing the collected data
2. The system efficiently organizes and processes gathered information, making it 
3. useful for automated research tasks.

Features

1. Web crawling using Tavily API
2. Summarization of research findings using
Transformer models
3. Multi-agent workflow with LangGraph
4. Saves results in a structured JSON format
5. Simple and interactive CLI-based input

Tech Stack
* Python
* LangGraph & LangChain
* Tavily API
* Transformers (Hugging Face)
* Logging & JSON Handling

Installation & Setup

1. Clone the Repository

git clone https://github.com/your-username/your-repo.git

cd your-repo

2. Set Up a Virtual Environment

python -m venv venv

Activate it:

* PowerShell: venv\Scripts\Activate
* CMD: venv\Scripts\activate.bat
* Git Bash: source venv/Scripts/activate

3. Install Dependencies

pip install -r requirements.txt

4. Set Up Environment Variables
Create a .env file and add your Tavily API key:

ini
TAVILY_API_KEY=your_api_key_here

Usage:
Run the AI Research System

python main.py

You will be prompted to enter a research topic in the terminal.

Example Queries:

1. Enter research topic: Latest advancements in AI  
2. Enter research topic: Impact of quantum computing on cryptography  
3. Enter research topic: Evolution of web development frameworks  
4. The system will fetch the research data, summarize it, and save the output to research_output.json.

^ Future Improvements

* Improve data filtering for more relevant results
* Optimize summarization for better context understanding
* Implement web UI for better user interaction

License

This project is for research purposes. Modify and use it as needed.

