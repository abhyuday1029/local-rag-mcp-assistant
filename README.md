# Local RAG Assistant with MCP Weather Integration

A Python chatbot that answers questions using a local knowledge base
and fetches current weather through an MCP tool.

## Features

- Local answer generation using Ollama and Gemma 3 1B
- Sentence Transformer embeddings
- In-memory cosine-similarity retrieval
- MCP weather server using wttr.in
- Terminal chat interface

## Requirements

- Python 3.12
- Ollama
- Internet access for initial downloads and weather requests

Developed and tested on Windows 11.

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install Ollama from https://ollama.com/download/windows
and download the model:

```powershell
ollama pull gemma3:1b
```

Keep Ollama running, then start the chatbot:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Examples

- What is RAG?
- What is the purpose of embeddings?
- What is the current weather?

Weather requests prompt for a city.

Type `exit` to stop.

## Project files

- `app.py`: Chat loop, routing, MCP client, and Ollama integration
- `rag.py`: Knowledge loading, embeddings, and similarity search
- `weather_server.py`: MCP weather tool
- `test_mcp.py`: MCP connection and weather-tool test
- `knowledge_base.txt`: Sample knowledge passages

## Limitations

- Tool selection uses a Python weather rule.
- Conversations are not remembered between questions.
- Retrieval searches all document vectors in memory.
- The small local model can produce inaccurate answers.
- Weather availability depends on the external service.

## Notes

This project was developed from an MCP and RAG tutorial,
then adapted to use Ollama and in-memory retrieval.
No paid model API key is required.