
# MCP OpenAI Demo 1

This project demonstrates a simple agentic architecture using the Model Context Protocol (MCP) to connect a Python client with an AI-powered server. It supports both OpenAI and Google Gemini models for tool-augmented conversations.

## Project Structure

- **mcp-client/**: Contains the client code to interact with the MCP server. Includes support for OpenAI and Gemini APIs.
  - `client.py`: Main client for Gemini API.
  - `openai_client.py`: Client for OpenAI API.
  - `requirements.txt` and `pyproject.toml`: Python dependencies.
  - `test_gemini.py`, `test_openai.py`: Example test scripts.
- **server/**: Contains the MCP server implementation.
  - `main.py`: Defines a simple MCP server with a shell command tool.
  - `pyproject.toml`: Server dependencies.

## How It Works

1. The server exposes tools (e.g., running shell commands) via MCP.
2. The client connects to the server and can use LLMs (OpenAI or Gemini) to call these tools.
3. The architecture allows for easy extension with new tools or models.

## Usage

1. **Install dependencies** in both `mcp-client` and `server` folders:
	```sh
	pip install -r requirements.txt
	```
2. **Set up your API keys** in a `.env` file in `mcp-client/` (e.g., `GEMINI_API_KEY` for Gemini, `OPENAI_API_KEY` for OpenAI).
3. **Start the server:**
	```sh
	python server/main.py
	```
4. **Run the client:**
	```sh
	python mcp-client/client.py server/main.py
	# or for OpenAI
	python mcp-client/openai_client.py server/main.py
	```

## License

MIT License