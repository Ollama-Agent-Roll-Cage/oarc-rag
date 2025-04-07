# OARC-RAG

A powerful Retrieval-Augmented Generation (RAG) system, designed for flexible integration with various AI applications.

## Key Features

| Feature                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Advanced RAG Engine**  | Built on LangChain and LlamaIndex for high-quality retrievals              |
| **Modular Architecture** | Easily integrate with diverse project architectures                        |
| **Ollama Integration**   | Seamless connection with local Ollama models                               |
| **Vector Database Support** | Utilize FAISS for efficient embedding storage and retrieval             |
| **Customizable Pipelines** | Adapt retrieval strategies based on your specific use case               |

## Installation

```bash
pip install oarc-rag
```

For development:

```bash
# Clone the repository
git clone https://github.com/yourusername/oarc-rag.git
cd oarc-rag

# Create and use virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Linux: source .venv/bin/activate 

# Install in development mode
pip install -e .  # For development: 'pip install -e .[dev]'
```

## Requirements

- Python 3.8,<=3.11
- [Ollama](https://ollama.ai/download) running locally with compatible models

## Usage

```python
from oarc_rag.engine import RAGEngine

# Initialize the RAG engine
engine = RAGEngine(
    documents=["path/to/document.pdf"],
    model="llama3"
)

# Query the RAG system
response = engine.query("What is retrieval-augmented generation?")
print(response)
```

## Documentation

For complete documentation, see the [docs](./docs/) directory.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Acknowledgements

This project was extracted originally from the [oarc_rag](https://github.com/Ollama-Agent-Roll-Cage/oarc-oarc_rag) curriculum generation system by [p3nGu1nZz](https://github.com/p3nGu1nZz)
