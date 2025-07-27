# obsidian-ai-refactor

Obsidian notes refactoring using AI Workflow and Agents

## Overview

This project provides AI-powered tools and workflows for refactoring and enhancing Obsidian notes. It includes multi-agent systems and automated workflows to improve note structure, content quality, and knowledge organization.

## Setup

### Prerequisites
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver

### Installation
```bash
# Initialize the project (creates pyproject.toml and basic structure)
uv init

# Install dependencies and create virtual environment
uv sync

# Add new packages (they will be automatically added to pyproject.toml)
uv add requests python-dotenv openai pypdf
```

### Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running Scripts

### Running Scripts

#### AI Workflow Demo
```bash
# Basic usage with vault path
uv run demo_obsidian_refactor_ai_workflow.py /path/to/vault

# Dry run to see what would be changed
uv run demo_obsidian_refactor_ai_workflow.py /path/to/vault --dry-run

# Live refactoring with custom template
uv run demo_obsidian_refactor_ai_workflow.py /path/to/vault --template=./my_template.md
```

#### Multi-Agent Demo
```bash
# Basic usage with vault path
uv run demo_obsidian_refactor_ai_multi_agent.py /path/to/vault

# Dry run to see what would be changed
uv run demo_obsidian_refactor_ai_multi_agent.py /path/to/vault --dry-run

# Live refactoring with custom template
uv run demo_obsidian_refactor_ai_multi_agent.py /path/to/vault --template=./my_template.md
```

**Command-line Options:**
- `--dry-run`: Preview changes without modifying files
- `--template=<path>`: Use a custom note template (default: `./template/note-template.md`)

**Example with Windows paths:**
```bash
uv run demo_obsidian_refactor_ai_multi_agent.py "C:\ObsidianWorkspace\205_AI_And_Machine_Learning" --dry-run
```

### Using with Virtual Environment
```bash
# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows

# Run scripts directly with vault path and options
python demo_obsidian_refactor_ai_workflow.py /path/to/vault --dry-run
python demo_obsidian_refactor_ai_multi_agent.py /path/to/vault --template=./custom_template.md
```

## Project Structure

```
obsidian-ai-refactor/
├── demo_obsidian_refactor_ai_workflow.py    # AI workflow demonstration
├── demo_obsidian_refactor_ai_multi_agent.py # Multi-agent system demo
├── template/
│   └── note-template.md                     # Improved Obsidian note template
├── examples/
│   ├── ollama-local-llm-setup.md           # Local LLM setup guide
│   └── rag-vs-cag-comparison.md            # RAG vs CAG comparison
├── pyproject.toml                           # Project dependencies (uv)
├── README.md                                # This file
├── LICENSE                                  # Project license
└── .env                                     # Environment variables (create this)
```

## Features

- **AI-Powered Note Refactoring**: Automatically improve note structure and content
- **Multi-Agent Collaboration**: Multiple AI agents working together for comprehensive analysis
- **Template System**: Standardized note templates for consistent formatting
- **Local LLM Support**: Integration with Ollama for privacy-focused processing
- **RAG/CAG Strategies**: Support for different data augmentation approaches

## Examples

The `examples/` directory contains sample notes demonstrating:
- **Ollama Setup**: Complete guide for running local LLMs
- **RAG vs CAG**: Comprehensive comparison of data augmentation strategies
- **Note Templates**: Improved templates following best practices

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
