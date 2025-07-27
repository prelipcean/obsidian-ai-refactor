---
tags: [ollama, local-llm, ai-models, privacy, offline-ai, command-line, llama, mistral, gemma]
aliases: [Ollama Guide, Local LLM Setup, Running LLMs Locally]
created: 2025-07-26
updated: 2025-07-26
---

# Running Local LLMs with Ollama

## Summary

Ollama is a powerful tool for running large language models locally on your machine. This provides privacy, offline capabilities, and cost savings while giving you full control over your AI interactions. This guide covers installation, model management, and usage patterns.

**Key takeaways:**
- Run LLMs locally for privacy and cost savings
- Simple CLI interface for model management
- Supports multiple popular models (Llama, Mistral, CodeLlama)
- REST API enables integration with applications
- Hardware requirements vary by model size

## What is Ollama?

**Ollama** is an open-source application that makes it easy to run large language models on your local machine. It handles model downloading, optimization, and serving through a simple command-line interface and REST API.

### Benefits of Local LLMs
- 🔒 **Privacy** - Data never leaves your machine
- 💰 **Cost-effective** - No API usage fees
- 🌐 **Offline capability** - Works without internet
- ⚡ **Speed** - No network latency for inference
- 🎛️ **Full control** - Customize parameters and behavior

## Installation

### Windows
```powershell
# Method 1: Download from website
# Visit https://ollama.com and download the installer

# Method 2: Using winget
winget install Ollama.Ollama

# Method 3: Using chocolatey
choco install ollama
```

### macOS
```bash
# Method 1: Download from website
# Visit https://ollama.com

# Method 2: Using Homebrew
brew install ollama
```

### Linux
```bash
# Official installation script
curl -fsSL https://ollama.com/install.sh | sh

# Or manually download from GitHub releases
```

## Essential Ollama Commands

### Model Management

#### List Available Models
```bash
ollama list                    # Show installed models
ollama ps                      # Show currently running models
```

#### Download Models
```bash
ollama pull llama3.2          # Download Llama 3.2 (8B)
ollama pull llama3.2:70b      # Download Llama 3.2 70B
ollama pull mistral           # Download Mistral 7B
ollama pull gemma2:27b        # Download Gemma 2 27B
ollama pull codellama         # Download Code Llama
ollama pull phi3              # Download Microsoft Phi-3
ollama pull qwen2.5-coder     # Download Qwen 2.5 Coder
ollama pull deepseek-coder    # Download DeepSeek Coder
```

#### Remove Models
```bash
ollama rm llama3.2            # Remove specific model
ollama rm mistral:7b          # Remove specific version
```

### Running Models

#### Interactive Chat
```bash
ollama run llama3.2           # Start interactive chat with Llama 3.2
ollama run mistral            # Start interactive chat with Mistral
ollama run codellama          # Start interactive chat with Code Llama

# Exit interactive mode with: /bye or Ctrl+C
```

#### Single Prompt
```bash
ollama run llama3.2 "Explain quantum computing in simple terms"
ollama run codellama "Write a Python function to sort a list"
```

### Advanced Commands

#### Model Information
```bash
ollama show llama3.2          # Show model details and parameters
ollama show mistral --verbose # Show detailed model information
```

#### Server Management
```bash
ollama serve                  # Start Ollama server manually
ollama stop                   # Stop all running models
```

#### Custom Model Creation
```bash
ollama create mymodel -f ./Modelfile    # Create custom model from Modelfile
ollama push mymodel                     # Share model (requires account)
```

## Popular Models and Use Cases

### General Purpose Models

#### Llama 3.2 (Meta)
```bash
ollama pull llama3.2          # 8B parameters - Good balance
ollama pull llama3.2:70b      # 70B parameters - Higher quality
```
- **Best for**: General conversation, reasoning, creative writing
- **Size**: 8B (~4.7GB), 70B (~40GB)
- **RAM needed**: 8GB minimum, 64GB+ for 70B

#### Mistral 7B
```bash
ollama pull mistral           # 7B parameters
ollama pull mistral:instruct  # Instruction-tuned version
```
- **Best for**: Fast responses, general tasks
- **Size**: ~4GB
- **RAM needed**: 8GB minimum

### Code-Focused Models

#### Code Llama
```bash
ollama pull codellama         # 7B version
ollama pull codellama:13b     # 13B version for better code quality
ollama pull codellama:34b     # 34B version (requires more RAM)
```
- **Best for**: Code generation, debugging, code explanation
- **Languages**: Python, JavaScript, C++, Java, and more

#### Qwen 2.5 Coder
```bash
ollama pull qwen2.5-coder     # Optimized for coding tasks
ollama pull qwen2.5-coder:7b  # 7B parameter version
```
- **Best for**: Advanced coding, code review, technical documentation

#### DeepSeek Coder
```bash
ollama pull deepseek-coder    # Strong coding capabilities
ollama pull deepseek-coder:6.7b
```
- **Best for**: Code completion, bug fixes, algorithm implementation

### Specialized Models

#### Phi-3 (Microsoft)
```bash
ollama pull phi3              # Small but capable model
ollama pull phi3:mini         # Even smaller version
```
- **Best for**: Resource-constrained environments, quick responses
- **Size**: ~2GB
- **RAM needed**: 4GB minimum

#### Gemma 2 (Google)
```bash
ollama pull gemma2:9b         # 9B parameter version
ollama pull gemma2:27b        # 27B parameter version
```
- **Best for**: Instruction following, factual accuracy

## Using Ollama with Code

### Python Integration

#### Basic Usage
```python
import requests
import json

def chat_with_ollama(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}"

# Usage example
response = chat_with_ollama("Write a Python function to calculate fibonacci")
print(response)
```

#### Streaming Responses
```python
import requests
import json

def stream_ollama_response(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    
    with requests.post(url, json=data, stream=True) as response:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if not chunk.get("done"):
                    print(chunk["response"], end="", flush=True)
                else:
                    print()  # New line when done
                    break

# Usage
stream_ollama_response("Explain machine learning concepts")
```

#### Chat API (Conversation)
```python
import requests

def chat_conversation(messages, model="llama3.2"):
    url = "http://localhost:11434/api/chat"
    data = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    return response.json()["message"]["content"]

# Multi-turn conversation
messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."},
    {"role": "user", "content": "Show me a simple example"}
]

response = chat_conversation(messages)
print(response)
```

### JavaScript/Node.js Integration
```javascript
const fetch = require('node-fetch');

async function chatWithOllama(prompt, model = 'llama3.2') {
    const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: model,
            prompt: prompt,
            stream: false
        })
    });
    
    const data = await response.json();
    return data.response;
}

// Usage
chatWithOllama("Explain async/await in JavaScript")
    .then(response => console.log(response))
    .catch(error => console.error('Error:', error));
```

## Configuration and Optimization

### Model Parameters
```bash
# Run with custom parameters
ollama run llama3.2 --temperature 0.7 --top-p 0.9 "Creative story about AI"

# Set context length
ollama run mistral --num-ctx 4096 "Long conversation context"
```

### Environment Variables
```bash
# Set default model directory
export OLLAMA_MODELS="/custom/path/to/models"

# Set server host and port
export OLLAMA_HOST="0.0.0.0:11434"

# Set number of parallel requests
export OLLAMA_NUM_PARALLEL=4

# Set GPU usage
export OLLAMA_GPU_DEVICE=0
```

### Modelfile Creation
Create a custom model configuration:

```dockerfile
# Modelfile
FROM llama3.2

# Set parameters
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40

# Set system message
SYSTEM You are a helpful coding assistant specialized in Python development.

# Set template
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
"""
```

```bash
# Create custom model
ollama create my-python-assistant -f ./Modelfile
ollama run my-python-assistant
```

## Performance Tips

### Hardware Requirements

| Model Size | Minimum RAM | Recommended RAM | VRAM (GPU) |
|------------|-------------|-----------------|------------|
| 7B models  | 8GB         | 16GB           | 6GB        |
| 13B models | 16GB        | 32GB           | 10GB       |
| 34B models | 32GB        | 64GB           | 20GB       |
| 70B models | 64GB        | 128GB          | 40GB       |

### Optimization Strategies

#### GPU Acceleration
```bash
# Check GPU support
ollama run llama3.2 --verbose

# Force CPU usage
OLLAMA_NUM_GPU=0 ollama run mistral

# Use specific GPU
CUDA_VISIBLE_DEVICES=1 ollama run llama3.2
```

#### Memory Management
```bash
# Limit concurrent models
export OLLAMA_MAX_LOADED_MODELS=1

# Set model timeout
export OLLAMA_KEEP_ALIVE=5m

# Reduce memory usage
ollama run llama3.2 --num-thread 4
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using port 11434
netstat -tulpn | grep 11434

# Kill process if needed
sudo kill -9 $(sudo lsof -t -i:11434)

# Start with different port
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

#### Model Download Issues
```bash
# Clear download cache
rm -rf ~/.ollama/models/.tmp

# Retry with verbose output
ollama pull llama3.2 --verbose

# Check disk space
df -h ~/.ollama
```

#### Out of Memory Errors
```bash
# Use smaller model
ollama run phi3 instead of llama3.2:70b

# Reduce context length
ollama run llama3.2 --num-ctx 2048

# Enable memory mapping
export OLLAMA_MMAP=1
```

#### Connection Issues
```bash
# Check if server is running
curl http://localhost:11434/api/version

# Restart Ollama service
sudo systemctl restart ollama  # Linux
brew services restart ollama   # macOS

# Check logs
journalctl -u ollama -f        # Linux
```

## Use Cases and Examples

## Use Cases

### Code Assistant
- Code review and analysis
- Bug detection and fixing
- Algorithm explanation and optimization
- Documentation generation

### Writing Assistant  
- Blog post and article writing
- Technical documentation creation
- Creative writing and storytelling
- Content editing and improvement

### Technical Analysis
- System architecture design
- Technology comparison and evaluation
- Problem-solving and troubleshooting
- Research and analysis tasks

## Integration with Other Tools

## Integration with Other Tools

### VS Code Extension
Install the "Ollama" extension for VS Code to use local models directly in your editor.

### Jupyter Notebooks
```python
# Install ollama library
!pip install ollama

import ollama

# Use in notebooks
response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Explain pandas dataframes'}
])
print(response['message']['content'])
```

### Command Line Tools
```bash
# Create aliases for common tasks
alias code-review="ollama run codellama 'Review this code:'"
alias explain-code="ollama run deepseek-coder 'Explain this code:'"
alias write-docs="ollama run mistral 'Write documentation for:'"
```

## Questions & Next Steps

- [ ] Test performance comparison between different models
- [ ] Explore custom Modelfile configurations for specific use cases  
- [ ] Research GPU optimization strategies for better performance
- [ ] Investigate integration with IDEs beyond VS Code
- [ ] Compare Ollama with other local LLM solutions (LMStudio, etc.)

## Use Cases
```

## Related Concepts

- [[AI Development Environment]] - Setting up development tools for AI
- [[Local AI Setup]] - General guide to running AI locally  
- [[Privacy-First AI]] - AI solutions that prioritize data privacy
- [[Offline AI Development]] - Working with AI without internet dependency
- [[Vector Databases]] - Storage solutions for AI applications

### See Also
- #local-llm #privacy #ai-tools
- [[MOC - AI Development]] - Map of Content for AI development topics

## References

- [Ollama Official Website](https://ollama.com) - Download and documentation
- [Ollama GitHub Repository](https://github.com/ollama/ollama) - Source code and issues
- [Model Hub](https://ollama.com/library) - Available models and specifications
- [Meta AI Research](https://ai.meta.com/research/) - Llama model research papers
- [Mistral AI Documentation](https://docs.mistral.ai/) - Mistral model information

## Metadata

**Confidence Level:** High
**Last Reviewed:** 2025-07-26
**Review Due:** 2025-08-26

---
*Created: 2025-07-26*
*Last updated: 2025-07-26*
