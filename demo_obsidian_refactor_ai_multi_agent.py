import sys
import os
import json
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# --- SETUP: Load environment and OpenAI client ---
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Default paths
DEFAULT_TEMPLATE_PATH = "./template/note-template.md"
DEFAULT_EXAMPLES_PATH = "./examples/"


# --- DATA STRUCTURES ---

@dataclass
class NoteInfo:
    """Information about a markdown note file"""
    file_path: str
    relative_path: str
    content: str
    frontmatter: str
    body_content: str
    creation_date: str
    modification_date: str


@dataclass
class AnalysisResult:
    """Result of note structure analysis"""
    has_frontmatter: bool
    missing_sections: List[str]
    current_sections: List[str]
    template_sections: List[str]
    compliance_score: float
    needs_restructuring: bool


@dataclass
class ProcessingResult:
    """Result of note processing"""
    success: bool
    action_taken: str
    error_message: Optional[str] = None
    backup_path: Optional[str] = None


# --- TOOL BASE CLASS ---

class Tool:
    """Base class for all tools used by agents"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def get_schema(self) -> Dict[str, Any]:
        """Returns OpenAI-compatible tool schema"""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "additionalProperties": False,
                "required": list(self.parameters.keys()),
            },
        }

    def execute(self, arguments: str) -> Any:
        """Execute the tool with given arguments"""
        raise NotImplementedError("Each tool must implement execute method")


# --- TOOL IMPLEMENTATIONS ---

class FileSystemTool(Tool):
    """Tool for file system operations"""
    
    def __init__(self):
        super().__init__(
            name="file_system_operation",
            description="Perform file system operations like reading files, finding markdown files, etc.",
            parameters={
                "operation": {
                    "type": "string", 
                    "enum": ["read_file", "write_file", "find_markdown_files", "validate_vault"],
                    "description": "The file system operation to perform"
                },
                "path": {"type": "string", "description": "File or directory path"},
                "content": {"type": "string", "description": "Content to write (for write operations)"}
            }
        )

    def execute(self, arguments: str) -> Any:
        try:
            args = json.loads(arguments)
            operation = args["operation"]
            path = args["path"]
            
            if operation == "read_file":
                return self._read_file(path)
            elif operation == "write_file":
                content = args.get("content", "")
                return self._write_file(path, content)
            elif operation == "find_markdown_files":
                return self._find_markdown_files(path)
            elif operation == "validate_vault":
                return self._validate_vault(path)
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"error": f"FileSystemTool error: {str(e)}"}

    def _read_file(self, path: str) -> Dict[str, Any]:
        """Read file content"""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return {"success": True, "content": file.read()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to file"""
        try:
            with open(path, 'w', encoding='utf-8') as file:
                file.write(content)
            return {"success": True, "message": f"File written: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _find_markdown_files(self, vault_path: str) -> Dict[str, Any]:
        """Find all markdown files in vault"""
        try:
            md_files = []
            for root, dirs, files in os.walk(vault_path):
                if '.obsidian' in dirs:
                    dirs.remove('.obsidian')
                
                for file in files:
                    if file.lower().endswith(('.md', '.markdown')):
                        md_files.append(os.path.join(root, file))
            
            return {"success": True, "files": md_files}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _validate_vault(self, vault_path: str) -> Dict[str, Any]:
        """Validate if path is a valid Obsidian vault"""
        try:
            if not os.path.exists(vault_path) or not os.path.isdir(vault_path):
                return {"success": False, "valid": False, "reason": "Path does not exist or is not a directory"}
            
            obsidian_dir = os.path.join(vault_path, '.obsidian')
            has_md_files = any(f.endswith('.md') for f in os.listdir(vault_path) 
                             if os.path.isfile(os.path.join(vault_path, f)))
            
            valid = os.path.exists(obsidian_dir) or has_md_files
            return {"success": True, "valid": valid, "has_obsidian_dir": os.path.exists(obsidian_dir), "has_md_files": has_md_files}
        except Exception as e:
            return {"success": False, "error": str(e)}


class NoteAnalysisTool(Tool):
    """Tool for analyzing note structure and compliance"""
    
    def __init__(self):
        super().__init__(
            name="analyze_note_structure",
            description="Analyze a note's structure against a template to determine compliance and missing sections",
            parameters={
                "note_content": {"type": "string", "description": "The content of the note to analyze"},
                "template_content": {"type": "string", "description": "The template content to compare against"}
            }
        )

    def execute(self, arguments: str) -> Any:
        try:
            args = json.loads(arguments)
            note_content = args["note_content"]
            template_content = args["template_content"]
            
            analysis = self._analyze_structure(note_content, template_content)
            return {"success": True, "analysis": analysis}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _analyze_structure(self, note_content: str, template_content: str) -> Dict[str, Any]:
        """Compare note structure to template"""
        note_frontmatter, note_body = self._extract_frontmatter_and_content(note_content)
        template_frontmatter, template_body = self._extract_frontmatter_and_content(template_content)
        
        note_sections = self._extract_sections(note_body)
        template_sections = self._extract_sections(template_body)
        
        missing_sections = [section for section in template_sections if section not in note_sections]
        compliance_score = len(set(note_sections) & set(template_sections)) / len(template_sections) if template_sections else 1.0
        
        return {
            "has_frontmatter": bool(note_frontmatter),
            "missing_sections": missing_sections,
            "current_sections": note_sections,
            "template_sections": template_sections,
            "compliance_score": compliance_score,
            "needs_restructuring": compliance_score < 0.8 or len(missing_sections) > 0
        }

    def _extract_frontmatter_and_content(self, content: str) -> Tuple[str, str]:
        """Extract frontmatter and body content"""
        lines = content.strip().splitlines()
        
        if len(lines) < 2 or lines[0].strip() != "---":
            return "", content
        
        frontmatter_lines = []
        content_lines = []
        frontmatter_end_found = False
        
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---" and not frontmatter_end_found:
                frontmatter_end_found = True
                continue
            
            if not frontmatter_end_found:
                frontmatter_lines.append(line)
            else:
                content_lines.append(line)
        
        frontmatter = "\n".join(frontmatter_lines) if frontmatter_lines else ""
        content = "\n".join(content_lines) if content_lines else ""
        
        return frontmatter, content

    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headers from content"""
        sections = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                section = line.lstrip('#').strip()
                if section:
                    sections.append(section)
        
        return sections


class TemplateTool(Tool):
    """Tool for processing template placeholders"""
    
    def __init__(self):
        super().__init__(
            name="process_template",
            description="Process template placeholders with actual file dates and metadata",
            parameters={
                "template_content": {"type": "string", "description": "Template content with placeholders"},
                "file_path": {"type": "string", "description": "Path to the file being processed"}
            }
        )

    def execute(self, arguments: str) -> Any:
        try:
            args = json.loads(arguments)
            template_content = args["template_content"]
            file_path = args["file_path"]
            
            processed_content = self._process_placeholders(template_content, file_path)
            return {"success": True, "processed_content": processed_content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _process_placeholders(self, template_content: str, file_path: str) -> str:
        """Replace template placeholders with actual values"""
        processed_content = template_content
        
        # Get file dates
        creation_date = self._get_file_creation_date(file_path)
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Replace date placeholders
        lines = processed_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('created:') and '{{date:YYYY-MM-DD}}' in line:
                lines[i] = line.replace('{{date:YYYY-MM-DD}}', creation_date)
            elif line.strip().startswith('updated:') and '{{date:YYYY-MM-DD}}' in line:
                lines[i] = line.replace('{{date:YYYY-MM-DD}}', current_date)
            elif '{{date:YYYY-MM-DD}}' in line and not line.strip().startswith(('created:', 'updated:')):
                # For other date placeholders, use creation date
                lines[i] = line.replace('{{date:YYYY-MM-DD}}', creation_date)
        
        processed_content = '\n'.join(lines)
        
        # Replace other placeholders
        processed_content = processed_content.replace('{{created}}', creation_date)
        processed_content = processed_content.replace('{{updated}}', current_date)
        processed_content = processed_content.replace('{{today}}', current_date)
        
        # Handle footer date patterns specifically
        lines = processed_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('*Created:') and '{{date:YYYY-MM-DD}}' in line:
                lines[i] = line.replace('{{date:YYYY-MM-DD}}', creation_date)
            elif line.strip().startswith('*Last updated:') and '{{date:YYYY-MM-DD}}' in line:
                lines[i] = line.replace('{{date:YYYY-MM-DD}}', current_date)
        
        processed_content = '\n'.join(lines)
        
        # Extract title from filename
        file_name = os.path.basename(file_path)
        title = os.path.splitext(file_name)[0].replace('_', ' ').replace('-', ' ').title()
        processed_content = processed_content.replace('{{title}}', title)
        
        return processed_content

    def _get_file_creation_date(self, file_path: str) -> str:
        """Get file creation date"""
        try:
            if os.name == 'nt':  # Windows
                path_obj = Path(file_path)
                stat_result = path_obj.stat()
                creation_time = min(stat_result.st_ctime, stat_result.st_mtime)
            else:  # Unix-like
                stat = os.stat(file_path)
                creation_time = getattr(stat, 'st_birthtime', stat.st_mtime)
            
            return datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d')
        except Exception:
            return datetime.now().strftime('%Y-%m-%d')


class AIRestructuringTool(Tool):
    """Tool for AI-powered note restructuring"""
    
    def __init__(self):
        super().__init__(
            name="restructure_note_ai",
            description="Use AI to restructure a note based on template and examples",
            parameters={
                "note_content": {"type": "string", "description": "Original note content"},
                "processed_template": {"type": "string", "description": "Template with processed placeholders"},
                "examples": {"type": "string", "description": "JSON string of example notes"},
                "analysis": {"type": "string", "description": "JSON string of analysis results"}
            }
        )

    def execute(self, arguments: str) -> Any:
        try:
            args = json.loads(arguments)
            note_content = args["note_content"]
            processed_template = args["processed_template"]
            examples = json.loads(args["examples"])
            analysis = json.loads(args["analysis"])
            
            restructured_content = self._restructure_with_ai(note_content, processed_template, examples, analysis)
            return {"success": True, "restructured_content": restructured_content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _restructure_with_ai(self, note_content: str, processed_template: str, examples: List[Dict], analysis: Dict) -> str:
        """Use OpenAI to restructure the note"""
        # Prepare examples text
        examples_text = ""
        for i, example in enumerate(examples[:2], 1):
            examples_text += f"\n--- Example {i}: {example['filename']} ---\n{example['content'][:2000]}...\n"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """
                        You are an expert knowledge management specialist and Obsidian notes restructuring expert.
                        Your primary goal is to transform existing notes into well-structured, template-compliant documents 
                        while preserving ALL original content and enhancing its organization and discoverability.

                        ## Core Principles:
                        1. **Content Preservation**: Never delete or lose any original content - reorganize and enhance it
                        2. **Template Compliance**: Follow the provided template structure exactly
                        3. **Intelligent Enhancement**: Add relevant metadata, tags, and cross-references based on content analysis
                        4. **Quality Improvement**: Improve readability, structure, and knowledge connections

                        CRITICAL: The template dates have been pre-filled with actual file dates. Use these exactly.
                        Return ONLY the restructured markdown content without any explanations or wrapper text.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                        ## Task: Restructure Obsidian Note
                        
                        Transform the following note to match the template structure while preserving all content.
                        
                        ### Current Note to Restructure:
                        ```markdown
                        {note_content}
                        ```
                        
                        ### Template Structure (with pre-filled dates):
                        ```markdown
                        {processed_template}
                        ```
                        
                        ### Reference Examples:
                        {examples_text}
                        
                        ### Analysis Report:
                        - **Missing sections**: {analysis['missing_sections']}
                        - **Current compliance score**: {analysis['compliance_score']:.2f}
                        - **Current sections found**: {analysis['current_sections']}
                        - **Template sections required**: {analysis['template_sections']}
                        
                        **Output the complete restructured note in markdown format only. No explanations or wrapper text.**
                    """
                }
            ]
        )

        generated_text = response.choices[0].message.content.strip()
        
        # Strip markdown code block markers if present
        if generated_text.strip().startswith("```markdown"):
            lines = generated_text.strip().splitlines()
            if len(lines) > 2 and lines[-1].strip() == "```":
                generated_text = "\n".join(lines[1:-1])

        return generated_text


class BackupTool(Tool):
    """Tool for creating file backups"""
    
    def __init__(self):
        super().__init__(
            name="create_backup",
            description="Create a backup of a file before modification",
            parameters={
                "file_path": {"type": "string", "description": "Path to the file to backup"},
                "content": {"type": "string", "description": "Content to backup"}
            }
        )

    def execute(self, arguments: str) -> Any:
        try:
            args = json.loads(arguments)
            file_path = args["file_path"]
            content = args["content"]
            
            backup_path = file_path + ".backup"
            with open(backup_path, 'w', encoding='utf-8') as file:
                file.write(content)
            
            return {"success": True, "backup_path": backup_path}
        except Exception as e:
            return {"success": False, "error": str(e)}


# --- AGENT BASE CLASS ---

class Agent:
    """Base class for all agents"""
    
    def __init__(self, name: str, role: str, model: str = "gpt-4o-mini"):
        self.name = name
        self.role = role
        self.client = OpenAI()
        self.model = model
        self.tools: Dict[str, Tool] = {}
        self.messages: List[Dict[str, Any]] = []

    def register_tool(self, tool: Tool):
        """Register a tool with the agent"""
        self.tools[tool.name] = tool

    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for OpenAI"""
        return [tool.get_schema() for tool in self.tools.values()]

    def execute_tool_call(self, tool_call: Any) -> str:
        """Execute a tool call"""
        fn_name = tool_call.name
        fn_args = tool_call.arguments

        if fn_name in self.tools:
            tool_to_call = self.tools[fn_name]
            try:
                result = tool_to_call.execute(fn_args)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": f"Error calling {fn_name}: {e}"})

        return json.dumps({"error": f"Unknown tool: {fn_name}"})

    def process_request(self, request: str) -> Any:
        """Process a request and return the result"""
        raise NotImplementedError("Each agent must implement process_request")


# --- AGENT IMPLEMENTATIONS ---

class VaultDiscoveryAgent(Agent):
    """Agent responsible for discovering and validating vault contents"""
    
    def __init__(self):
        super().__init__("VaultDiscovery", "Discover and validate markdown files in Obsidian vault")
        self.register_tool(FileSystemTool())

    def process_request(self, vault_path: str) -> Dict[str, Any]:
        """Discover and validate vault contents"""
        print(f"[{self.name}] Discovering vault contents at: {vault_path}")
        
        # Validate vault
        validation_result = json.loads(self.execute_tool_call(
            type('obj', (object,), {
                'name': 'file_system_operation',
                'arguments': json.dumps({"operation": "validate_vault", "path": vault_path})
            })
        ))
        
        if not validation_result.get("success") or not validation_result.get("valid"):
            return {"success": False, "error": "Invalid vault path"}
        
        # Find markdown files
        files_result = json.loads(self.execute_tool_call(
            type('obj', (object,), {
                'name': 'file_system_operation',
                'arguments': json.dumps({"operation": "find_markdown_files", "path": vault_path})
            })
        ))
        
        if not files_result.get("success"):
            return {"success": False, "error": "Failed to find markdown files"}
        
        md_files = files_result.get("files", [])
        print(f"[{self.name}] Found {len(md_files)} markdown files")
        
        return {
            "success": True,
            "vault_path": vault_path,
            "markdown_files": md_files,
            "file_count": len(md_files)
        }


class NoteAnalysisAgent(Agent):
    """Agent responsible for analyzing note structure and compliance"""
    
    def __init__(self):
        super().__init__("NoteAnalysis", "Analyze note structure against template")
        self.register_tool(FileSystemTool())
        self.register_tool(NoteAnalysisTool())

    def process_request(self, file_path: str, template_content: str) -> Dict[str, Any]:
        """Analyze a single note file"""
        print(f"[{self.name}] Analyzing: {os.path.basename(file_path)}")
        
        # Read note content
        read_result = json.loads(self.execute_tool_call(
            type('obj', (object,), {
                'name': 'file_system_operation',
                'arguments': json.dumps({"operation": "read_file", "path": file_path})
            })
        ))
        
        if not read_result.get("success"):
            return {"success": False, "error": f"Failed to read file: {file_path}"}
        
        note_content = read_result.get("content", "")
        
        # Analyze structure
        analysis_result = json.loads(self.execute_tool_call(
            type('obj', (object,), {
                'name': 'analyze_note_structure',
                'arguments': json.dumps({
                    "note_content": note_content,
                    "template_content": template_content
                })
            })
        ))
        
        if not analysis_result.get("success"):
            return {"success": False, "error": "Failed to analyze note structure"}
        
        analysis = analysis_result.get("analysis", {})
        
        print(f"[{self.name}] Compliance score: {analysis.get('compliance_score', 0):.2f}")
        print(f"[{self.name}] Needs restructuring: {analysis.get('needs_restructuring', False)}")
        
        return {
            "success": True,
            "file_path": file_path,
            "note_content": note_content,
            "analysis": analysis
        }


class ContentRestructuringAgent(Agent):
    """Agent responsible for AI-powered content restructuring"""
    
    def __init__(self):
        super().__init__("ContentRestructuring", "Restructure notes using AI")
        self.register_tool(TemplateTool())
        self.register_tool(AIRestructuringTool())
        self.register_tool(BackupTool())
        self.register_tool(FileSystemTool())

    def process_request(self, note_data: Dict[str, Any], template_content: str, examples: List[Dict]) -> Dict[str, Any]:
        """Restructure a note using AI"""
        file_path = note_data["file_path"]
        note_content = note_data["note_content"]
        analysis = note_data["analysis"]
        
        print(f"[{self.name}] Restructuring: {os.path.basename(file_path)}")
        
        # Process template with file-specific data
        template_result = json.loads(self.execute_tool_call(
            type('obj', (object,), {
                'name': 'process_template',
                'arguments': json.dumps({
                    "template_content": template_content,
                    "file_path": file_path
                })
            })
        ))
        
        if not template_result.get("success"):
            return {"success": False, "error": "Failed to process template"}
        
        processed_template = template_result.get("processed_content", "")
        
        # Create backup
        backup_result = json.loads(self.execute_tool_call(
            type('obj', (object,), {
                'name': 'create_backup',
                'arguments': json.dumps({
                    "file_path": file_path,
                    "content": note_content
                })
            })
        ))
        
        backup_path = backup_result.get("backup_path") if backup_result.get("success") else None
        
        # Restructure with AI
        restructure_result = json.loads(self.execute_tool_call(
            type('obj', (object,), {
                'name': 'restructure_note_ai',
                'arguments': json.dumps({
                    "note_content": note_content,
                    "processed_template": processed_template,
                    "examples": json.dumps(examples),
                    "analysis": json.dumps(analysis)
                })
            })
        ))
        
        if not restructure_result.get("success"):
            return {"success": False, "error": "Failed to restructure note"}
        
        restructured_content = restructure_result.get("restructured_content", "")
        
        # Write restructured content
        write_result = json.loads(self.execute_tool_call(
            type('obj', (object,), {
                'name': 'file_system_operation',
                'arguments': json.dumps({
                    "operation": "write_file",
                    "path": file_path,
                    "content": restructured_content
                })
            })
        ))
        
        if not write_result.get("success"):
            return {"success": False, "error": "Failed to write restructured content"}
        
        print(f"[{self.name}] Successfully restructured note")
        
        return {
            "success": True,
            "file_path": file_path,
            "backup_path": backup_path,
            "action": "restructured"
        }


class RefactoringOrchestratorAgent(Agent):
    """Main orchestrator agent that coordinates the entire refactoring workflow"""
    
    def __init__(self):
        super().__init__("RefactoringOrchestrator", "Coordinate entire vault refactoring workflow")
        
        # Initialize sub-agents
        self.vault_discovery = VaultDiscoveryAgent()
        self.note_analysis = NoteAnalysisAgent()
        self.content_restructuring = ContentRestructuringAgent()

    def load_template_and_examples(self, template_path: str, examples_path: str) -> Tuple[str, List[Dict]]:
        """Load template and example files"""
        print(f"[{self.name}] Loading template: {template_path}")
        
        # Validate template path exists
        if not os.path.exists(template_path):
            raise Exception(f"Template file not found: {template_path}")
        
        # Load template
        fs_tool = FileSystemTool()
        template_result = fs_tool.execute(json.dumps({
            "operation": "read_file",
            "path": template_path
        }))
        
        if not template_result.get("success"):
            raise Exception(f"Failed to load template: {template_path}")
        
        template_content = template_result.get("content", "")
        
        # Load examples
        examples = []
        if os.path.exists(examples_path) and os.path.isdir(examples_path):
            print(f"[{self.name}] Loading examples from: {examples_path}")
            for filename in os.listdir(examples_path):
                if filename.lower().endswith(('.md', '.markdown')):
                    file_path = os.path.join(examples_path, filename)
                    example_result = fs_tool.execute(json.dumps({
                        "operation": "read_file",
                        "path": file_path
                    }))
                    
                    if example_result.get("success"):
                        examples.append({
                            'filename': filename,
                            'content': example_result.get("content", "")
                        })
        else:
            print(f"[{self.name}] Warning: Examples directory not found or empty: {examples_path}")
        
        print(f"[{self.name}] Loaded template and {len(examples)} examples")
        return template_content, examples

    def process_request(self, vault_path: str, template_path: str = DEFAULT_TEMPLATE_PATH, 
                       examples_path: str = DEFAULT_EXAMPLES_PATH, dry_run: bool = False) -> Dict[str, Any]:
        """Orchestrate the complete vault refactoring workflow"""
        print(f"[{self.name}] Starting vault refactoring workflow")
        print(f"[{self.name}] Vault: {vault_path}")
        print(f"[{self.name}] Template: {template_path}")
        print(f"[{self.name}] Examples: {examples_path}")
        print(f"[{self.name}] Dry run: {dry_run}")
        print("-" * 60)
        
        try:
            # Step 1: Load template and examples
            template_content, examples = self.load_template_and_examples(template_path, examples_path)
            
            # Step 2: Discover vault contents
            discovery_result = self.vault_discovery.process_request(vault_path)
            if not discovery_result.get("success"):
                return {"success": False, "error": discovery_result.get("error")}
            
            markdown_files = discovery_result.get("markdown_files", [])
            
            # Step 3: Process each file
            processed_count = 0
            skipped_count = 0
            error_count = 0
            results = []
            
            for file_path in markdown_files:
                relative_path = os.path.relpath(file_path, vault_path)
                print(f"\n[{self.name}] Processing: {relative_path}")
                
                # Analyze note
                analysis_result = self.note_analysis.process_request(file_path, template_content)
                if not analysis_result.get("success"):
                    print(f"[{self.name}] ✗ Analysis failed: {analysis_result.get('error')}")
                    error_count += 1
                    continue
                
                analysis = analysis_result.get("analysis", {})
                needs_restructuring = analysis.get("needs_restructuring", False)
                
                if not needs_restructuring:
                    print(f"[{self.name}] ✓ Note is already well-structured, skipping")
                    skipped_count += 1
                    results.append({
                        "file_path": file_path,
                        "action": "skipped",
                        "reason": "already well-structured"
                    })
                    continue
                
                if dry_run:
                    print(f"[{self.name}] → Would restructure this note (dry run mode)")
                    processed_count += 1
                    results.append({
                        "file_path": file_path,
                        "action": "would_restructure",
                        "reason": "dry run mode"
                    })
                    continue
                
                # Restructure note
                restructure_result = self.content_restructuring.process_request(
                    analysis_result, template_content, examples
                )
                
                if restructure_result.get("success"):
                    print(f"[{self.name}] ✓ Successfully restructured")
                    processed_count += 1
                    results.append(restructure_result)
                else:
                    print(f"[{self.name}] ✗ Restructuring failed: {restructure_result.get('error')}")
                    error_count += 1
                    results.append({
                        "file_path": file_path,
                        "action": "failed",
                        "error": restructure_result.get("error")
                    })
            
            # Summary
            print(f"\n" + "=" * 60)
            print(f"[{self.name}] Vault refactor completed!")
            print(f"[{self.name}] Processed: {processed_count} notes")
            print(f"[{self.name}] Skipped: {skipped_count} notes (already well-structured)")
            print(f"[{self.name}] Errors: {error_count} notes")
            print(f"[{self.name}] Total files: {len(markdown_files)}")
            
            if not dry_run and processed_count > 0:
                print(f"[{self.name}] Backup files created for modified notes (.backup extension)")
            
            return {
                "success": True,
                "summary": {
                    "total_files": len(markdown_files),
                    "processed": processed_count,
                    "skipped": skipped_count,
                    "errors": error_count
                },
                "results": results
            }
            
        except Exception as e:
            print(f"[{self.name}] ✗ Workflow failed: {str(e)}")
            return {"success": False, "error": str(e)}


# --- MAIN ENTRY POINT ---

def main():
    """Main entry point for the multi-agent Obsidian refactor tool"""
    
    # Validate OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Multi-Agent Obsidian Vault Refactor Tool")
        print("=" * 50)
        print("Usage: python demo_obsidian_refactor_ai_multi_agent.py <obsidian_vault_path> [--dry-run] [--template=path] [--examples=path]")
        print("  obsidian_vault_path: Path to the Obsidian vault to refactor")
        print("  --dry-run: Analyze notes without making changes")
        print("  --template=path: Path to template file (default: ./template/note-template.md)")
        print("  --examples=path: Path to examples directory (default: ./examples/)")
        print("\nExample:")
        print("  python demo_obsidian_refactor_ai_multi_agent.py /path/to/my/vault --dry-run")
        print("  python demo_obsidian_refactor_ai_multi_agent.py /path/to/my/vault --template=./my_template.md")
        sys.exit(1)

    vault_path = sys.argv[1]
    
    # Validate vault path exists
    if not os.path.exists(vault_path):
        print(f"❌ Error: Vault path does not exist: {vault_path}")
        sys.exit(1)
    
    if not os.path.isdir(vault_path):
        print(f"❌ Error: Vault path is not a directory: {vault_path}")
        sys.exit(1)
    
    # Parse command line arguments
    dry_run = "--dry-run" in sys.argv
    
    template_path = DEFAULT_TEMPLATE_PATH
    examples_path = DEFAULT_EXAMPLES_PATH
    
    for arg in sys.argv[2:]:
        if arg.startswith("--template="):
            template_path = arg.split("=", 1)[1]
        elif arg.startswith("--examples="):
            examples_path = arg.split("=", 1)[1]
    
    print("Multi-Agent Obsidian Vault Refactor Tool")
    print("=" * 50)
    print(f"Vault path: {vault_path}")
    print(f"Template: {template_path}")
    print(f"Examples: {examples_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE REFACTOR'}")
    print("=" * 50)
    
    # Initialize and run the orchestrator
    orchestrator = RefactoringOrchestratorAgent()
    result = orchestrator.process_request(
        vault_path=vault_path,
        template_path=template_path,
        examples_path=examples_path,
        dry_run=dry_run
    )
    
    if result.get("success"):
        print(f"\n🎉 Multi-agent workflow completed successfully!")
        summary = result.get("summary", {})
        print(f"📊 Final Summary:")
        print(f"   Total files: {summary.get('total_files', 0)}")
        print(f"   Processed: {summary.get('processed', 0)}")
        print(f"   Skipped: {summary.get('skipped', 0)}")
        print(f"   Errors: {summary.get('errors', 0)}")
    else:
        print(f"\n❌ Workflow failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()