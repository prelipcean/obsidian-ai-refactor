import sys
import os
import json
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime

# --- SETUP: Load environment and OpenAI client ---
# These lines load your API keys and set up the AI model for use.
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (for API keys, etc.)
load_dotenv()

# Initialize OpenAI client (for AI-powered restructuring)
client = OpenAI()

# Default paths for template and examples
DEFAULT_TEMPLATE_PATH = "./template/note-template.md"
DEFAULT_EXAMPLES_PATH = "./examples/"


def load_file(path: str) -> str:
    """
    Loads the contents of a file as a string. Exits if the file does not exist.
    """
    if not os.path.exists(path):
        print(f"Error: The file '{path}' does not exist.")
        sys.exit(1)

    print("Loading file:", path)
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


def get_file_creation_date(file_path: str) -> str:
    """
    Returns the file's creation date (YYYY-MM-DD).
    On Windows: uses the earlier of creation or modification time (best effort).
    On Unix: uses birth time if available, else modification time.
    """
    try:
        if os.name == 'nt':  # Windows
            # Use pathlib.Path.stat() to get file times
            path_obj = Path(file_path)
            stat_result = path_obj.stat()
            # On Windows, ctime is not always true creation, so we use the earlier of ctime/mtime for best accuracy
            ctime = stat_result.st_ctime  # This is often metadata change time
            mtime = stat_result.st_mtime  # This is modification time
            print(f"    Windows ctime (metadata change): {datetime.fromtimestamp(ctime)}")
            print(f"    Windows mtime (modification): {datetime.fromtimestamp(mtime)}")
            # Use the earlier of the two as the true creation time (handles file copies/metadata changes)
            creation_time = min(ctime, mtime)
            print(f"    Using earlier time as creation: {datetime.fromtimestamp(creation_time)}")
                
        else:  # Unix-like systems
            stat = os.stat(file_path)
            # Try to get birth time (creation time) if available
            if hasattr(stat, 'st_birthtime'):
                creation_time = stat.st_birthtime
                print(f"    Unix birth time: {datetime.fromtimestamp(creation_time)}")
            else:
                # Fall back to modification time if birth time not available
                creation_time = stat.st_mtime
                print(f"    Unix fallback to mtime: {datetime.fromtimestamp(creation_time)}")
        
        result = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d')
        print(f"    Final creation date: {result}")
        return result
    except Exception as e:
        print(f"Warning: Could not get creation date for {file_path}: {e}")
        return datetime.now().strftime('%Y-%m-%d')


def get_file_modification_date(file_path: str) -> str:
    """
    Returns the file's last modification date (YYYY-MM-DD).
    """
    try:
        modification_time = os.path.getmtime(file_path)
        result = datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d')
        print(f"    Modification time: {datetime.fromtimestamp(modification_time)}")
        print(f"    Final modification date: {result}")
        return result
    except Exception as e:
        print(f"Warning: Could not get modification date for {file_path}: {e}")
        return datetime.now().strftime('%Y-%m-%d')


def save_file(path: str, content: str) -> None:
    """
    Saves a string to a file (overwrites if exists).
    """
    print("Saving file:", path)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)


def process_template_placeholders(template_content: str, file_path: str) -> str:
    """
    Replaces template placeholders (like {{date:YYYY-MM-DD}}) with actual file dates and title.
    Ensures 'created' is the file's creation date, 'updated' is today.
    """
    processed_content = template_content
    
    # Get file dates
    creation_date = get_file_creation_date(file_path)
    modification_date = get_file_modification_date(file_path)
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"  File dates summary:")
    print(f"    Created: {creation_date} (from file creation time)")
    print(f"    Modified: {modification_date} (original file modification)")
    print(f"    Updated: {current_date} (script run date - when note is restructured)")
    
    # Logical validation - just informational now since updated = current date
    try:
        creation_dt = datetime.strptime(creation_date, '%Y-%m-%d')
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        if creation_dt <= current_dt:
            print(f"    ✅ Dates look logical: created <= updated (script run date)")
        else:
            print(f"    ⚠️  Creation date is in the future relative to today")
    except Exception as e:
        print(f"    Could not validate dates: {e}")
    
    # Replace date placeholders in frontmatter - need to handle created vs updated differently
    lines = processed_content.split('\n')
    for i, line in enumerate(lines):
        original_line = line
        # Replace 'created' with file creation date, 'updated' with today's date for clarity in note history
        if line.strip().startswith('created:') and '{{date:YYYY-MM-DD}}' in line:
            lines[i] = line.replace('{{date:YYYY-MM-DD}}', creation_date)
            print(f"  ✅ CREATED field: '{original_line.strip()}' → '{lines[i].strip()}'")
        elif line.strip().startswith('updated:') and '{{date:YYYY-MM-DD}}' in line:
            lines[i] = line.replace('{{date:YYYY-MM-DD}}', current_date)
            print(f"  ✅ UPDATED field: '{original_line.strip()}' → '{lines[i].strip()}'")
        # Any other date placeholders default to creation date
        else:
            if '{{date:YYYY-MM-DD}}' in line:
                lines[i] = line.replace('{{date:YYYY-MM-DD}}', creation_date)
                print(f"  ℹ️  Other date field: '{original_line.strip()}' → '{lines[i].strip()}'")
    
    processed_content = '\n'.join(lines)
    
    # Replace other date format placeholders
    processed_content = processed_content.replace('{{date:DD-MM-YYYY}}', 
                                                 datetime.strptime(creation_date, '%Y-%m-%d').strftime('%d-%m-%Y'))
    
    # Handle other common date formats
    processed_content = processed_content.replace('{{created}}', creation_date)
    processed_content = processed_content.replace('{{updated}}', current_date)  # Use current date for updated
    processed_content = processed_content.replace('{{today}}', current_date)
    
    # Extract title from file name for {{title}} placeholder
    file_name = os.path.basename(file_path)
    title = os.path.splitext(file_name)[0].replace('_', ' ').replace('-', ' ').title()
    processed_content = processed_content.replace('{{title}}', title)
    print(f"  ✅ TITLE: '{title}'")
    
    return processed_content


def load_examples(examples_path: str) -> List[Dict[str, str]]:
    """
    Loads all example markdown files from the examples directory.
    Returns a list of dicts with filename and content.
    """
    examples = []
    if not os.path.exists(examples_path):
        print(f"Warning: Examples directory '{examples_path}' does not exist.")
        return examples
    
    for filename in os.listdir(examples_path):
        if filename.lower().endswith(('.md', '.markdown')):
            file_path = os.path.join(examples_path, filename)
            content = load_file(file_path)
            examples.append({
                'filename': filename,
                'content': content
            })
            print(f"Loaded example: {filename}")
    
    print(f"Loaded {len(examples)} example files.")
    return examples


def validate_vault_structure(vault_path: str) -> bool:
    """
    Checks if the provided path is a valid Obsidian vault (has .obsidian or .md files).
    """
    if not os.path.exists(vault_path):
        print(f"Error: Vault path '{vault_path}' does not exist.")
        return False
    
    if not os.path.isdir(vault_path):
        print(f"Error: Vault path '{vault_path}' is not a directory.")
        return False
    
    # Check for common Obsidian vault indicators
    obsidian_dir = os.path.join(vault_path, '.obsidian')
    has_md_files = any(f.endswith('.md') for f in os.listdir(vault_path) if os.path.isfile(os.path.join(vault_path, f)))
    
    if os.path.exists(obsidian_dir) or has_md_files:
        print(f"Valid Obsidian vault detected at: {vault_path}")
        return True
    else:
        print(f"Warning: '{vault_path}' may not be a valid Obsidian vault (no .obsidian directory or .md files found)")
        return True  # Continue anyway, user might have a custom setup


def find_markdown_files(vault_path: str) -> List[str]:
    """
    Recursively finds all markdown files in the vault (ignores .obsidian folder).
    """
    md_files = []
    for root, dirs, files in os.walk(vault_path):
        # Skip .obsidian directory
        if '.obsidian' in dirs:
            dirs.remove('.obsidian')
        
        for file in files:
            if file.lower().endswith(('.md', '.markdown')):
                md_files.append(os.path.join(root, file))
    
    return md_files


def extract_frontmatter_and_content(note_content: str) -> Tuple[str, str]:
    """
    Splits a note into frontmatter (header) and main content.
    Returns (frontmatter, content).
    """
    lines = note_content.strip().splitlines()
    
    if len(lines) < 2 or lines[0].strip() != "---":
        return "", note_content
    
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


def analyze_note_structure(note_content: str, template_content: str) -> Dict[str, any]:
    """
    Compares a note's structure to the template.
    Returns which sections are missing and a compliance score.
    """
    frontmatter, content = extract_frontmatter_and_content(note_content)
    template_frontmatter, template_content = extract_frontmatter_and_content(template_content)
    
    # Extract sections from content
    content_sections = extract_sections_from_content(content)
    template_sections = extract_sections_from_content(template_content)
    
    missing_sections = [section for section in template_sections if section not in content_sections]
    
    analysis = {
        'has_frontmatter': bool(frontmatter),
        'missing_sections': missing_sections,
        'current_sections': content_sections,
        'template_sections': template_sections,
        'compliance_score': calculate_compliance_score(content_sections, template_sections)
    }
    
    return analysis


def extract_sections_from_content(content: str) -> List[str]:
    """
    Finds all section headers (lines starting with #) in markdown content.
    """
    sections = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            # Remove the # characters and clean up
            section = line.lstrip('#').strip()
            if section:
                sections.append(section)
    
    return sections


def calculate_compliance_score(current_sections: List[str], template_sections: List[str]) -> float:
    """
    Returns a score (0.0-1.0) for how well the note matches the template sections.
    """
    if not template_sections:
        return 1.0
    
    matching_sections = len(set(current_sections) & set(template_sections))
    return matching_sections / len(template_sections)


def restructure_note_with_ai(note_content: str, template_content: str, examples: List[Dict[str, str]], 
                           analysis: Dict[str, any], file_path: str) -> str:
    """
    Uses OpenAI to restructure a note based on the template and examples.
    Returns the improved note as a string.
    """
    print("Restructuring note using AI with template and examples...")
    
    # Process template placeholders with actual file dates
    processed_template = process_template_placeholders(template_content, file_path)
    
    # Prepare examples text
    examples_text = ""
    for i, example in enumerate(examples[:2], 1):  # Limit to 2 examples to avoid token limits
        examples_text += f"\n--- Example {i}: {example['filename']} ---\n{example['content'][:2000]}...\n"
    
    frontmatter, content = extract_frontmatter_and_content(note_content)
    template_frontmatter, template_content_only = extract_frontmatter_and_content(processed_template)
    
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

                    ## Restructuring Guidelines:
                    
                    ### Frontmatter Enhancement:
                    - Use the pre-filled creation/update dates exactly as provided
                    - Generate 3-8 relevant tags based on content themes, technologies, concepts
                    - Create 2-4 meaningful aliases (alternative names, abbreviations, related terms)
                    - Add custom fields if they enhance the note's utility
                    
                    ### Content Organization:
                    - Map existing content to appropriate template sections
                    - If content doesn't fit existing sections, create subsections within the most relevant section
                    - Maintain the logical flow and context of information
                    - Preserve code blocks, lists, tables, and formatting exactly
                    
                    ### Section Management:
                    - Ensure all template sections are present
                    - For missing sections, add brief but meaningful placeholders that invite future content
                    - Use subsections (###, ####) to organize complex content within main sections
                    
                    ### Knowledge Enhancement:
                    - Identify potential wiki-links [[]] for concepts mentioned in the text
                    - Suggest related concepts that could be linked
                    - Maintain existing links and references
                    - Add "See also" or "Related" sections if beneficial
                    
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
                    
                    ### Specific Instructions:
                    
                    1. **Content Mapping**: Analyze the current note content and map it to the most appropriate template sections
                    2. **Frontmatter Enhancement**: 
                       - Keep the pre-filled creation/update dates exactly as shown in the template
                       - Generate relevant tags based on the note's content, themes, and subject matter
                       - Create meaningful aliases (abbreviations, alternative names, related terms)
                    3. **Section Population**:
                       - Fill template sections with reorganized content from the original note
                       - If content spans multiple sections, distribute it logically
                       - For empty sections, add brief placeholders that invite future content
                    4. **Structure Preservation**:
                       - Maintain all code blocks, lists, tables, and special formatting
                       - Preserve the technical accuracy and context of information
                       - Keep existing wiki-links and references intact
                    5. **Enhancement Opportunities**:
                       - Identify concepts that could benefit from wiki-links [[concept]]
                       - Improve headings and subheadings for better navigation
                       - Add cross-references where relevant
                    
                    ### Quality Criteria:
                    - ✅ All original content is preserved and properly placed
                    - ✅ Template structure is followed exactly
                    - ✅ Frontmatter includes relevant tags and aliases
                    - ✅ All sections are present (with content or meaningful placeholders)
                    - ✅ Content flows logically and maintains readability
                    - ✅ Technical accuracy and context are preserved
                    
                    **Output the complete restructured note in markdown format only. No explanations or wrapper text.**
                """
            }
        ]
    )

    generated_text = response.choices[0].message.content.strip()
    
    # If the model returns a markdown code block, strip the code block markers
    if generated_text.strip().startswith("```markdown"):
        lines = generated_text.strip().splitlines()
        if len(lines) > 2 and lines[-1].strip() == "```":
            generated_text = "\n".join(lines[1:-1])

    return generated_text


def refactor_obsidian_vault(obsidian_vault_path: str, template_path: str = DEFAULT_TEMPLATE_PATH, 
                          examples_path: str = DEFAULT_EXAMPLES_PATH, dry_run: bool = False) -> None:
    """
    Orchestrates the full workflow: loads template/examples, finds notes, analyzes, and restructures them.
    Backs up originals and prints a summary at the end.
    """
    print(f"Starting Obsidian vault refactor...")
    print(f"Vault path: {obsidian_vault_path}")
    print(f"Template path: {template_path}")
    print(f"Examples path: {examples_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)
    
    # Validate vault
    if not validate_vault_structure(obsidian_vault_path):
        return
    
    # Load template and examples
    template_content = load_file(template_path)
    examples = load_examples(examples_path)
    
    # Find all markdown files in vault
    md_files = find_markdown_files(obsidian_vault_path)
    print(f"Found {len(md_files)} markdown files in vault.")
    
    if not md_files:
        print("No markdown files found in vault.")
        return
    
    # Process each file
    processed_count = 0
    skipped_count = 0
    
    for file_path in md_files:
        relative_path = os.path.relpath(file_path, obsidian_vault_path)
        print(f"\nProcessing: {relative_path}")
        
        # Load and analyze note
        note_content = load_file(file_path)
        analysis = analyze_note_structure(note_content, template_content)
        
        print(f"  Compliance score: {analysis['compliance_score']:.2f}")
        print(f"  Missing sections: {analysis['missing_sections']}")
        
        # Skip if already well-structured
        # If note is already 80%+ compliant and has no missing sections, skip to avoid unnecessary changes
        if analysis['compliance_score'] >= 0.8 and len(analysis['missing_sections']) == 0:
            print(f"  ✓ Note is already well-structured, skipping.")
            skipped_count += 1
            continue
        
        # In dry-run mode, just report what would be changed, don't actually modify files
        if dry_run:
            print(f"  → Would restructure this note (dry run mode)")
            processed_count += 1
            continue
        
        # Restructure the note
        try:
            # Call the AI model to generate a restructured note based on template and examples
            restructured_content = restructure_note_with_ai(note_content, template_content, examples, analysis, file_path)
            # Back up the original note before making changes for safety
            backup_path = file_path + ".backup"
            save_file(backup_path, note_content)
            print(f"  Created backup: {backup_path}")
            # Save the new, improved note content
            save_file(file_path, restructured_content)
            print(f"  ✓ Successfully restructured note")
            processed_count += 1
        except Exception as e:
            print(f"  ✗ Error restructuring note: {str(e)}")
    
    print(f"\n" + "=" * 60)
    print(f"Vault refactor completed!")
    print(f"Processed: {processed_count} notes")
    print(f"Skipped: {skipped_count} notes (already well-structured)")
    print(f"Total files: {len(md_files)}")
    
    if not dry_run and processed_count > 0:
        print(f"\nBackup files created for modified notes (.backup extension)")
        print(f"Review the changes and delete backups when satisfied.")


def main():
    """
    Parses command-line arguments and starts the refactor workflow.
    Shows usage instructions if arguments are missing.
    """
    if len(sys.argv) < 2:
        print("Usage: python 31_obsidian_refactor.py <obsidian_vault_path> [--dry-run] [--template=path] [--examples=path]")
        print("  obsidian_vault_path: Path to the Obsidian vault to refactor")
        print("  --dry-run: Analyze notes without making changes")
        print("  --template=path: Path to template file (default: ./obsidian_refactor/template/Note_Template.md)")
        print("  --examples=path: Path to examples directory (default: ./obsidian_refactor/examples/)")
        print("\nExample:")
        print("  python 31_obsidian_refactor.py /path/to/my/vault --dry-run")
        print("  python 31_obsidian_refactor.py /path/to/my/vault --template=./my_template.md")
        sys.exit(1)

    obsidian_vault_path = sys.argv[1]
    
    # Parse command line arguments
    dry_run = "--dry-run" in sys.argv
    
    template_path = DEFAULT_TEMPLATE_PATH
    examples_path = DEFAULT_EXAMPLES_PATH
    
    for arg in sys.argv[2:]:
        if arg.startswith("--template="):
            template_path = arg.split("=", 1)[1]
        elif arg.startswith("--examples="):
            examples_path = arg.split("=", 1)[1]
    
    print("Obsidian Vault Refactor Tool")
    print("=" * 40)
    print(f"Vault path: {obsidian_vault_path}")
    print(f"Template: {template_path}")
    print(f"Examples: {examples_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE REFACTOR'}")
    print("=" * 40)
    
    # Run the refactor workflow
    refactor_obsidian_vault(
        obsidian_vault_path=obsidian_vault_path,
        template_path=template_path,
        examples_path=examples_path,
        dry_run=dry_run
    )
    
    print("\nWorkflow completed successfully!")


if __name__ == "__main__":
    main()
