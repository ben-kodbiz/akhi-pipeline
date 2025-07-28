#!/usr/bin/env python3
"""
CrewAI Basic Example with LM Studio Local LLM
Working example for Qwen 3 14B model via LM Studio API

This example demonstrates:
- Local LLM configuration with LM Studio
- Agent definition with local LLM
- Task creation and crew orchestration
- Output generation (Markdown, HTML, PDF)
"""

import markdown
import os
from pathlib import Path
from weasyprint import HTML
from crewai import Agent, Task, Crew, LLM

# 1. Define the LLM connected to LM Studio
local_llm = LLM(
    model="lm_studio/qwen-3-14b",   # Match your LM Studio model name
    base_url="http://192.168.0.74:1234/v1",  # Default LM Studio endpoint
    api_key=None                    # Just needs to be set, not real
)

# 2. Define agents using the local LLM
researcher = Agent(
    role="Senior Researcher",
    goal="Research the latest techniques in QLoRA and LoRA training",
    backstory="You are an experienced ML researcher who stays on top of model training workflows.",
    verbose=True,
    allow_delegation=False,
    llm=local_llm
)

writer = Agent(
    role="Technical Writer",
    goal="Document complex AI workflows in a clear, concise manner",
    backstory="You specialize in writing guides for developers and researchers.",
    verbose=True,
    allow_delegation=False,
    llm=local_llm
)

# 3. Define tasks assigned to agents
task_research = Task(
    description="Summarize the current state-of-the-art methods for training QLoRA or LoRA models.",
    expected_output="A markdown summary with at least 3 advanced techniques used in 2024-2025.",
    agent=researcher
)

task_write = Task(
    description="Turn the research summary into a short tutorial blog post (500 words max) for new ML engineers.",
    expected_output="A structured blog post in markdown with headers and code snippets.",
    agent=writer
)

# 4. Run the Crew with both agents and tasks
crew = Crew(
    agents=[researcher, writer],
    tasks=[task_research, task_write],
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== FINAL RESULT ===\n")
    print(result)
    
    # Convert Markdown to HTML
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save as Markdown
    md_file = output_dir / "crew_result.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(str(result))
    
    print(f"\n✅ Saved markdown to {md_file.resolve()}")
    
    html_content = markdown.markdown(str(result))
    html_file = output_dir / "crew_result.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Convert HTML to PDF
    pdf_file = output_dir / "crew_result.pdf"
    HTML(string=html_content).write_pdf(str(pdf_file))
    
    print(f"📄 Saved PDF to {pdf_file.resolve()}")