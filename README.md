# Enterprise Multi-Agent Copilot

**Enterprise-grade Multi-Agent AI system** for structured planning, retrieval-grounded research, report generation, and automated verification.

Built with **LangGraph, LangChain, Chroma Vector Store, OpenAI, and Streamlit.**

---

## Overview

Enterprise Multi-Agent Copilot is a **Retrieval-Augmented Generation (RAG)** system designed to analyze enterprise documents in a reliable and controlled manner.

The system:

- Creates a structured execution plan  
- Retrieves relevant documents from a vector database  
- Extracts only grounded, cited facts  
- Generates structured, professional reports  
- Verifies that all claims are supported by evidence  
- Automatically retries if unsupported claims are detected  

The architecture enforces **strict grounding rules** to minimize hallucinations and ensure traceable outputs.

---

## System Architecture

The workflow is orchestrated using **LangGraph** and follows a deterministic control flow:

```
Planner → Researcher → Writer → Verifier
            ↑               ↓
        ←—— Retry Loop ——→
```

---

## Agent Responsibilities

### Planner

- Generates a structured execution plan (3–6 steps)  
- Outputs JSON  
- Does not perform retrieval or drafting  

### Researcher

- Queries Chroma vector store  
- Extracts only facts supported by retrieved documents  
- Attaches citations to every fact  
- Returns `"Not found in sources"` if evidence is missing  

### Writer

- Produces a clean, structured Markdown deliverable  
- Uses only the Researcher’s notes  
- Does not introduce external knowledge  
- Explicitly states if research is insufficient  

### Verifier

- Validates that every claim is supported by research notes  
- Detects hallucinated or unsupported content  
- Triggers retry loop when necessary  
- Stops after a configurable maximum retry count  

---

## Project Structure

```
Enterprise-Multi-Agent-Copilot/
│
├── agents/
│   ├── planner.py
│   ├── researcher.py
│   ├── writer.py
│   ├── verifier.py
│   └── graph.py
│
├── schemas/
│   └── state.py
│
├── tools/
│   └── retriever.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── sample_docs/
│
├── eval/
│   ├── run_eval.py
│   └── test_cases.json
│
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the repository

```bash
git clone <your-repository-url>
cd Enterprise-Multi-Agent-Copilot
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## Running the Application

```bash
streamlit run app/streamlit_app.py
```

Then:

- Enter a task or question  
- The system executes the full agent workflow  
- Review the final grounded output with citations  

---
