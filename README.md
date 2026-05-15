# GitMCTS

GitMCTS is an innovative coding agent that leverages Monte Carlo Tree Search (MCTS) directly on top of Git's Merkle DAG. 

Rather than maintaining an abstract, disconnected state tree in memory, GitMCTS views a Git repository as a "Virtual File System" (VFS). Every node in the MCTS search tree corresponds to an actual Git commit, and isolated explorations happen in parallel via `git worktree`.

## Core Features
1. **True VFS State**: Every branch of thought is a real, isolated branch in the filesystem.
2. **LLM Agnostic**: Powered by the Gemma 4-31b-it model.
3. **Rigorous Validation (The Oracle)**: Relies on `pytest` to definitively prove the success of a leaf node.
4. **Safety Gated**: Irreversible (I-class) operations require a strict human-in-the-loop approval before committing to the main branch.

## Installation

You can install this as a package natively:
```bash
pip install .
```

## Setup & Usage

Ensure you have your API key set up (Never commit your API keys!):
```bash
export GEMINI_API_KEY="your-api-key-here"
```

To run:
```bash
gitmcts --help
```