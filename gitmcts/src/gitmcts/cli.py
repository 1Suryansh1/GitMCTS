"""CLI Entry Point - GitMCTS command line interface."""

import os
import re
import click
import sys
from pathlib import Path
from .chains import mcts_search, chain10_commit_gate, merge_winner, GitVFS, SearchTree
from .ui import BranchTreeDisplay
from rich.live import Live


def fetch_issue_text(url: str) -> str:
    """Fetch GitHub issue text from URL."""
    try:
        import urllib.request
        import json
        match = re.match(r"https://github.com/([^/]+)/([^/]+)/issues/(\d+)", url)
        if not match:
            return url
        owner, repo, num = match.groups()
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{num}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "GitMCTS"})
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read().decode())
        return f"Title: {data['title']}\n\n{data['body']}"
    except Exception:
        return url


@click.command()
@click.option("--issue", "-i", required=True, help="GitHub issue URL or text description")
@click.option("--repo", "-r", default=".", help="Path to git repository")
@click.option("--budget", default=20, help="Number of MCTS simulations")
@click.option("--max-depth", default=5, help="Maximum tree depth")
@click.option("--exploration", "-C", default=1.0, help="UCT exploration constant")
def fix(issue: str, repo: str, budget: int, max_depth: int, exploration: float):
    """
    GitMCTS: Fix a GitHub issue using MCTS tree search.

    Uses the 11-chain MCTS system:
    - Chain 0: Bootstrap (Opus)
    - Chain 1: Static Prune (Python)
    - Chain 2: UCT Select (Python math)
    - Chain 3: Skill Expansion (Sonnet)
    - Chain 4: Execution (branch, execute, commit)
    - Chain 5: Fast Eval + Oracle (combined)
    - Chain 7: Backprop (Python)
    - Chain 8: Terminal Check (Python, kill LLM)
    - Chain 10: Commit Gate (human approval)
    """
    repo_path = Path(repo).resolve()

    if not (repo_path / ".git").exists():
        click.echo(f"Error: {repo} is not a git repository")
        return

    goal = fetch_issue_text(issue) if issue.startswith("http") else issue

    click.echo(f"\n[*] GitMCTS starting")
    click.echo(f"   Goal: {goal[:60]}...")
    click.echo(f"   Budget: {budget} simulations")
    click.echo(f"   Max depth: {max_depth}\n")

    if not os.environ.get("GEMINI_API_KEY"):
        click.echo("Note: GEMINI_API_KEY not set. Using fallback heuristics.\n")

    # Run MCTS search
    config = {
        "simulation_budget": budget,
        "max_depth": max_depth,
        "C": exploration
    }

    # Create display for FIX 5 (Rich Live UI)
    display = BranchTreeDisplay()
    display.set_goal(goal)

    result = mcts_search(goal, str(repo_path), config, display=display)

    if not result.get("success"):
        click.echo(f"\n[X] Search failed: {result.get('error', 'unknown')}")
        return

    # Chain 10: Commit Gate
    vfs = GitVFS(str(repo_path))
    winner_id = result["winner_node_id"]

    click.echo(f"\n[OK] Found solution: {winner_id[:8]} (Q={result['best_Q']:.3f})")

    approved = chain10_commit_gate(winner_id, vfs, skill_used=None, reversibility=None, Q=result['best_Q'])

    if not approved:
        click.echo("\n[X] Commit blocked by user. Exiting.")
        return

    # Merge winner
    commit_message = f"GitMCTS: solution via MCTS | Q={result['best_Q']:.3f} | {result['simulations']} sims"
    commit_hash = merge_winner(vfs, winner_id, commit_message)

    click.echo(f"""
[*] WINNER
   Node: {winner_id[:8]}
   Q: {result['best_Q']:.3f}
   Simulations: {result['simulations']}
   Commit: {commit_hash[:8]}

   git log --oneline -3 to see your fix.
""")


@click.command()
@click.option("--issue", "-i", required=True, help="GitHub issue URL or text description")
@click.option("--repo", "-r", default=".", help="Path to git repository")
@click.option("--branches", "-b", default=3, help="Number of parallel branches (legacy mode)")
@click.option("--model", "-m", default="claude-sonnet-4-6", help="Claude model")
@click.option("--auto-approve", is_flag=True, help="Auto-approve I-class actions")
def fix_parallel(issue: str, repo: str, branches: int, model: str, auto_approve: bool):
    """Legacy parallel branch mode (from original spec)."""
    click.echo("Running in parallel branch mode...")
    click.echo("Use 'gitmcts fix' for the new MCTS tree search mode.")


@click.group()
def cli():
    """GitMCTS - MCTS code agent using Git as world-state substrate."""
    pass


cli.add_command(fix)
cli.add_command(fix_parallel, name="fix-parallel")


if __name__ == "__main__":
    cli()