"""Terminal UI - Rich animated display for branch tree."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from typing import List, Optional
from .vfs import Branch


class BranchTreeDisplay:
    """Animated terminal display for GitMCTS branch exploration."""

    def __init__(self):
        self.console = Console()
        self.branches: List[Branch] = []
        self.current_branch: Optional[str] = None
        self.goal: str = ""

    def set_goal(self, goal: str):
        """Set the search goal."""
        self.goal = goal[:60] + "..." if len(goal) > 60 else goal

    def add_branch(self, branch: Branch):
        """Add a new branch to track."""
        self.branches.append(branch)

    def set_exploring(self, branch_name: str):
        """Mark branch as currently exploring."""
        for b in self.branches:
            if b.name == branch_name:
                b.status = "exploring"
        self.current_branch = branch_name

    def set_scored(self, branch_name: str, test_result):
        """Mark branch as scored with test results."""
        for b in self.branches:
            if b.name == branch_name:
                b.status = "scored"
                b.score = test_result.score
                b.tests_passed = test_result.passed
                b.tests_total = test_result.passed + test_result.failed + test_result.errors

    def set_pruned(self, branch_name: str, reason: str):
        """Mark branch as pruned."""
        for b in self.branches:
            if b.name == branch_name:
                b.status = "pruned"
                b.is_pruned = True
                b.prune_reason = reason

    def set_winner(self, branch_name: str):
        """Mark branch as winner."""
        for b in self.branches:
            if b.name == branch_name:
                b.status = "winner"

    def render(self) -> Panel:
        """Render the branch tree display."""
        lines = []

        header = f"[*] GitMCTS | goal: {self.goal}"
        lines.append(header)
        lines.append("")

        active_branches = [b for b in self.branches if not b.is_pruned]
        pruned_branches = [b for b in self.branches if b.is_pruned]

        for b in active_branches:
            status_icon = self._get_status_icon(b.status)
            if b.status == "exploring":
                line = f"├── {status_icon} {b.name} [exploring...]"
            elif b.status == "scored":
                score_pct = b.score * 100
                line = f"├── {status_icon} {b.name} tests: {b.tests_passed}/{b.tests_total} Q={b.score:.2f}"
            elif b.status == "winner":
                line = f"[*] {b.name} [WINNER] tests: {b.tests_passed}/{b.tests_total}"
            else:
                line = f"├── {status_icon} {b.name}"
            lines.append(line)

        for b in pruned_branches:
            line = f"--- [X] {b.name} pruned: {b.prune_reason}"
            lines.append(line)

        if not self.branches:
            lines.append("  (no branches yet)")

        return Panel(
            "\n".join(lines),
            title="GitMCTS Branch Explorer",
            border_style="cyan"
        )

    def _get_status_icon(self, status: str) -> str:
        icons = {
            "created": "[ ]",
            "exploring": "[>]",
            "scored": "[OK]",
            "pruned": "[X]",
            "winner": "[*]"
        }
        return icons.get(status, "?")

    def print_summary(self, winner: Branch, cost: float, rollbacks: int):
        """Print final summary."""
        self.console.print(f"""
[green]⬡  WINNER {winner.name}[/green]
   [yellow]Score: {winner.score:.2f} | Tests: {winner.tests_passed}/{winner.tests_total}[/yellow]
   [cyan]rollbacks: {rollbacks} | branches: {len(self.branches)} | cost: ${cost:.4f}[/cyan]
""")

    def print_status(self, message: str, style: str = "bold"):
        """Print a status message."""
        self.console.print(f"[{style}]{message}[/{style}]")


class LiveDisplay:
    """Live updating display using Rich."""

    def __init__(self):
        self.console = Console()
        self.progress = None

    def create_progress(self):
        """Create a progress bar."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )
        return self.progress

    def print_panel(self, title: str, content: str, style: str = "cyan"):
        """Print a panel with content."""
        self.console.print(Panel(content, title=title, border_style=style))