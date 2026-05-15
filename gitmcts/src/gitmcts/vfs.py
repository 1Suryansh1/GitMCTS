"""GitVFS Layer - Git worktree-based state management for MCTS."""

import git
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Branch:
    """Represents an MCTS search branch (git worktree)."""
    name: str
    node_id: str  # = git commit hash
    worktree: Path
    score: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    is_pruned: bool = False
    prune_reason: str = ""
    depth: int = 1
    status: str = "created"  # created, exploring, scored, pruned, winner


class GitVFS:
    """
    The world-state substrate using Git worktrees.
    Every MCTS node = one Git commit.
    """

    def __init__(self, repo_path: str):
        self.repo = git.Repo(repo_path)
        self.root = Path(repo_path)
        self.worktrees_dir = self.root / ".mcts_worktrees"
        self.worktrees_dir.mkdir(exist_ok=True)
        self.branches: list[Branch] = []
        self.root_commit = self.repo.head.commit.hexsha

    def create_branch(self, branch_name: str) -> Branch:
        """
        git checkout -b {branch_name} — O(1), no data copy.
        git worktree add — isolated filesystem for this branch.
        """
        new_branch = self.repo.create_head(branch_name)
        worktree_path = self.worktrees_dir / branch_name.replace("/", "_")
        self.repo.git.worktree("add", str(worktree_path), branch_name)

        branch = Branch(
            name=branch_name,
            node_id=self.repo.head.commit.hexsha,
            worktree=worktree_path
        )
        self.branches.append(branch)
        return branch

    def commit_branch(self, branch: Branch, message: str, metadata: dict) -> str:
        """
        Stage all changes + commit. Returns new commit hash = node_id.
        The commit hash IS the world-state fingerprint.
        """
        worktree_repo = git.Repo(branch.worktree)

        meta_path = branch.worktree / ".mcts_node.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        worktree_repo.git.add("-A")
        try:
            commit = worktree_repo.index.commit(message)
            branch.node_id = commit.hexsha
            return commit.hexsha
        except git.exc.GitCommandError:
            return branch.node_id

    def rollback(self, branch: Branch):
        """
        O(1). Remove worktree. Branch and commits remain in object store.
        """
        branch.is_pruned = True
        branch.status = "pruned"
        worktree_path = branch.worktree
        try:
            self.repo.git.worktree("remove", str(worktree_path), "--force")
        except Exception:
            pass
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)

    def diff_from_root(self, branch: Branch) -> str:
        """What changed in this branch vs HEAD at search start."""
        try:
            return self.repo.git.diff(self.root_commit, branch.node_id, "--stat")
        except Exception:
            return "(no diff available)"

    def merge_winner(self, winner: Branch, commit_message: str) -> str:
        """
        git merge {winner.name} — the only real-world commit.
        Crosses the VFS → real world boundary.
        """
        main = self.repo.active_branch
        self.repo.git.merge(winner.name, "--no-ff", "-m", commit_message)

        for b in self.branches:
            if not b.is_pruned and b.worktree.exists():
                try:
                    self.repo.git.worktree("remove", str(b.worktree), "--force")
                except Exception:
                    pass

        return self.repo.head.commit.hexsha

    def cleanup(self):
        """Remove all worktrees on abort/failure."""
        for b in self.branches:
            if b.worktree.exists():
                try:
                    self.repo.git.worktree("remove", str(b.worktree), "--force")
                    shutil.rmtree(b.worktree, ignore_errors=True)
                except Exception:
                    pass
            try:
                self.repo.delete_head(b.name, force=True)
            except Exception:
                pass

    def get_branch_status_summary(self) -> dict:
        """Get summary of all branches for UI."""
        return {
            "total": len(self.branches),
            "active": len([b for b in self.branches if not b.is_pruned]),
            "pruned": len([b for b in self.branches if b.is_pruned]),
            "branches": [
                {
                    "name": b.name,
                    "status": b.status,
                    "score": b.score,
                    "tests": f"{b.tests_passed}/{b.tests_total}",
                    "pruned": b.is_pruned,
                    "prune_reason": b.prune_reason
                }
                for b in self.branches
            ]
        }