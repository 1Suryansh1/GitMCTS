"""MCTS Node and Tree data structures - matching PART 1 of spec."""

import json
import time
import math
import random
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# ── CORE NODE SCHEMA ──────────────────────────────────────────────────────────────
@dataclass
class MCTSNode:
    """A single node in the MCTS tree = a Git commit."""
    node_id: str  # = git commit hash (sha256, 40 chars)
    parent_id: Optional[str]  # parent commit hash; None = root
    depth: int

    # Goal
    goal_slice: str  # 1-sentence goal for this node's scope
    goal_hash: str  # hash of root goal — used for drift detection

    # Search state - Multi-objective Q (matching spec)
    Q: dict[str, float] = field(default_factory=lambda: {
        "quality": 0.0,
        "cost": 1.0,
        "latency": 1.0,
        "safety": 1.0
    })
    N: int = 0  # visit count
    virtual_loss: float = 0.0
    variance: float = 0.0

    # Git substrate
    git_branch: str = ""
    git_worktree: str = ""

    # Execution
    skill_used: Optional[str] = None
    skill_params: Optional[dict] = None
    reversibility: Optional[str] = None  # "R", "W", or "I"
    execution_output: Optional[dict] = None

    # Evaluation
    fast_score: float = 0.0
    deep_score: float = -1.0  # -1 = not yet run
    is_terminal: bool = False
    terminal_prob: float = 0.0
    is_pruned: bool = False
    pruned_reason: str = ""

    # Context State
    context_buffer: dict = field(default_factory=lambda: {"last_read_content": "", "last_read_file": ""})

    # Children
    children: list = field(default_factory=list)

    # Timestamps
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# ── SKILL REGISTRY ENTRY ─────────────────────────────────────────────────────────
@dataclass
class SkillEntry:
    """An MCP tool / skill that can be executed."""
    skill_id: str
    name: str
    description: str

    # R/W/I classification
    reversibility: str  # "R", "W", or "I"
    reversibility_rationale: str = ""

    # Preconditions (static pruning)
    requires_files: list = field(default_factory=list)
    requires_state_keys: list = field(default_factory=list)
    forbidden_after: list = field(default_factory=list)

    # Cost model
    cost_estimate: float = 0.5  # normalized 0–1
    latency_estimate_ms: float = 1000.0

    # Output schema
    output_schema: dict = field(default_factory=dict)

    # L3 stats (updated async after each run)
    l3_success_rate: float = 0.5
    l3_avg_q_when_used: float = 0.5


# ── SEARCH TREE (global state) ───────────────────────────────────────────────────
@dataclass
class SearchTree:
    """The complete MCTS search tree - matching spec."""
    tree_id: str = ""  # = session ID
    root_goal: str = ""
    root_node_id: str = ""  # git hash of root commit
    nodes: dict = field(default_factory=dict)

    # Budget
    simulation_budget: int = 20
    simulations_used: int = 0
    cost_budget_usd: float = 10.0
    cost_used_usd: float = 0.0

    # Meta-controller state
    adaptive_C: float = 1.0  # current exploration constant
    domain_familiarity: int = 0  # n past episodes in this goal class (from L3)

    # Best known
    best_node_id: Optional[str] = None
    best_Q_quality: float = 0.0

    # Async state
    active_async_nodes: list = field(default_factory=list)

    # State
    is_terminal: bool = False
    i_class_allowed: bool = True

    def add_node(self, node: MCTSNode):
        """Add a node to the tree."""
        self.nodes[node.node_id] = node
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node.node_id)

    def get_node(self, node_id: str) -> Optional[MCTSNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_best_node(self) -> Optional[MCTSNode]:
        """Get the node with highest Q."""
        if not self.best_node_id:
            return None
        return self.nodes.get(self.best_node_id)

    def update_best(self, node_id: str, Q: float):
        """Update best known node."""
        if Q > self.best_Q_quality:
            self.best_Q_quality = Q
            self.best_node_id = node_id


# ── SKILL REGISTRY ───────────────────────────────────────────────────────────────
class SkillRegistry:
    """Registry of available MCP skills."""

    def __init__(self):
        self.skills: dict[str, SkillEntry] = {}
        self._register_default_skills()

    def _register_default_skills(self):
        """Register the 3 core MCP tools from spec."""
        self.skills["read_file"] = SkillEntry(
            skill_id="read_file",
            name="Read File",
            description="Read contents of a file",
            reversibility="R",
            reversibility_rationale="Reading is reversible - just read the file",
            requires_files=[],
            requires_state_keys=[],
            cost_estimate=0.1,
            latency_estimate_ms=100.0
        )
        self.skills["edit_file"] = SkillEntry(
            skill_id="edit_file",
            name="Edit File",
            description="Edit a file with targeted changes",
            reversibility="W",
            reversibility_rationale="Editing changes the file but can be rolled back via git",
            requires_files=[],
            requires_state_keys=[],
            cost_estimate=0.3,
            latency_estimate_ms=200.0
        )
        self.skills["run_tests"] = SkillEntry(
            skill_id="run_tests",
            name="Run Tests",
            description="Run the test suite to verify correctness",
            reversibility="R",
            reversibility_rationale="Running tests is read-only",
            requires_files=["**/test_*.py", "**/*_test.py"],
            requires_state_keys=[],
            cost_estimate=0.5,
            latency_estimate_ms=30000.0  # 30 seconds typical
        )

    def get(self, skill_id: str) -> Optional[SkillEntry]:
        return self.skills.get(skill_id)

    def get_all(self) -> list[SkillEntry]:
        return list(self.skills.values())


# Global skill registry
skill_registry = SkillRegistry()


# ── STATIC PRUNE FUNCTION (matching spec) ────────────────────────────────────────
def static_prune_skills(current_node: MCTSNode,
                        skill_registry: SkillRegistry,
                        vfs_state: dict) -> list[SkillEntry]:
    """
    Chain 1: Static Pruning - eliminate clearly inapplicable skills before LLM call.
    O(n_skills) — fast, cheap, no tokens.
    Matching PART 3 Chain 1 spec.
    """
    candidates = []
    for skill in skill_registry.get_all():
        # Check preconditions
        if skill.requires_files:
            root = vfs_state.get("root", ".")
            path = Path(root)
            has_files = False
            for pattern in skill.requires_files:
                clean_pattern = pattern.replace("**/", "").replace("*", "")
                if clean_pattern and list(path.glob(clean_pattern)):
                    has_files = True
                    break
            if not has_files:
                continue

        if skill.requires_state_keys:
            if not all(k in vfs_state for k in skill.requires_state_keys):
                continue

        if skill.forbidden_after:
            if current_node.skill_used in skill.forbidden_after:
                continue

        # Check reversibility gate - I-class only if allowed
        if skill.reversibility == "I" and not vfs_state.get("i_class_allowed", True):
            continue

        candidates.append(skill)

    # Context-based filtering (matching spec rules)
    has_code_changes = vfs_state.get("has_code_changes", False)

    # Don't run_tests if no code was edited yet
    if not has_code_changes:
        candidates = [c for c in candidates if c.skill_id != "run_tests"]

    return candidates


# ── UCT SELECT FUNCTION (matching spec) ─────────────────────────────────────────
def uct_select(root_node: MCTSNode, tree: SearchTree) -> MCTSNode:
    """
    Chain 2: UCT Selection - pure Python PUCT calculation.
    KILL Haiku LLM tie-breaker - use random tie-break instead.
    Matching PART 3 Chain 2 spec.
    """
    if not root_node.children:
        return root_node  # leaf — expand this

    # Multi-objective scalarization (from spec)
    w = {"quality": 0.7, "cost": 0.2, "latency": 0.05, "safety": 0.05}

    best_child = None
    best_ucb = float('-inf')
    best_candidates = []

    for child_id in root_node.children:
        child = tree.get_node(child_id)
        if not child or child.is_pruned:
            continue

        # Effective Q (scalarized multi-objective)
        Q_scalar = sum(w[k] * child.Q.get(k, 0.0) for k in w)

        # Cost-weighted exploration
        skill_cost = 0.5  # default
        if child.skill_used:
            skill = skill_registry.get(child.skill_used)
            if skill:
                skill_cost = skill.cost_estimate
        C_effective = tree.adaptive_C / max(0.1, skill_cost)

        # UCB-V formula (from spec)
        if child.N == 0:
            ucb = float('inf')  # always explore unvisited
        else:
            exploit = Q_scalar
            variance_bonus = math.sqrt(2 * child.variance * math.log(root_node.N + 1) / child.N)
            explore = C_effective * math.sqrt(math.log(root_node.N + 1) / child.N)
            ucb = exploit + variance_bonus + explore

        # Virtual loss
        ucb -= child.virtual_loss / max(1, child.N)

        if ucb > best_ucb:
            best_ucb = ucb
            best_candidates = [child]
        elif ucb == best_ucb:
            best_candidates.append(child)

    if not best_candidates:
        return root_node

    # Random tie-break (KILL Haiku - matching spec requirement)
    best_child = random.choice(best_candidates)

    # Apply virtual loss
    best_child.virtual_loss += 1.0

    return best_child