"""MCTS Prompt Chains implementation - matching PART 3 of spec.

Chain 0: Bootstrap (Opus) - Analyze goal, create root node
Chain 1: Static Prune (Python) - Filter skills by context
Chain 2: UCT Select (Python math) - Pure PUCT calculation
Chain 3: Skill Expansion (Sonnet) - Pick skill + update plan
Chain 4: Execution - Branch, execute, commit
Chain 5: Fast Eval + Oracle combined
Chain 7: Backprop - Update N and Q up the tree
Chain 8: Terminal Check - Pure Python (kill LLM)
Chain 10: Commit Gate - Human approval for I-class
"""

import json
import math
import random
import subprocess
import os
import shutil
from pathlib import Path
from typing import Optional
from .mcts_node import MCTSNode, SearchTree, SkillEntry, skill_registry, static_prune_skills, uct_select


# =============================================================================
# CHAIN 0: BOOTSTRAP (Opus) - matching spec
# =============================================================================

from google import genai
from google.genai import types

def get_genai_client():
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    return genai.Client()


def _thinking_config():
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="high")
    )


# Matching PART 3 Chain 0 spec
CHAIN0_SYSTEM = """You are the GitMCTS architect. Your job is to analyze a task and produce the
structured initialization payload for an MCTS search session. You must output
ONLY valid JSON matching the schema below. No prose, no explanation.

Schema:
{
  "goal_class": string,
  "goal_summary": string,
  "complexity_score": float,
  "estimated_depth": int,
  "initial_C": float,
  "subtasks": [{"id": string, "description": string, "can_parallelize": bool}],
  "relevant_skill_ids": [string],
  "irreversible_risk": "low"|"medium"|"high",
  "suggested_budget": {"simulations": int, "cost_usd": float},
  "context_handoff_key_fields": [string]
}"""


def chain0_bootstrap(goal: str, repo_path: str, l3_count: int = 0) -> dict:
    """
    Chain 0: Bootstrap - Analyze goal and create initial plan.
    Uses Gemini API.
    Matching PART 3 Chain 0 spec.
    """
    client = get_genai_client()

    user_prompt = f"""
Goal: {goal}
Repository: {repo_path}
Available skills: {json.dumps([s.__dict__ for s in skill_registry.get_all()])}
Past episodes in this goal class from L3: {l3_count}
"""

    if client:
        try:
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=f"{CHAIN0_SYSTEM}\n\n{user_prompt}",
                config=_thinking_config()
            )
            result_text = response.text
            import re
            m = re.search(r'\{.*\}', result_text, re.DOTALL)
            if m:
                result_text = m.group(0)
            result = json.loads(result_text)
            return result
        except Exception as e:
            print(f"DEBUG: LLM ERROR in chain0_bootstrap: {e}")
            pass

    # Fallback (no LLM)
    return {
        "goal_class": "code_fix",
        "goal_summary": goal[:50],
        "complexity_score": 0.5,
        "estimated_depth": 3,
        "initial_C": 1.0 if l3_count < 5 else 0.8,
        "subtasks": [],
        "relevant_skill_ids": ["read_file", "edit_file", "run_tests"],
        "irreversible_risk": "low",
        "suggested_budget": {"simulations": 20, "cost_usd": 1.0},
        "context_handoff_key_fields": []
    }


# =============================================================================
# CHAIN 1: STATIC PRUNE (Python) - matching spec
# =============================================================================

def chain1_static_prune(current_node: MCTSNode, vfs_state: dict) -> list[SkillEntry]:
    """
    Chain 1: Static Pruning - filter skills based on context.
    Pure Python, no LLM.
    Matching PART 3 Chain 1 spec.
    """
    return static_prune_skills(current_node, skill_registry, vfs_state)


# =============================================================================
# CHAIN 2: UCT SELECT (Python math) - matching spec
# =============================================================================

def chain2_uct_select(tree: SearchTree) -> MCTSNode:
    """
    Chain 2: UCT Selection - pure PUCT calculation.
    KILL Haiku LLM tie-breaker - random tie-break.
    Matching PART 3 Chain 2 spec.
    """
    if not tree.nodes:
        return None

    root = tree.get_node(tree.root_node_id)
    if not root:
        return None

    return uct_select(root, tree)


# =============================================================================
# CHAIN 3: SKILL EXPANSION (Sonnet) - matching spec
# =============================================================================

# Matching PART 3 Chain 3 spec
CHAIN3_SYSTEM = """You are the GitMCTS skill expander. Given the current world state and a filtered
list of candidate skills, propose the single best (skill, parameter_binding)
action to expand from the current node.

Rules:
1. Output ONLY valid JSON for the primary payload.
2. Parameters must be SPECIFIC and grounded in the VFS state.
3. IMPORTANT: If using the `edit_file` skill, do NOT put the full file content inside the JSON. Instead, set `"new_content": "<SEE_BLOCK>"` in the JSON, and output the actual replacement code IMMEDIATELY AFTER the JSON block, wrapped in ```python ... ``` tags.
4. Check for semantic similarity to already-expanded siblings. If your proposed
   action is >0.85 similar to any sibling, propose the second-best alternative.
5. For I-class skills, flag explicitly in your reasoning field.

Output schema:
{
  "skill_id": string,
  "params": dict,
  "reversibility": "R"|"W"|"I",
  "confidence": float,
  "rationale": string,
  "context_key": string,
  "plan_md_update": string,
  "is_macro": bool,
  "macro_steps": []
}"""


def chain3_skill_expansion(
    current_node: MCTSNode,
    candidates: list[SkillEntry],
    goal: str,
    vfs_state: dict,
    plan_md: str,
    tree: SearchTree,
    context_buffer: dict = None
) -> dict:
    """
    Chain 3: Skill Expansion - pick best skill and update plan.
    Uses Gemini API.
    Matching PART 3 Chain 3 spec.
    """
    client = get_genai_client()

    # Build candidate info
    candidate_info = json.dumps([
        {"skill_id": s.skill_id, "name": s.name, "reversibility": s.reversibility}
        for s in candidates
    ])

    # Sibling info
    siblings = []
    if current_node.children:
        for child_id in current_node.children:
            child = tree.get_node(child_id)
            if child:
                siblings.append({"skill_id": child.skill_used, "params": child.skill_params})

    user_prompt = f"""
Goal slice for this node: {current_node.goal_slice or goal}

Current VFS state: {json.dumps(vfs_state)}

Candidate skills: {candidate_info}

Already-expanded siblings: {json.dumps(siblings)}

Node depth: {current_node.depth}
"""

    if context_buffer and context_buffer.get("last_read_content"):
        user_prompt += f"\n\nLast file read ({context_buffer['last_read_file']}):\n{context_buffer['last_read_content']}"

    if client:
        try:
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=f"{CHAIN3_SYSTEM}\n\n{user_prompt}",
                config=_thinking_config()
            )
            result_text = response.text
            import re
            # 1. Parse JSON safely
            m = re.search(r'\{.*\}', result_text, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
            else:
                return {"skill_id": "run_tests", "params": {}, "reversibility": "R", "confidence": 0.5}

            # 2. Extract code block if editing
            if result.get("skill_id") == "edit_file":
                code_match = re.search(r'```(?:python)?(.*?)```', result_text, re.DOTALL)
                if code_match:
                    result["params"]["new_content"] = code_match.group(1).strip()

            return result
        except Exception:
            pass

    # Fallback
    if candidates:
        return {
            "skill_id": candidates[0].skill_id,
            "params": {},
            "reversibility": candidates[0].reversibility,
            "confidence": 0.5,
            "rationale": "default",
            "context_key": ""
        }
    return {"skill_id": "run_tests", "params": {}, "reversibility": "R", "confidence": 0.5}


# =============================================================================
# CHAIN 4: EXECUTION - matching spec GitVFSLayer
# =============================================================================

class GitVFS:
    """
    The world-state substrate. Every MCTS node = one Git commit.
    All operations are O(1) or O(changed_subtree).
    Matching PART 2 spec.
    """

    def __init__(self, repo_path: str):
        import git
        self.repo = git.Repo(repo_path)
        self.root = Path(repo_path)
        self.prefix = "mcts"
        self.object_store = self.repo.odb

    def init_search(self, root_goal: str) -> str:
        """
        git checkout -b mcts/root/{goal_hash}
        echo goal > .mcts/goal.txt
        git add . && git commit -m "init: {goal_hash}"
        Returns: root commit hash (= root node_id)
        Matching PART 2 spec.
        """
        import hashlib
        goal_hash = hashlib.sha256(root_goal.encode()).hexdigest()[:8]
        branch_name = f"{self.prefix}/root/{goal_hash}"

        # Create root branch
        root_branch = self.repo.create_head(branch_name)
        root_branch.checkout()

        # Write goal file to VFS
        mcts_dir = self.root / ".mcts"
        mcts_dir.mkdir(exist_ok=True)
        (mcts_dir / "goal.txt").write_text(root_goal)
        (mcts_dir / "node_meta.json").write_text(json.dumps({
            "goal_hash": goal_hash, "depth": 0, "skill": None
        }))

        self.repo.index.add([".mcts/goal.txt", ".mcts/node_meta.json"])
        commit = self.repo.index.commit(f"init: goal/{goal_hash}")

        return commit.hexsha

    def branch_node(self, parent_node_id: str, skill_id: str) -> tuple[str, Path]:
        """
        git checkout {parent_node_id}
        git checkout -b mcts/node/{new_branch_id}
        O(1) pointer copy. No data duplication.
        Returns: branch_name
        Matching PART 2 spec.
        """
        import uuid

        # Get parent commit
        if parent_node_id:
            parent_commit = self.repo.commit(parent_node_id)
        else:
            parent_commit = self.repo.head.commit

        # Create child branch
        child_branch_name = f"{self.prefix}/node/{uuid.uuid4().hex[:8]}"
        new_branch = self.repo.create_head(child_branch_name, commit=parent_commit)

        # Add worktree
        worktree_path = self.root / ".mcts_worktrees" / child_branch_name.replace("/", "_")
        worktree_path.mkdir(parents=True, exist_ok=True)
        self.repo.git.worktree("add", str(worktree_path), child_branch_name)

        return child_branch_name, worktree_path

    def stage_output(self, branch_name: str, output: dict, skill_id: str, worktree_path: Path):
        """Stage skill output to VFS. Matching PART 2 spec."""
        import hashlib
        artifacts_dir = worktree_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        output_bytes = json.dumps(output, sort_keys=True).encode()
        blob_hash = hashlib.sha256(output_bytes).hexdigest()[:16]
        artifact_file = artifacts_dir / f"{skill_id}_{blob_hash}.json"
        artifact_file.write_bytes(output_bytes)

        import git as gitlib
        worktree_repo = gitlib.Repo(worktree_path)
        worktree_repo.index.add([str(artifact_file.relative_to(worktree_path))])

    def commit_node(self, branch_name: str, skill_id: str, params: dict,
                    fast_score: float, worktree_path: Path, parent_node_id: str) -> str:
        """
        git commit -m "skill: {skill_id} | score: {fast_score}"
        The commit hash IS the node_id.
        Matching PART 2 spec.
        """
        import git as gitlib
        worktree_repo = gitlib.Repo(worktree_path)

        meta_file = worktree_path / ".mcts" / "node_meta.json"
        meta_file.parent.mkdir(exist_ok=True)
        meta_file.write_text(json.dumps({
            "skill_id": skill_id,
            "params": params,
            "fast_score": fast_score,
            "parent_id": parent_node_id
        }, indent=2))
        worktree_repo.index.add([".mcts/node_meta.json"])

        commit_msg = f"skill: {skill_id} | score: {fast_score:.3f} | branch: {branch_name}"
        try:
            commit = worktree_repo.index.commit(commit_msg)
            return commit.hexsha
        except Exception:
            return parent_node_id

    def rollback(self, node_id: str, worktree_path: Path, branch_name: str = ""):
        """
        O(1). Remove worktree. The branch and its commits remain in object store.
        Matching PART 2 spec.
        """
        try:
            self.repo.git.worktree("remove", str(worktree_path), "--force")
        except Exception:
            pass
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)

    def diff_from_root(self, node_id: str) -> str:
        """Git diff from root to node. Matching PART 2 spec."""
        try:
            root = self.repo.head.commit.hexsha
            return self.repo.git.diff(root, node_id, "--stat")
        except Exception:
            return ""

    def get_episode_trace(self, leaf_node_id: str) -> list[dict]:
        """git log from leaf to root. Matching PART 2 spec."""
        trace = []
        try:
            commit = self.repo.commit(leaf_node_id)
            while commit:
                trace.append({
                    "node_id": commit.hexsha,
                    "message": commit.message
                })
                commit = commit.parents[0] if commit.parents else None
        except Exception:
            pass
        return trace[::-1]


def chain4_execute(
    vfs: GitVFS,
    parent_node_id: str,
    skill: SkillEntry,
    params: dict,
    goal: str,
    plan_md: str,
    expansion_result: dict = None
) -> tuple[str, Path, str, dict]:
    """
    Chain 4: Execution - Branch git tree, execute skill, save plan, commit.
    Matching PART 3 Chain 4 spec.
    """
    # Branch (create worktree)
    branch_name, worktree_path = vfs.branch_node(parent_node_id, skill.skill_id)

    # Execute the skill
    result = execute_mcp_tool(skill.skill_id, params, worktree_path, goal)

    # Update plan - use plan_md_update from expansion if available
    if expansion_result and expansion_result.get("plan_md_update"):
        updated_plan = expansion_result["plan_md_update"]
    else:
        step_num = params.get("step", 0)
        updated_plan = plan_md + f"\n\n### Step {step_num}: {skill.skill_id}\n- Result: {result.get('summary', 'executed')}"

    # Write plan to filesystem (Evolving Plan DNA)
    (worktree_path / "mcts_plan.md").write_text(updated_plan)

    # Commit
    score = result.get("score", 0.0)
    new_node_id = vfs.commit_node(branch_name, skill.skill_id, params, score, worktree_path, parent_node_id)

    return new_node_id, worktree_path, updated_plan, result


def execute_mcp_tool(skill_id: str, params: dict, worktree_path: Path, goal: str) -> dict:
    """Execute an MCP tool in the worktree."""

    if skill_id == "read_file":
        return execute_read_file(params, worktree_path, goal)
    elif skill_id == "edit_file":
        return execute_edit_file(params, worktree_path, goal)
    elif skill_id == "run_tests":
        return execute_run_tests(params, worktree_path)
    else:
        return {"success": False, "error": f"Unknown skill: {skill_id}", "score": 0.0}


def execute_read_file(params: dict, worktree_path: Path, goal: str) -> dict:
    """Execute read_file skill."""
    file_path = params.get("file_path", "")

    if not file_path:
        py_files = [f for f in worktree_path.rglob("*.py") if "test" not in f.name.lower()]
        if py_files:
            file_path = str(py_files[0].relative_to(worktree_path))

    if not file_path:
        return {"success": False, "error": "No file found", "score": 0.2}

    full_path = worktree_path / file_path
    if not full_path.exists():
        return {"success": False, "error": f"File not found: {file_path}", "score": 0.0}

    content = full_path.read_text(errors="ignore")
    return {
        "success": True,
        "file_path": file_path,
        "content": content[:2000],
        "summary": f"Read {file_path} ({len(content)} chars)",
        "score": 0.2  # MAX capped at 0.4 - tree must run tests to get reward
    }


def execute_edit_file(params: dict, worktree_path: Path, goal: str) -> dict:
    """Execute edit_file skill."""
    file_path = params.get("file_path", "")
    new_content = params.get("new_content", "")

    if not file_path:
        return {"success": False, "error": "No file_path specified", "score": 0.0}

    full_path = worktree_path / file_path
    if not full_path.exists():
        return {"success": False, "error": f"File not found: {file_path}", "score": 0.0}

    if new_content:
        full_path.write_text(new_content)
        return {
            "success": True,
            "file_path": file_path,
            "summary": f"Wrote {len(new_content)} chars",
            "score": 0.4  # MAX capped at 0.4 - tree must run tests to get reward
        }
    return {"success": True, "file_path": file_path, "summary": "Edit requested", "score": 0.4}


def execute_run_tests(params: dict, worktree_path: Path) -> dict:
    """Execute run_tests skill - THE ORACLE. Matching PART 3 Chain 5."""
    import sys
    test_cmd = params.get("test_cmd", "pytest")

    cmd_list = [sys.executable, "-m", "pytest", "--json-report", "--json-report-file=.mcts_test_report.json", "-x", "-q"] if test_cmd == "pytest" else [test_cmd, "--json-report", "--json-report-file=.mcts_test_report.json", "-x", "-q"]

    try:
        result = subprocess.run(
            cmd_list,
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=120
        )

        report_path = worktree_path / ".mcts_test_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
                summary = report.get("summary", {})
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                total = passed + failed
                score = passed / total if total > 0 else 0.0
                return {
                    "success": True,
                    "passed": passed,
                    "failed": failed,
                    "total": total,
                    "score": score,
                    "summary": f"{passed}/{total} tests passing"
                }
            except json.JSONDecodeError:
                pass

        # FALLBACK: Use returncode if JSON parsing fails
        if result.returncode == 0:
            return {"success": True, "passed": 1, "failed": 0, "total": 1, "score": 1.0, "summary": "All tests passed (via returncode)"}
        else:
            return {"success": True, "passed": 0, "failed": 1, "total": 1, "score": 0.0, "summary": "Tests failed (via returncode)"}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Test timeout", "score": 0.0}
    except Exception as e:
        return {"success": False, "error": str(e), "score": 0.0}


# =============================================================================
# CHAIN 5: FAST EVAL + ORACLE (Combined) - matching spec
# =============================================================================

def chain5_fast_eval(skill_id: str, execution_result: dict) -> float:
    """
    Chain 5: Fast Evaluation + Oracle combined.
    - If run_tests: use Pytest Oracle (objective, ungameable) - CAN return > 0.5
    - If read_file/edit_file: return heuristic, MAX capped at 0.4
    The tree must learn that ONLY run_tests can reach Terminal State (1.0).
    """
    if skill_id == "run_tests":
        # Pytest Oracle - objective, tests pass or they don't
        return execution_result.get("score", 0.0)
    else:
        # Fast heuristic - capped at 0.4 max
        return min(execution_result.get("score", 0.4), 0.4)


# =============================================================================
# CHAIN 7: BACKPROPAGATION (Python) - matching spec
# =============================================================================

def chain7_backpropagate(leaf_node_id: str, tree: SearchTree, gamma: float = 0.95, alpha_decay: float = 0.1):
    """
    Chain 7: Backpropagation - Pure Python.
    Walk up the parent_id chain. Update N+=1 and Q using simple moving average.
    Matching PART 3 Chain 7 spec.
    """
    leaf = tree.get_node(leaf_node_id)
    if not leaf:
        return

    leaf_score = leaf.Q.get("quality", leaf.Q.get("quality", 0.0) if isinstance(leaf.Q, dict) else leaf.Q)
    if isinstance(leaf.Q, dict):
        leaf_score = leaf.Q.get("quality", 0.0)
    else:
        leaf_score = leaf.Q

    discount = 1.0
    current = leaf

    while current:
        # Decaying learning rate (Robbins-Monro)
        alpha_t = 1.0 / (1.0 + alpha_decay * current.N)

        # Update Q with simple moving average
        current.Q["quality"] = ((current.Q.get("quality", 0.0) * current.N) + leaf_score) / (current.N + 1)

        # Update visit count
        current.N += 1

        # Remove virtual loss
        current.virtual_loss = max(0, current.virtual_loss - 1.0)

        # Move to parent
        current = tree.get_node(current.parent_id) if current.parent_id else None
        discount *= gamma

    # Update tree best
    if leaf_score > tree.best_Q_quality:
        tree.best_Q_quality = leaf_score
        tree.best_node_id = leaf_node_id


# =============================================================================
# CHAIN 8: TERMINAL CHECK (Python - kill LLM) - matching spec
# =============================================================================

def chain8_terminal_check(tree: SearchTree, last_node_id: str) -> bool:
    """
    Chain 8: Terminal Check - Pure Python (KILL LLM version).
    If pytest_score == 1.0: is_terminal = True.
    Matching PART 3 Chain 8 spec (lite version).
    """
    last_node = tree.get_node(last_node_id)
    if not last_node:
        return False

    # Check if this was a run_tests node with perfect score
    if last_node.skill_used == "run_tests":
        score = last_node.Q.get("quality", last_node.Q) if isinstance(last_node.Q, dict) else last_node.Q
        if score >= 1.0:
            last_node.is_terminal = True
            return True

    # Check max depth (simplified from spec)
    if last_node.depth >= 5:
        last_node.is_terminal = True
        return True

    return False


# =============================================================================
# CHAIN 10: COMMIT GATE (Human approval) - matching spec
# =============================================================================

def chain10_commit_gate(winner_node_id: str, vfs: GitVFS, skill_used: str = None, reversibility: str = None, Q: float = 0.0) -> bool:
    """
    Chain 10: Commit Gate - Pause for human approval if irreversible actions.
    Returns True if approved.
    Matching PART 3 Chain 10 spec.
    """
    # Check if path involves I-class actions
    has_i_class = reversibility == "I"

    if not has_i_class:
        return True  # No I-class - auto approve

    diff = vfs.diff_from_root(winner_node_id)

    print(f"""
+----------------------------------------------------------+
|  WARNING - IRREVERSIBLE ACTION DETECTED                  |
+----------------------------------------------------------+
|  GitMCTS found a solution (Q={Q:.3f})                    |
|  The winning path includes an irreversible action:       |
|  Skill: {skill_used or 'unknown'}                                      |
|                                                          |
|  VFS changes:                                           |
|  {diff[:50] if diff else 'no diff'}                           |
|                                                          |
|  Type YES to commit, NO to cancel:                      |
+----------------------------------------------------------+
    """, end="")

    try:
        response = input().strip().upper()
        return response in ("YES", "Y")
    except (EOFError, KeyboardInterrupt):
        return False


def merge_winner(vfs: GitVFS, winner_node_id: str, commit_message: str) -> str:
    """Merge the winning node's branch back into the original branch."""
    try:
        # Find the branch that points to this commit
        winner_branch = None
        for ref in vfs.repo.references:
            if hasattr(ref, 'commit') and ref.commit.hexsha == winner_node_id:
                winner_branch = ref.name
                break

        # Get back to original branch (non-mcts branch)
        original = vfs.repo.active_branch

        if winner_branch:
            vfs.repo.git.merge(winner_branch, "--no-ff", "-m", commit_message)
        else:
            # Fallback: cherry-pick the commit
            vfs.repo.git.cherry_pick(winner_node_id)

        return vfs.repo.head.commit.hexsha
    except Exception as e:
        return winner_node_id  # return the node id so caller knows what won


# =============================================================================
# MAIN MCTS LOOP (replacing Chain 9 Meta-Controller) - matching spec
# =============================================================================

def mcts_search(goal: str, repo_path: str, config: dict = None, display=None) -> dict:
    """
    Main MCTS search loop.
    Replaces Chain 9 Meta-Controller with simple while loop.
    Matching PART 4 spec.
    FIX 5: Added Rich Live UI integration for demo.
    """
    config = config or {}
    budget = config.get("simulation_budget", 20)
    C = config.get("C", 1.0)

    # Initialize VFS
    vfs = GitVFS(repo_path)

    # Chain 0: Bootstrap
    bootstrap = chain0_bootstrap(goal, repo_path, l3_count=0)
    initial_C = bootstrap.get("initial_C", 1.0)

    # Init search (create root commit)
    root_node_id = vfs.init_search(goal)

    # Initialize tree
    tree = SearchTree(
        root_goal=goal,
        root_node_id=root_node_id,
        simulation_budget=budget,
        adaptive_C=initial_C,
        domain_familiarity=0
    )

    # Create root node
    root_node = MCTSNode(
        node_id=root_node_id,
        parent_id=None,
        depth=0,
        goal_slice=goal,
        goal_hash=bootstrap.get("goal_summary", "root")[:8],
        Q={"quality": 0.0, "cost": 1.0, "latency": 1.0, "safety": 1.0},
        N=0
    )
    tree.add_node(root_node)

    # Initial plan
    plan_md = f"""# GitMCTS Plan

## Goal
{goal}

## Strategy
- Complexity: {bootstrap.get('complexity_score', 0.5)}
- Estimated depth: {bootstrap.get('estimated_depth', 3)}
- Budget: {budget} simulations

## Steps
"""

    # VFS state
    vfs_state = {
        "has_code_changes": False,
        "has_test_files": True,
        "i_class_allowed": True,
        "root": repo_path
    }

    current_node = root_node
    simulations = 0
    is_terminal = False

    # Import Live for FIX 5
    from rich.live import Live

    print(f"\n[*] GitMCTS starting | goal: {goal[:50]}... | budget: {budget}\n")

    # Main loop (replacing Chain 9 Meta-Controller)
    with Live(display.render(), refresh_per_second=4) as live:
        while simulations < budget and not is_terminal:
            # Update display
            if display:
                live.update(display.render())

            # Chain 1: Static Prune
            candidates = chain1_static_prune(current_node, vfs_state)

            if not candidates:
                current_node.is_pruned = True
                current_node.pruned_reason = "no_valid_skills"
                if display:
                    from .vfs import Branch
                    pruned_branch = Branch(
                        name=f"sim-{simulations}|pruned",
                        node_id="",
                        worktree=Path(".")
                    )
                    display.add_branch(pruned_branch)
                    display.set_pruned(pruned_branch.name, "no_valid_skills")
                if current_node.parent_id:
                    current_node = tree.get_node(current_node.parent_id)
                continue

            # Chain 2: UCT Select (if we have children)
            if current_node.children:
                selected = uct_select(current_node, tree)
                if selected and selected.node_id != current_node.node_id:
                    current_node = selected

            # Chain 3: Skill Expansion - pass expansion for plan update
            expansion = chain3_skill_expansion(current_node, candidates, goal, vfs_state, plan_md, tree, current_node.context_buffer)
            skill_id = expansion.get("skill_id", "run_tests")
            skill_params = expansion.get("params", {})
            # Use plan_md_update from expansion if available (FIX 2)
            if expansion.get("plan_md_update"):
                plan_md = expansion["plan_md_update"]

            skill = skill_registry.get(skill_id)
            if not skill:
                skill = SkillEntry(skill_id=skill_id, name=skill_id, description="", reversibility="R")

            # Chain 4: Execution - pass expansion for plan DNA (FIX 2)
            try:
                new_node_id, worktree, plan_md, result = chain4_execute(
                    vfs, current_node.node_id, skill, skill_params, goal, plan_md, expansion
                )
            except Exception as e:
                if display:
                    display.set_pruned(f"node-{simulations}", f"error: {e}")
                simulations += 1
                live.update(display.render())
                continue

            # Chain 5: Fast Eval + Oracle
            score = chain5_fast_eval(skill_id, result)

            new_context = current_node.context_buffer.copy()
            if skill_id == "read_file":
                new_context["last_read_content"] = result.get("content", "")[:2000]
                new_context["last_read_file"] = result.get("file_path", "")

            if display:
                class _R:
                    passed = result.get("passed", 0)
                    failed = result.get("failed", 0)
                    errors = 0
                    def __init__(self):
                        self.score = score
                r_obj = _R()
                display.set_scored(f"sim-{simulations}|{skill_id}", r_obj)

            # Update VFS state
            vfs_state["has_code_changes"] = vfs_state["has_code_changes"] or (skill_id == "edit_file")

            # Create new node
            new_node = MCTSNode(
                node_id=new_node_id,
                parent_id=current_node.node_id,
                depth=current_node.depth + 1,
                goal_slice=goal,
                goal_hash=root_node.goal_hash,
                Q={"quality": score, "cost": 1.0, "latency": 1.0, "safety": 1.0},
                N=0,
                skill_used=skill_id,
                skill_params=skill_params,
                reversibility=skill.reversibility,
                context_buffer=new_context
            )

            if display:
                from .vfs import Branch
                from pathlib import Path
                ui_branch = Branch(
                    name=f"sim-{simulations}|{skill_id}",
                    node_id=new_node_id,
                    worktree=worktree
                )
                display.add_branch(ui_branch)
                display.set_exploring(ui_branch.name)

            tree.add_node(new_node)
            simulations += 1

            # Update display after evaluation
            if display:
                display.set_scored(f"node-{simulations-1}", type('obj', (object,), {'passed': result.get('passed', 0), 'failed': result.get('failed', 0), 'score': score})())
                live.update(display.render())

            # Chain 7: Backpropagation
            chain7_backpropagate(new_node_id, tree)

            # Chain 8: Terminal Check (pure Python - kill LLM)
            if chain8_terminal_check(tree, new_node_id):
                is_terminal = True
                if display:
                    display.set_winner(f"sim-{simulations-1}|{skill_id}")

            current_node = new_node

    tree.simulations_used = simulations

    # Find best
    winner = tree.get_best_node()
    if not winner:
        return {"success": False, "error": "No solution found"}

    print(f"\n[OK] Search complete | Simulations: {simulations} | Best Q: {tree.best_Q_quality:.3f}")

    return {
        "success": True,
        "winner_node_id": winner.node_id,
        "best_Q": tree.best_Q_quality,
        "simulations": simulations,
        "plan": plan_md
    }