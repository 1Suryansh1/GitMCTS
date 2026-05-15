"""I-class gate - Block irreversible actions until human approval."""

I_CLASS_PATTERNS = [
    "git push",
    "git push origin",
    "git push --all",
    "deploy",
    "rm -rf",
    "rmdir /s /q",
    "del /f /s /q",
    "DROP TABLE",
    "DELETE FROM",
    "kubectl apply",
    "kubectl delete",
    "terraform apply",
    "terraform destroy",
    "send_email",
    "stripe.charge",
    "twilio.send",
    "aws s3 rm",
    "gcloud run deploy",
    "npm publish",
    "pip publish",
    "docker push",
    "chmod -R 000",
    "> file",
    "echo >",
]


def classify_action(action: str) -> str:
    """Classify action as R (reversible), W (write), or I (irreversible)."""
    action_lower = action.lower()
    for pattern in I_CLASS_PATTERNS:
        if pattern.lower() in action_lower:
            return "I"
    write_patterns = ["write", "create", "mkdir", "touch", "append", "sed", "tee"]
    for w in write_patterns:
        if w in action_lower:
            return "W"
    return "R"


def gate_i_class(action: str, context: str, auto_approve: bool = False) -> bool:
    """
    Display the action, explain why it's irreversible, ask for approval.
    Returns True if approved.
    """
    if auto_approve:
        return True

    print(f"""
╔════════════════════════════════════════════════════════╗
║  ⚠️  IRREVERSIBILITY GATE — ACTION REQUIRES APPROVAL   ║
╠════════════════════════════════════════════════════════╣
║  Action:  {action[:54]:<54} ║
║  Context: {context[:54]:<54} ║
║                                                       ║
║  This action cannot be undone by git rollback.        ║
║  GitMCTS has paused to confirm.                        ║
╚════════════════════════════════════════════════════════╝
    Approve? [y/N]: """, end="")

    try:
        response = input().strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def check_command_safety(command: str, goal: str, auto_approve: bool = False) -> tuple[bool, str]:
    """
    Check if a command is safe to execute or needs gating.
    Returns (is_safe, classification).
    """
    action_class = classify_action(command)
    if action_class == "I":
        approved = gate_i_class(command, goal[:100], auto_approve)
        return (approved, "I-gated" if not approved else "I-approved")
    return (True, action_class)