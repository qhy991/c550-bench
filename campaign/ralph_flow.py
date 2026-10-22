"""Return the author entry prompt for one C500 Bench session."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def entry_prompt(root=None):
    """Build the initial prompt passed to the restricted Claude Code author session."""
    if root is None:
        root = ROOT
    root = Path(root)
    binding = __import__('json').loads((root / 'campaign/binding.json').read_text())
    task_text = (root / 'campaign/TASK.md').read_text()
    scaffold_text = (root / binding['author']['scaffold']).read_text()
    return '\n\n'.join([scaffold_text, task_text])
