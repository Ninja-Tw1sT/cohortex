"""
Sequential crew (config-driven): researcher -> writer -> editor, each fed the
previous agent's output. Defined entirely in configs/crews/research_team.yaml.
Runs locally on Ollama.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cohortex.runtime import run_crew


def main() -> None:
    task = "Why vector databases matter for AI applications"
    print(f"TASK: {task}\n" + "=" * 60)
    result = run_crew("research_team", task)
    for step in result.steps:
        print(f"\n[{step.agent}]\n{step.output}")
    print("\n" + "=" * 60 + f"\nFINAL:\n{result.output}")


if __name__ == "__main__":
    main()
