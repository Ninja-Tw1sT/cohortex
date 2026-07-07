"""
Supervisor crew (config-driven): a supervisor delegates subtasks to a math
specialist (tool-using) and a summarizer, then writes the final answer.
Defined in configs/crews/assistant.yaml. Runs locally on Ollama.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cohortex.runtime import run_crew


def main() -> None:
    task = "Compute 145 * 12 - 40, then summarize in one sentence what a supervisor agent does."
    print(f"TASK: {task}\n" + "=" * 60)
    result = run_crew("assistant", task)
    for step in result.steps:
        print(f"\n[{step.agent}]\n{step.output}")
    print("\n" + "=" * 60 + f"\nFINAL:\n{result.output}")


if __name__ == "__main__":
    main()
