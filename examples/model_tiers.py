"""Model tiers: use provider-agnostic model selection."""

import os

os.environ["CAW_LOG"] = "full"

from caw import Agent, ModelTier


def main():
    # Use the fast model tier — each provider maps this to its cheapest/fastest model.
    # Claude Code: claude-haiku-4-5-20251001, Codex: gpt-5.3-codex-spark
    agent = Agent(model=ModelTier.FAST, data_dir="caw_data")

    traj = agent.completion("What model are you? Answer in one sentence.")
    print(traj.result)
    print(f"\nmodel: {traj.model}")
    print(f"tokens: {traj.usage.total_tokens}")


if __name__ == "__main__":
    main()
