"""Subagent demo: a parent agent delegates code review to a subagent."""

import os

os.environ["CAW_LOG"] = "full"

from caw import Agent, AgentSpec


def main():
    reviewer = AgentSpec(
        name="Code Reviewer",
        description="Review code for correctness and style issues.",
        system_prompt="You are a code reviewer. Given code, identify bugs and style issues. Be concise.",
    )

    agent = Agent(
        system_prompt="You are a senior engineer. Use the Code Reviewer tool to review code when asked.",
    )
    agent.add_subagent(reviewer)

    with agent.start_session() as session:
        turn = session.send("Review this Python function:\n\ndef add(a, b):\n    return a - b\n")

        traj = session.trajectory
        print(f"\nParent own usage: ${traj.usage.cost_usd:.4f}")
        print(f"Parent total usage (with subagents): ${traj.total_usage.cost_usd:.4f}")
        print(f"Parent total tokens: {traj.total_usage.total_tokens}")

        for tc in turn.tool_calls:
            if tc.subagent_trajectory:
                st = tc.subagent_trajectory
                print(f"\n  Subagent '{tc.name}':")
                print(f"    Model: {st.model}")
                print(f"    System prompt: {st.system_prompt[:60]}...")
                print(f"    Tool calls: {st.total_tool_calls}")
                print(f"    Usage: ${st.usage.cost_usd:.4f} ({st.usage.total_tokens} tokens)")


if __name__ == "__main__":
    main()
