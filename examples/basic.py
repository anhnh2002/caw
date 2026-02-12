"""Basic usage: single turn, tool use, and multi-turn sessions."""

import os

os.environ["CAW_LOG"] = "full"

from caw import Agent


def main():
    agent = Agent()

    print("=== Single turn ===")
    with agent.start_session() as session:
        session.send("What is 2 + 2? Answer in one sentence.")
        print()

    print("=== Tool use turn ===")
    with agent.start_session() as session:
        session.send("List files in the current directory.")
        print()

    print("=== Multi-turn ===")
    with agent.start_session() as session:
        session.send("Remember the number 42.")
        session.send("What number did I just tell you?")

        traj = session.trajectory
        print(f"\nTurns: {traj.num_turns}")
        print(f"Total tool calls: {traj.total_tool_calls}")
        print(f"Total tokens: {traj.usage.total_tokens}")


if __name__ == "__main__":
    main()
