"""Stateless tools demo: pass plain functions directly to an agent."""

import os

os.environ["CAW_LOG"] = "full"

from caw import Agent, tool


@tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b


@tool(description="Multiply two numbers")
def multiply(a: int, b: int) -> int:
    return a * b


def main():
    agent = Agent(
        system_prompt="You have access to math tools. Use them to answer questions.",
        stateless_tools=[add, multiply],
    )

    with agent.start_session() as session:
        session.send("List every tool you have access to by name.")
        session.send("What is 3 + 4? Then multiply the result by 5.")

        traj = session.trajectory
        print(f"\nTurns: {traj.num_turns}, Tool calls: {traj.total_tool_calls}")


if __name__ == "__main__":
    main()
