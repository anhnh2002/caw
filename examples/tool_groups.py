"""Tool groups demo: restrict an agent to read-only tools."""

import os

os.environ["CAW_LOG"] = "full"

from caw import Agent, ToolGroup


def main():
    # Only allow Read, Glob, Grep — no Bash, Write, Edit, WebSearch, etc.
    agent = Agent(tools=ToolGroup.READER)

    traj = agent.completion(
        "List every tool you have access to by name. "
        "Then answer: can you use the Bash tool? Can you use the Write tool? "
        "Can you use the Edit tool? Can you use the WebSearch tool?"
    )
    print(traj.result)
    print(f"\nis_complete: {traj.is_complete}")
    print(f"is_usage_limited: {traj.is_usage_limited}")


if __name__ == "__main__":
    main()
