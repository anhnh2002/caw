"""Interactive mode — launch the agent and let the user take over."""

from caw import Agent


def main():
    agent = Agent()

    prompt = (
        "First, list the directories in the current directory. "
        "Then use AskUserQuestion to ask the user which directory is their favorite. "
        "Finally, count and tell the user how many python files are in that directory."
    )
    result = agent.interactive(prompt, capture_bytes=4096)

    print(f"\nExit code: {result.exit_code}")
    if result.session_id:
        print(f"Session ID: {result.session_id}")
    print(f"Captured {len(result.output)} chars of terminal output")


if __name__ == "__main__":
    main()
