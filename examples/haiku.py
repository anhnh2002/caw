import os

os.environ["CAW_LOG"] = "full"

from caw import Agent

if __name__ == "__main__":
    agent = Agent(
        system_prompt="You are a software engineer.",
        data_dir="caw_data",
    )

    with agent.start_session() as session:
        session.send(
            "Can you explore the codebase at caw/auth and tell me what it does in five sentences? If possible, do the exploration use Haiku model"
        )
