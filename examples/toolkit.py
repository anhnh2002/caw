"""Custom tool server demo: a stateful user database exposed via ToolKit."""

import os

os.environ["CAW_LOG"] = "full"

from caw import Agent, ToolKit, tool


class UserDB(ToolKit, server_name="user_db", display_name="User Database"):
    def __init__(self):
        self.users = ["Alice", "Bob", "Charlie"]
        self.count = 0

    @tool(description="List all users in the database")
    async def list_users(self) -> str:
        self.count += 1
        return f"Users: {', '.join(self.users)} (queried {self.count} time(s))"

    @tool(description="Add a user to the database")
    async def add_user(self, name: str) -> str:
        self.users.append(name)
        return f"Added {name}. Total users: {len(self.users)}"


def main():
    db = UserDB()
    agent = Agent(
        system_prompt="You have access to a user database. Use the tools to answer questions about users.",
        tool_servers=[db],
    )

    with agent.start_session() as session:
        session.send("How many users are in the database? List them.")
        session.send("Add a user named Diana, then list all users again.")

        traj = session.trajectory
        print(f"\nTurns: {traj.num_turns}, Tool calls: {traj.total_tool_calls}")

    # State persists across turns (server stayed alive for the whole session)
    print(f"Final DB state: users={db.users}, count={db.count}")


if __name__ == "__main__":
    main()
