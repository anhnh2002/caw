"""Trajectory viewer: save a session trajectory and view it in the browser."""

import os
import tempfile

os.environ["CAW_LOG"] = "full"

from caw import Agent
from caw.viewer import start_viewer_server


def main():
    agent = Agent()

    # Run a short session and save the trajectory to a temp file
    traj_path = os.path.join(tempfile.gettempdir(), "caw_example_traj.json")

    with agent.start_session(traj_path=traj_path) as session:
        session.send("What is 2 + 2? Answer in one sentence.")

    print(f"\nTrajectory saved to {traj_path}")

    # Start the viewer server and print the URL
    server = start_viewer_server()
    print(f"Open in browser: {server.url}?path={traj_path}")

    print("Press Ctrl+C to stop the viewer.")
    try:
        import signal

        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
