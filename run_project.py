"""Entry point script for the project demo."""

from src.demo import run_project_demo


def main() -> None:
    log_path = run_project_demo()
    print(f"Project demo completed successfully. Log: {log_path}")


if __name__ == "__main__":
    main()
