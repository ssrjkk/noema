"""Allow running ``python -m noema`` as an alias for the ``noema`` CLI."""

from noema.cli.main import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
