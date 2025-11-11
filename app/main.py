"""Entrypoint for the MLX OpenAI Server CLI that forwards control to the Click-based CLI.

This module provides a lightweight wrapper so the package can be executed
with ``python -m app.main`` and behave the same as the installed console
script. It collects command-line arguments, applies a small
backwards-compatibility rule (inserting the ``launch`` subcommand when no
explicit subcommand is given), and calls the Click command group defined
in ``app.cli:cli``.

Usage examples:

    # Run the default launch command (when no args are given)
    python -m app.main

    # Forward custom arguments to the CLI
    python -m app.main launch --port 8000

The actual CLI implementation and commands live in :mod:`app.cli`.
"""

import sys

from app.cli import cli


def main(argv: list | None = None):
    """Build and forward CLI arguments to the Click-based CLI.

    This function mirrors the behavior of the installed console script and
    delegates execution to the Click command group defined in
    `app.cli:cli`. It collects arguments from the current process
    (``sys.argv[1:]``) and optionally appends any values passed in via
    ``argv``.

    For backwards compatibility, if no subcommand is supplied (for
    example when running just ``python -m app.main``) or the first
    argument appears to be an option (starts with a dash), the
    "launch" subcommand is inserted as the first argument.

    Args:
        argv (Optional[Iterable[str]]): Extra arguments to append to the
            process arguments. If provided, these values are converted to
            strings and appended to the computed argument list. If ``None``,
            only ``sys.argv`` values are used.

    Returns:
        None: This function delegates to ``cli.main`` which may exit the
        process or raise exceptions depending on the invoked command.
    """

    cli_args = [str(x) for x in sys.argv[1:]]
    # Keep backwards compatibility: Add 'launch' subcommand if none is provided
    if not cli_args or cli_args[0].startswith("-"):
        cli_args.insert(0, "launch")
    if argv is not None:
        method_args = [str(x) for x in argv]
        args = cli_args + method_args
    else:
        args = cli_args
    cli.main(args=args)


if __name__ == "__main__":
    main()
