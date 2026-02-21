# /// script
# dependencies = [
#   "rich",
# ]
# ///

import argparse
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run QxQ tests.")
    parser.add_argument("--release", action="store_true", help="Use release binary")
    parser.add_argument(
        "--test", action="store_true", help="Run only traditional tests"
    )
    parser.add_argument(
        "--test-expect", action="store_true", help="Run only expect-tests"
    )
    parser.add_argument(
        "--update-expect", action="store_true", help="Update EXPECT: blocks"
    )
    parser.add_argument(
        "--skip-multiple-expect", action="store_true", help="Pass to qxq during update"
    )

    args = parser.parse_args()

    mode = "release" if args.release else "debug"
    project_root = Path(__file__).parent.parent.resolve()
    tests_dir = project_root / "tests"
    binary_path = project_root / "target" / mode / "qxq"

    if not binary_path.exists():
        console.print(
            f"[yellow]Binary not found at {binary_path}. Building...[/yellow]"
        )
        subprocess.run(
            ["cargo", "build"] + (["--release"] if args.release else []),
            cwd=project_root,
            check=True,
        )

    test_files = sorted(list(tests_dir.rglob("*.qxq")))
    results = []

    # Default to traditional test if no flags provided
    is_traditional_only = args.test or (not args.test_expect and not args.update_expect)

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing tests...", total=len(test_files))

        for file in test_files:
            rel_path = file.relative_to(tests_dir)
            category = file.parent.name if file.parent != tests_dir else "uncategorized"
            content = file.read_text()
            has_run = "(* RUN:" in content or "(* RUN-EXPECT-ERROR:" in content

            skip = False
            run_mode = "TRADITIONAL"
            cmd_args = [str(file)]

            if args.update_expect:
                if not has_run:
                    skip = True
                else:
                    run_mode = "UPDATE"
                    cmd_args = ["--update-expect", str(file)]
                    if args.skip_multiple_expect:
                        cmd_args.append("--skip-multiple-expect")
            elif args.test_expect:
                if not has_run:
                    skip = True
                else:
                    run_mode = "EXPECT"
                    cmd_args = ["--test-expect", str(file)]
            else:
                # Traditional test mode (--test)
                run_mode = "TRADITIONAL"
                # Always run in traditional mode unless explicitly told to skip elsewhere

            if skip:
                results.append(
                    {
                        "category": category,
                        "file": str(rel_path),
                        "success": True,
                        "skipped": True,
                        "mode": "SKIPPED",
                    }
                )
            else:
                process = subprocess.run(
                    [str(binary_path)] + cmd_args, capture_output=True, text=True
                )
                success = process.returncode == 0
                is_skipped = process.returncode == 2

                results.append(
                    {
                        "category": category,
                        "file": str(rel_path),
                        "success": success or is_skipped,
                        "skipped": is_skipped,
                        "mode": run_mode,
                        "error": process.stderr
                        if not (success or is_skipped)
                        else None,
                        "stdout": process.stdout,
                    }
                )

            progress.update(task, advance=1)

    table = Table(title="QxQ Testsuite Execution Results")
    table.add_column("Category", style="cyan")
    table.add_column("Test Case", style="magenta")
    table.add_column("Mode", style="blue")
    table.add_column("Result", justify="center")

    passed = 0
    for res in results:
        if res["skipped"]:
            status = "[yellow]SKIP[/yellow]"
        else:
            status = "[green]PASS[/green]" if res["success"] else "[red]FAIL[/red]"
            if res["success"]:
                passed += 1
        table.add_row(res["category"], res["file"], res["mode"], status)

    console.print(table)

    if any(r["skipped"] for r in results):
        console.print(
            f"\n[yellow]Total skipped: {sum(1 for r in results if r['skipped'])}[/yellow]"
        )

    for res in results:
        if res.get("stdout") and "Skipping update" in res["stdout"]:
            console.print(f"[yellow]{res['stdout'].strip()}[/yellow]")

    failures = [r for r in results if not r["success"] and not r["skipped"]]
    if failures:
        console.print("\n[red]Failures:[/red]")
        for fail in failures:
            console.print(
                f"[bold red]File: {fail['file']}[/bold red]\n{fail['error']}\n{'-' * 20}"
            )
        sys.exit(1)

    console.print(
        f"\n[bold]Summary: {passed}/{sum(1 for r in results if not r['skipped'])} passed[/bold]"
    )


if __name__ == "__main__":
    main()
