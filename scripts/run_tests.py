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
    parser.add_argument(
        "--release",
        action="store_true",
        help="Run tests using the release binary (default is debug)",
    )
    args = parser.parse_args()

    mode = "release" if args.release else "debug"
    project_root = Path(__file__).parent.parent.resolve()
    tests_dir = project_root / "tests"
    binary_path = project_root / "target" / mode / "qxq"

    if not binary_path.exists():
        console.print(
            f"[yellow]Binary not found at {binary_path.relative_to(project_root)}. Building project in {mode} mode...[/yellow]"
        )
        build_cmd = ["cargo", "build"]
        if args.release:
            build_cmd.append("--release")
        subprocess.run(build_cmd, cwd=project_root, check=True)

    test_files = sorted(list(tests_dir.rglob("*.qxq")))

    if not test_files:
        console.print("[red]No test files found in tests/ directory.[/red]")
        sys.exit(1)

    # Group tests by their parent directory name (category)
    categories = {}
    for test_file in test_files:
        category = test_file.parent.name
        if category == "tests":
            category = "uncategorized"
        if category not in categories:
            categories[category] = []
        categories[category].append(test_file)

    results = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Running tests...", total=len(test_files))

        for category, files in categories.items():
            for file in files:
                rel_path = file.relative_to(tests_dir)
                try:
                    process = subprocess.run(
                        [str(binary_path), str(file)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    success = process.returncode == 0
                    results.append(
                        {
                            "category": category,
                            "file": str(rel_path),
                            "success": success,
                            "error": process.stderr if not success else None,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "category": category,
                            "file": str(rel_path),
                            "success": False,
                            "error": str(e),
                        }
                    )
                progress.update(task, advance=1)

    # Display results
    table = Table(title="QxQ Test Results")
    table.add_column("Category", style="cyan")
    table.add_column("Test Case", style="magenta")
    table.add_column("Result", justify="center")

    passed_count = 0
    for res in results:
        status = "[green]PASS[/green]" if res["success"] else "[red]FAIL[/red]"
        if res["success"]:
            passed_count += 1
        table.add_row(res["category"], res["file"], status)

    console.print(table)

    # Show failures
    failures = [r for r in results if not r["success"]]
    if failures:
        console.print("\n[red]Failures:[/red]")
        for fail in failures:
            console.print(f"[bold red]File: {fail['file']}[/bold red]")
            console.print(fail["error"])
            console.print("-" * 20)

    console.print(f"\n[bold]Summary: {passed_count}/{len(results)} passed[/bold]")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
