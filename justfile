# Default target
default:
    @just --list

# Run the test suite (use --release for release mode)
test *args:
    @uv run scripts/run_tests.py {{args}}

# Build the project
build:
    @cargo build

# Run the REPL
repl:
    @cargo run

