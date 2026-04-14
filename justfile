# Default target
default:
    @just --list

# Build the project
build:
    @cargo build

# Build the project in release mode
release:
    @cargo build --release

# Run the REPL
repl:
    @cargo run

# Run traditional tests (exit code based)
test *args:
    @uv run scripts/run_tests.py --test {{args}}

# Run only expectation tests ((* RUN: *) based)
test-expect *args:
    @uv run scripts/run_tests.py --test-expect {{args}}

# Update EXPECT: blocks and skip those with multiple blocks
update-expect *args:
    @uv run scripts/run_tests.py --update-expect --skip-multiple-expect {{args}}

# Run a single expect test file directly
expect file:
    @cargo run -- --test-expect {{file}}

unit-test:
    @cargo test

# Run all tests
test-all: unit-test test (test "--release") test-expect (test-expect "--release")

# Install tree-sitter highlighter
highlight-install:
    @just -f highlighting/justfile install

# Uninstall tree-sitter highlighter
highlight-uninstall:
    @just -f highlighting/justfile uninstall

