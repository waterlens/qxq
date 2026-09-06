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

# Run all tests: the Rust unit tests and every tests/**/*.qxq file
test *args:
    @cargo test {{args}}

# Update the EXPECT: block of every test file, skipping files with multiple blocks
update-expect:
    @cargo run -- --update-expect tests --skip-multiple-expect

# Run a single expect test file directly
expect file:
    @cargo run -- --test-expect {{file}}

# Run only the unit tests in the Rust sources
unit-test:
    @cargo test --lib --bins

# Run all tests in debug and release mode
test-all:
    @cargo test
    @cargo test --release

# Install tree-sitter highlighter
highlight-install:
    @just -f highlighting/justfile install

# Uninstall tree-sitter highlighter
highlight-uninstall:
    @just -f highlighting/justfile uninstall
