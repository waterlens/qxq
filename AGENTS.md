# QxQ Workflow

## Project Structure

- Rust compiler: `./`
- C VM (submodule): `./vm/`

## Primary Commands

- List tasks: `just`
- Build debug: `just build`
- Build release: `just release`
- Start REPL: `just repl`
- Run unit tests: `just unit-test`
- Run all non-expect tests: `just test`
- Run all expect tests: `just test-expect`
- Run the full suite: `just test-all`
- Update `EXPECT:` blocks: `just update-expect`
- Run one expect test file: `just expect tests/path/to/file.qxq`
- Compile vm submodule and update `vm.s`: `make -C vm rel`

## Direct Cargo Usage

- Run a source file: `cargo run -- path/to/file.qxq`
- Print bytecode only: `cargo run -- path/to/file.qxq --no-tree`
- Dump bytecode to a file (`-` for stdout): `cargo run -- path/to/file.qxq --dump out.qxc`
- Load bytecode from a file (`-` for stdin): `cargo run -- --load out.qxc`
- Run one expect file directly: `cargo run -- --test-expect tests/path/to/file.qxq`
- Check stdin against an expect file: `some_command | cargo run -- --check-expect tests/path/to/file.qxq`
- Update one expect file directly: `cargo run -- --update-expect tests/path/to/file.qxq`

## Test Workflow

- `just unit-test` runs units tests in Rust source files.
- `just test` runs every `tests/**/*.qxq` file in traditional mode through `scripts/run_tests.py`.
- `just test-expect` runs only files that contain `(* RUN: ... *)` or `(* RUN-EXPECT-ERROR: ... *)`.
- `just update-expect` rewrites a single `EXPECT:` block from current command output and skips multi-block files.
- Expect files support `%s` for the current file path and `{preferred|fallback}` command selection.

## Git Workflow

- Match the existing commit message style: usually a short lowercase subject.
- Commit `vm` submodule changes before committing the parent `qxq` repo.

## Common Loop

1. Edit Rust compiler code.
2. Run `just test-all` for all kinds of tests.
3. Run `just update-expect` after intentional output changes.
4. Run `make -C vm rel` to update the change in vm to `vm.s` if needed.

## Code Style

- Follow existing naming conventions
- Match the surrounding style before refactoring:
  keep simple control flow, early checks, and macro-oriented C code straightforward rather than abstracting it prematurely.
