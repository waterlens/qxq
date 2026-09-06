# QxQ Workflow

## Project Structure

- Rust compiler: `./`
- C VM (submodule): `./vm/`
- VM configuration macros, one file per architecture: `./vm/flags/<arch>` (read by `vm/Makefile`, `build.rs` and `vm/scripts/asm.py`)

## Primary Commands

- List tasks: `just`
- Build debug: `just build`
- Build release: `just release`
- Start REPL: `just repl`
- Run unit tests: `just unit-test`
- Run all tests (unit tests and `tests/**/*.qxq`): `just test` or `cargo test`
- Run the full suite in debug and release mode: `just test-all`
- Benchmark the release binary: `just bench`
- Update `EXPECT:` blocks: `just update-expect`
- Run one expect test file: `just expect tests/path/to/file.qxq`
- Build the vm binary: `make -C vm rel`
- Regenerate the review-only vm assembly dumps (`vm/asm/vm.<target>.s`, needs zig 0.15.1 on PATH): `make -C vm asm`
- Check the dumps are current: `make -C vm asm-check`
- One-time per clone, so dump diffs show the handler in hunk headers: `git -C vm config diff.asm.xfuncname '^_?[A-Za-z0-9_]+:'`

## Direct Cargo Usage

- Run a source file: `cargo run -- path/to/file.qxq`
- Print bytecode only: `cargo run -- path/to/file.qxq --no-tree`
- Dump bytecode to a file (`-` for stdout): `cargo run -- path/to/file.qxq --dump out.qxc`
- Load bytecode from a file (`-` for stdin): `cargo run -- --load out.qxc`
- Run one expect file directly: `cargo run -- --test-expect tests/path/to/file.qxq`
- Check stdin against an expect file: `some_command | cargo run -- --check-expect tests/path/to/file.qxq`
- Update one expect file directly: `cargo run -- --update-expect tests/path/to/file.qxq`

## Test Workflow

- `cargo test` (or `just test`) is the single entry point: it runs the Rust unit tests and, through `tests/filetests.rs`, every `tests/**/*.qxq` file as one test named by its path relative to `tests/`.
- `cargo test --release` does the same with the release build; `just test-all` runs both.
- Files with a `(* RUN: ... *)` or `(* RUN-EXPECT-ERROR: ... *)` block run through `qxq --test-expect`; all other files must succeed under `qxq --inspect`.
- Filter by name as with any cargo test, e.g. `cargo test -- 6_execution`; `just unit-test` runs only the Rust unit tests.
- A file containing `(* SKIP: reason *)` is reported as ignored, with the reason in the test name; `qxq` itself refuses such a file with `skipped: reason` and exit code 2.
- `just update-expect` rewrites the single `EXPECT:` block of every `tests/**/*.qxq` file from current command output, skipping multi-block and SKIP files; `--update-expect` takes a directory or a single file.
- Expect files support `%s` for the current file path and `{preferred|fallback}` command selection.
- `just bench` builds release and runs `hyperfine` on `bench/*.qxq`; `just bench path/to/other/qxq` compares another binary.

## Git Workflow

- Match the existing commit message style: usually a short lowercase subject.
- Commit `vm` submodule changes before committing the parent `qxq` repo.

## Common Loop

1. Edit Rust compiler code.
2. Run `just test-all` for all kinds of tests.
3. Run `just update-expect` after intentional output changes.
4. Run `make -C vm asm` after changing `vm/src/vm.c` and commit the updated `vm/asm/*.s` with the vm change (`make -C vm asm-check` must pass).

## Code Style

- Follow existing naming conventions
- Match the surrounding style before refactoring:
  keep simple control flow, early checks, and macro-oriented C code straightforward rather than abstracting it prematurely.
