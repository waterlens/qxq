# qxq Tree-sitter Highlighting

This directory contains the Tree-sitter grammar and highlighting queries for the `qxq` language.

## Development

This project uses [just](https://github.com/casey/just) to manage common tasks.

- **Generate the parser:** `just generate` (runs `tree-sitter generate`)
- **Build the parser:** `just build` (compiles the parser to a shared object)
- **Run tests:** `just test` (runs corpus tests in `test/corpus/`)
- **Clean build artifacts:** `just clean` or `just dist-clean`

## Integration with Neovim

The easiest way to install the highlighting for Neovim is to use the provided automation script.

### Prerequisites

- `tree-sitter-cli`
- `just`
- `uv` (for running the management script)

### Automatic Installation

Run the following command from this directory:

```bash
just install
```

This will:
1.  Generate and build the parser.
2.  Copy the compiled parser (`qxq.so`) to your Neovim site parser directory.
3.  Install highlighting queries to `~/.config/nvim/queries/qxq/highlights.scm`.
4.  Configure filetype detection for `.qxq` files.
5.  (Optional) Create a LazyVim plugin spec if you use LazyVim.

To remove the installation:

```bash
just uninstall
```

### Manual Configuration

If you prefer to manage the configuration yourself, you can add the following to your Neovim setup:

```lua
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
parser_config.qxq = {
  install_info = {
    url = "/path/to/qxq/highlighting", -- Change to your actual path
    files = {"src/parser.c"},
  },
  filetype = "qxq",
}

vim.filetype.add({
  extension = {
    qxq = "qxq",
  },
})
```

## Features

- **Keywords:** `let` (and `rec`), `fn` / `end`, `if` / `then` / `else` / `end`.
- **Nested Comments:** Supports OCaml-style nested comments `(* ... (* ... *) ... *)`.
- **Literals:**
    - Numbers: Hexadecimal (`0x`), Octal (`0o`), Binary (`0b`), and Decimal.
    - Strings: Double-quoted with escape sequence support (`\n`, `\r`, `\t`, `\\`, `\"`).
    - Booleans: `true` and `false`.
- **Call Expressions:** Supports multiple bracket styles: `f(x)`, `f[x]`, and `f{x}`.
- **Tuples & Parentheses:** `(a, b, c)` and `(expression)`.
- **Operators:**
    - Flexible operator naming (e.g., `>>=`, `<+>`, `-->`).
    - Dedicated precedence for `:`, `@`, `*`/`/`, `+`/`-`, and comparison operators.
- **Operator Identifiers:** Wrap operators in parentheses `(+)` or backticks `` `raw op` `` to use them as identifiers.
