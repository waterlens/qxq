# qxq Tree-sitter Highlighting

This directory contains the Tree-sitter grammar and highlighting queries for the `qxq` language.

## Integration with Neovim

To use this highlighting in Neovim, follow these steps:

### 1. Install Tree-sitter CLI (if not already installed)

You may need to compile the grammar.
```bash
npm install -g tree-sitter-cli
cd @highlighting
tree-sitter generate
```

### 2. Configure Neovim

Add the following to your Neovim configuration (e.g., `init.lua`):

```lua
-- 1. Register the qxq parser
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
parser_config.qxq = {
  install_info = {
    url = "/absolute/path/to/your/project/@highlighting", -- Change this to your actual path
    files = {"src/parser.c"},
    branch = "main",
  },
  filetype = "qxq",
}

-- 2. Associate .qxq files with the qxq filetype
vim.filetype.add({
  extension = {
    qxq = "qxq",
  },
})

-- 3. Ensure the queries are loaded
-- Link or copy @highlighting/queries/highlights.scm to 
-- ~/.config/nvim/queries/qxq/highlights.scm
-- or add the @highlighting directory to your runtimepath:
vim.opt.rtp:append("/absolute/path/to/your/project/@highlighting")
```

### 3. Load the parser

Restart Neovim and run `:TSInstall qxq`.

## Features
- Keyword highlighting (`let`, `fn`, `if`, etc.)
- Nested comment support `(* ... (* ... *) ... *)`
- Various number formats (hex, oct, bin)
- Operator and identifier highlighting
- Function call identification
