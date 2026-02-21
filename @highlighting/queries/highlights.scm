; Keywords
"let" @keyword
"rec" @keyword
"fn" @keyword
"end" @keyword
"if" @keyword
"then" @keyword
"else" @keyword

; Literals
(number) @number
(string) @string
(boolean) @boolean
(escape_sequence) @string.escape

; Comments
(comment) @comment

; Identifiers
(identifier) @variable
(let_binding name: (identifier) @variable.definition)
(function_definition (parameters (identifier) @variable.parameter))
(call_expression (identifier) @function.call)

; Operators
(operator) @operator
"@" @operator
":" @operator
"=" @operator
";" @punctuation.delimiter
"," @punctuation.delimiter

; Brackets
"(" @punctuation.bracket
")" @punctuation.bracket
"[" @punctuation.bracket
"]" @punctuation.bracket
"{" @punctuation.bracket
"}" @punctuation.bracket
