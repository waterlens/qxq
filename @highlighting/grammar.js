const PREC = {
  BINARY: 1,
  UNARY: 2,
  CALL: 3,
};

module.exports = grammar({
  name: 'qxq',

  extras: $ => [
    $.comment,
    /\s/,
  ],

  rules: {
    source_file: $ => repeat($._expression_with_separator),

    _expression_with_separator: $ => seq(
      $._expression,
      optional(';')
    ),

    _expression: $ => choice(
      $.identifier,
      $.number,
      $.string,
      $.boolean,
      $.let_binding,
      $.function_definition,
      $.if_expression,
      $.binary_expression,
      $.unary_expression,
      $.call_expression,
      $.parenthesized_expression,
      $.tuple_expression,
      $.operator_identifier,
    ),

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_!$%&*+\-/:<=>?@^~]*/,

    operator_identifier: $ => choice(
      seq('(', $.operator, ')'),
      seq('`', /[^`]*/, '`')
    ),

    operator: $ => token(/[!$%&*+\-/:<=>?@^~|.#&?~]+/),

    number: $ => {
      const hex = /0[xX][0-9a-fA-F]+/;
      const octal = /0[oO][0-7]+/;
      const binary = /0[bB][01]+/;
      const decimal = /[0-9]+/;
      return token(seq(
        optional(choice('+', '-')),
        choice(hex, octal, binary, decimal)
      ));
    },

    string: $ => seq(
      '"',
      repeat(choice(
        /[^"\\\n]+/,
        $.escape_sequence
      )),
      '"'
    ),

    escape_sequence: $ => token(seq(
      '\\',
      /[nrt\\"]/
    )),

    boolean: $ => choice('true', 'false'),

    let_binding: $ => seq(
      'let',
      optional('rec'),
      field('name', $.identifier),
      '=',
      field('value', $._expression)
    ),

    function_definition: $ => seq(
      'fn',
      field('parameters', $.parameters),
      field('body', repeat($._expression_with_separator)),
      'end'
    ),

    parameters: $ => seq(
      '(',
      sepBy(',', $.identifier),
      ')'
    ),

    if_expression: $ => seq(
      'if',
      field('condition', $._expression),
      'then',
      field('consequence', repeat($._expression_with_separator)),
      'else',
      field('alternative', repeat($._expression_with_separator)),
      'end'
    ),

    binary_expression: $ => choice(
      prec.left(PREC.BINARY, seq($._expression, $.operator, $._expression)),
      prec.left(PREC.BINARY, seq($._expression, '@', $._expression)),
      prec.left(PREC.BINARY, seq($._expression, ':', $._expression)),
    ),

    unary_expression: $ => prec(PREC.UNARY, seq($.operator, $._expression)),

    call_expression: $ => prec.left(PREC.CALL, seq(
      $._expression,
      choice(
        seq('(', sepBy(',', $._expression), ')'),
        seq('[', sepBy(',', $._expression), ']'),
        seq('{', sepBy(',', $._expression), '}'),
      )
    )),

    parenthesized_expression: $ => seq(
      '(',
      $._expression,
      ')'
    ),

    tuple_expression: $ => seq(
      '(',
      $._expression,
      repeat1(seq(',', $._expression)),
      ')'
    ),

    comment: $ => seq(
      '(*',
      repeat(choice(
        $.comment,
        /[^*]+/,
        /\*[^)]/
      )),
      '*)'
    ),
  }
});

function sepBy(sep, rule) {
  return optional(seq(rule, repeat(seq(sep, rule))));
}
