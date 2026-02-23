const PREC = {
  UNARY: 100,
  CALL: 200,
  OP_ID: 150,
};

module.exports = grammar({
  name: 'qxq',

  extras: $ => [
    $.comment,
    /\s/,
  ],

  rules: {
    source_file: $ => optional($._block),

    _block: $ => seq(
      $._expression,
      repeat(seq(';', $._expression))
    ),

    _expression: $ => choice(
      $.binary_expression,
      $.unary_expression,
      $.call_expression,
      $.let_binding,
      $.function_definition,
      $.if_expression,
      $.parenthesized_expression,
      $.tuple_expression,
      $.identifier,
      $.number,
      $.string,
      $.boolean,
      $.operator_identifier,
    ),

    binary_expression: $ => choice(
      prec.right(10, seq(field('left', $._expression), alias(':', $.operator), field('right', $._expression))),
      prec.right(8, seq(field('left', $._expression), alias('@', $.operator), field('right', $._expression))),
      prec.left(6, seq(field('left', $._expression), alias(choice('*', '/'), $.operator), field('right', $._expression))),
      prec.left(4, seq(field('left', $._expression), alias(choice('+', '-'), $.operator), field('right', $._expression))),
      prec.left(2, seq(field('left', $._expression), alias(choice('==', '!=', '<', '>', '<=', '>='), $.operator), field('right', $._expression))),
      prec.left(1, seq(field('left', $._expression), $.operator, field('right', $._expression))),
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
      field('body', $._block),
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
      field('consequence', $._expression),
      'else',
      field('alternative', $._expression),
      'end'
    ),

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_!$%&*+\-/:<=>?@^~]*/,

    operator_identifier: $ => choice(
      prec(PREC.OP_ID, seq('(', alias($.operator, $.operator), ')')),
      prec(PREC.OP_ID, seq('`', alias(/[^`]*/, $.operator), '`')),
    ),

    operator: $ => token(choice(
      /[!$%&*+\-/:<=>?@^~|.#&?~]+/,
      ',' , ';' , ':' , '.' , '?' , '~' , '!' , '$' , '&' , '*' , '+' , '-' , '/' , '=' , '>' , '@' , '^' , '|' , '%' , '<' , '#'
    )),

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

    comment: $ => seq(
      '(*',
      repeat(choice(
        $.comment,
        /[^*]/,
        seq('*', /[^)]/)
      )),
      '*)'
    ),
  }
});

function sepBy(sep, rule) {
  return optional(seq(rule, repeat(seq(sep, rule))));
}
