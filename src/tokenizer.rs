use anyhow::Result;
use small_map::ASmallMap;

use std::{
  cell::OnceCell,
  hash::{Hash, Hasher},
};

#[derive(Debug, Clone, Copy)]
pub struct Loc {
  pub line: u32,
  pub column: u32,
  pub index: usize,
}

impl Loc {
  pub fn invalid_loc() -> Self {
    Loc { line: u32::MAX, column: u32::MAX, index: usize::MAX }
  }

  pub fn is_valid(&self) -> bool {
    self.line != u32::MAX && self.column != u32::MAX && self.index != usize::MAX
  }

  pub fn new(line: u32, column: u32, index: usize) -> Self {
    Loc { line, column, index }
  }
}

impl std::fmt::Display for Loc {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let index = self.index;
    let line = self.line;
    let column = self.column;
    write!(f, "{line}:{column}[{index}]")
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenStr<'a>(pub &'a str);
impl<'a> TokenStr<'a> {
  pub fn new(s: &'a str) -> Self {
    TokenStr(s)
  }
  pub fn from_span<'arena>(arena: &'arena bumpalo::Bump, span: TokenSpan<'a>) -> TokenStr<'arena> {
    TokenStr::new(arena.alloc(span.to_string()))
  }
}

impl Hash for TokenStr<'_> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.0.hash(state)
  }
}

impl std::fmt::Display for TokenStr<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl AsRef<str> for TokenStr<'_> {
  fn as_ref(&self) -> &str {
    self.0
  }
}

impl<'a> From<&'a str> for TokenStr<'a> {
  fn from(s: &'a str) -> Self {
    TokenStr::new(s)
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan<'a>(&'a [char]);

impl<'a> TokenSpan<'a> {
  const EMPTY: TokenSpan<'static> = TokenSpan(&['<', 'e', 'm', 'p', 't', 'y', '>']);
  pub fn new(s: &'a [char]) -> Self {
    TokenSpan(s)
  }
  pub const fn empty() -> Self {
    Self::EMPTY
  }
  pub fn is_empty(&self) -> bool {
    self.0 == Self::EMPTY.0
  }
}

impl ToString for TokenSpan<'_> {
  fn to_string(&self) -> String {
    self.0.iter().collect()
  }
}

impl AsRef<[char]> for TokenSpan<'_> {
  fn as_ref(&self) -> &[char] {
    self.0
  }
}

impl<'a> From<&'a [char]> for TokenSpan<'a> {
  fn from(s: &'a [char]) -> Self {
    TokenSpan::new(s)
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenTag<'a> {
  Eof,
  Newline,
  Identifer,
  Kw(Keyword),
  Op(&'a str),
  RawOp(&'a str),
  StrLiteral(&'a str),
  IntLiteral(i128),
  PairedOpen(Paired),
  PairedClose(Paired),
  Error(TokenErr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenErr {
  EarlyEof,
  NewlineInLiteral,
  InvalidEscapeSequence,
  InvalidIntLiteralPrefix,
  InvalidIntLiteralDigit,
  SymbolLikeOperatorFollowedByNonSpace,
  InvalidUnaryOperator,
  InvalidBinaryOperator,
  UnexpectedChar,
}

impl std::fmt::Display for TokenErr {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use TokenErr::*;
    match self {
      EarlyEof => write!(f, "unexpected end of file"),
      NewlineInLiteral => write!(f, "newline in literal"),
      InvalidEscapeSequence => write!(f, "invalid escape sequence"),
      InvalidIntLiteralPrefix => write!(f, "invalid integer literal prefix"),
      InvalidIntLiteralDigit => write!(f, "invalid integer literal digit"),
      SymbolLikeOperatorFollowedByNonSpace => {
        write!(f, "symbol-like operator followed by non-space")
      }
      InvalidUnaryOperator => write!(f, "invalid unary operator"),
      InvalidBinaryOperator => write!(f, "invalid binary operator"),
      UnexpectedChar => write!(f, "unexpected character"),
    }
  }
}

impl std::error::Error for TokenErr {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
  Fn,
  Let,
  Rec,
  With,
  And,
  Is,
  If,
  Else,
  Then,
  End,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Paired {
  Parenthesis,
  Bracket,
  Brace,
}

impl std::fmt::Display for Paired {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use Paired::*;
    match self {
      Parenthesis => write!(f, "()"),
      Bracket => write!(f, "[]"),
      Brace => write!(f, "{{}}"),
    }
  }
}

impl std::fmt::Display for Keyword {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use Keyword::*;
    match self {
      Fn => write!(f, "fn"),
      Let => write!(f, "let"),
      Rec => write!(f, "rec"),
      With => write!(f, "with"),
      And => write!(f, "and"),
      Is => write!(f, "is"),
      If => write!(f, "if"),
      Else => write!(f, "else"),
      Then => write!(f, "then"),
      End => write!(f, "end"),
    }
  }
}

impl std::fmt::Display for Token<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.tag {
      Eof => write!(f, "<end of file>"),
      Newline => write!(f, "<new line>"),
      Identifer => write!(f, "<identifier {}>", self.span.0.iter().collect::<String>()),
      Kw(kw) => write!(f, "<keyword {kw}>"),
      Op(op) => write!(f, "<op {op}>"),
      RawOp(op) => write!(f, "<raw op: `{op}`>"),
      StrLiteral(s) => write!(f, "<string literal {s:?}>"),
      IntLiteral(i) => write!(f, "<string literal {i}>"),
      PairedOpen(po) => write!(f, "<paired open {po}>"),
      PairedClose(pc) => write!(f, "<paired close {pc}>"),
      Error(err) => write!(f, "<error {err}>"),
    }?;
    write!(f, " {}", self.loc)
  }
}

#[derive(Debug, Clone)]
pub struct Token<'a> {
  pub tag: TokenTag<'a>,
  pub span: TokenSpan<'a>,
  pub loc: Loc,
}

impl<'a> Token<'a> {
  pub fn new(tag: TokenTag<'a>, span: TokenSpan<'a>, loc: Loc) -> Self {
    Token { tag, span, loc }
  }
}

use bumpalo::Bump;

pub struct Tokenizer<'a> {
  arena: &'a bumpalo::Bump,
  buffer: &'a [char],
  index: usize,
  colstart: usize,
  line: u32,
}

use once_cell::sync::Lazy;
use Paired::*;
use TokenErr::*;
use TokenTag::*;

impl<'a, 'b> Tokenizer<'a>
where
  'a: 'b,
{
  const KEYWORDS: Lazy<OnceCell<ASmallMap<16, &'static [char], Keyword>>> = Lazy::new(|| {
    let cell = OnceCell::new();
    let _ = cell.set({
      let mut map: ASmallMap<16, &'static [char], Keyword> = ASmallMap::new();
      map.insert(&['f', 'n'], Keyword::Fn);
      map.insert(&['l', 'e', 't'], Keyword::Let);
      map.insert(&['r', 'e', 'c'], Keyword::Rec);
      map.insert(&['w', 'i', 't', 'h'], Keyword::With);
      map.insert(&['a', 'n', 'd'], Keyword::And);
      map.insert(&['i', 's'], Keyword::Is);
      map.insert(&['i', 'f'], Keyword::If);
      map.insert(&['e', 'l', 's', 'e'], Keyword::Else);
      map.insert(&['t', 'h', 'e', 'n'], Keyword::Then);
      map.insert(&['e', 'n', 'd'], Keyword::End);
      map
    });
    cell
  });

  pub fn new<I>(arena: &'a Bump, input: I) -> Self
  where
    I: AsRef<str>,
  {
    let i = input.as_ref();
    let buffer = arena.alloc({
      let mut buffer = i.chars().collect::<Vec<_>>();
      if buffer.last() != Some(&'\0') {
        buffer.push('\0');
      }
      buffer.into_boxed_slice()
    });
    Tokenizer { arena, buffer, index: 0, colstart: 0, line: 1 }
  }

  fn move_to_newline_begin(&mut self) {
    self.colstart = self.index;
    self.line += 1;
  }

  fn get_loc(&self) -> Loc {
    Loc::new(self.line, (self.index - self.colstart + 1) as u32, self.index)
  }

  fn make_span(&'b self, start: usize) -> TokenSpan<'a> {
    TokenSpan::new(&self.buffer[start..self.index])
  }

  fn make_token(&'b self, tag: TokenTag<'a>, loc: Loc) -> Token<'a> {
    Token::new(tag, self.make_span(loc.index), loc)
  }

  fn make_error(&'b self, err: TokenErr, loc: Loc) -> Token<'a> {
    Token::new(Error(err), self.make_span(loc.index), loc)
  }

  fn eof(&'b self) -> Token<'a> {
    Token::new(
      Eof,
      TokenSpan::empty(),
      Loc::new(self.line, (self.index - self.colstart) as u32, self.index),
    )
  }

  fn is_keyword(&self, s: &[char]) -> Option<Keyword> {
    Self::KEYWORDS.get().unwrap().get(s).cloned()
  }

  fn ch(&self) -> char {
    self.buffer[self.index]
  }
  fn ch_at(&self, i: usize) -> char {
    self.buffer[i]
  }

  fn string_literal(&'b mut self, loc: Loc) -> Token<'a> {
    self.index += 1;
    let mut buf = String::new();
    loop {
      match self.ch() {
        '\0' => return self.make_error(EarlyEof, loc),
        '\n' => {
          self.index += 1;
          self.move_to_newline_begin();
          return self.make_error(NewlineInLiteral, loc);
        }
        '"' => break,
        '\\' => {
          self.index += 1;
          match self.ch() {
            '\0' => return self.make_error(EarlyEof, loc),
            'n' | 'r' | 't' | '\\' | '"' => buf.push(match self.ch() {
              'n' => '\n',
              'r' => '\r',
              't' => '\t',
              _ => self.ch(),
            }),
            _ => {
              self.index += 1;
              return self.make_error(InvalidEscapeSequence, loc);
            }
          }
        }
        _ => {
          buf.push(self.ch());
        }
      }
      self.index += 1;
    }
    self.index += 1;
    self.make_token(TokenTag::StrLiteral(self.arena.alloc_str(&buf)), loc)
  }

  fn integer_literal(&'b mut self, loc: Loc, first_c: char, neg: bool) -> Token<'a> {
    self.index += 1;
    let mut base: u32 = 10;
    let mut value: i128 = first_c.to_digit(10).unwrap().into();
    if neg {
      value = -value;
    }
    if first_c == '0' {
      match self.ch() {
        'b' | 'B' => {
          self.index += 1;
          base = 2;
        }
        'o' | 'O' => {
          self.index += 1;
          base = 8;
        }
        'x' | 'X' => {
          self.index += 1;
          base = 16;
        }
        '0'..='9' => {}
        _ => return self.make_token(IntLiteral(value), loc),
      }
    }
    if base != 10 {
      match self.ch() {
        '0'..='9' | 'a'..='f' | 'A'..='F' => {}
        _ => return self.make_error(InvalidIntLiteralPrefix, loc),
      }
    }
    loop {
      let n = match self.ch() {
        '0'..='9' | 'a'..='f' | 'A'..='F' => match self.ch().to_digit(base) {
          Some(n) => n as i128,
          None => {
            self.index += 1;
            return self.make_error(InvalidIntLiteralDigit, loc);
          }
        },
        _ => break,
      };
      value *= base as i128;
      value = if neg { value - n } else { value + n };
      self.index += 1;
    }
    self.make_token(IntLiteral(value), loc)
  }

  fn ident(&'b mut self, loc: Loc) -> Token<'a> {
    self.index += 1;
    let mut symbol_like = false;
    loop {
      match self.ch() {
        'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => (),
        '!' | '$' | '%' | '&' | '*' | '+' | '-' | '/' | ':' | '<' | '=' | '>' | '?' | '@' | '^'
        | '~' => symbol_like |= true,
        '\0' | ' ' | '\t' | '\r' | '\n' => break,
        ')' | ']' | '}' | ',' | ';' => break,
        _ => {
          if !symbol_like {
            break;
          } else {
            return self.make_error(SymbolLikeOperatorFollowedByNonSpace, self.get_loc());
          }
        }
      }
      self.index += 1;
    }

    let tok = self.make_token(Identifer, loc);
    if let Some(kw) = self.is_keyword(tok.span.as_ref()) {
      return self.make_token(Kw(kw), loc);
    }
    tok
  }

  fn skip_comment(&'b mut self) -> bool {
    self.index += 1;
    let mut level = 1;
    loop {
      match self.ch() {
        '(' => {
          self.index += 1;
          if self.ch() == '*' {
            self.index += 1;
            level += 1;
          }
        }
        '*' => {
          self.index += 1;
          if self.ch() == ')' {
            self.index += 1;
            level -= 1;
            if level == 0 {
              return true;
            }
          }
        }
        '\0' => return false,
        '\n' => {
          self.index += 1;
          self.move_to_newline_begin();
        }
        _ => self.index += 1,
      }
    }
  }

  fn operator(&'b mut self, loc: Loc, first_c: char) -> Token<'a> {
    self.index += 1;
    let mut buf = first_c.to_string();
    match first_c {
      ',' | ';' | ':' | '.' => return self.make_token(Op(self.arena.alloc_str(&buf)), loc),
      '?' | '~' | '!' => {
        match self.ch() {
          '$' | '&' | '*' | '+' | '-' | '/' | '=' | '>' | '@' | '^' | '|' | '%' | '<' => {}
          _ => {
            if first_c != '!' {
              return self.make_error(InvalidUnaryOperator, self.get_loc());
            }
          }
        }
        while matches!(
          self.ch(),
          '$' | '&' | '*' | '+' | '-' | '/' | '=' | '>' | '@' | '^' | '|' | '%' | '<'
        ) {
          buf.push(self.ch());
          self.index += 1;
        }
      }
      '$' | '&' | '*' | '+' | '-' | '/' | '=' | '>' | '@' | '^' | '|' | '%' | '<' | '#' => {
        match self.ch() {
          '$' | '&' | '*' | '+' | '-' | '/' | '=' | '>' | '@' | '^' | '|' | '%' | '<' | '!'
          | '.' | ':' | '?' | '~' => {}
          _ => {
            if first_c == '#' {
              return self.make_error(InvalidBinaryOperator, self.get_loc());
            }
          }
        }
        while matches!(
          self.ch(),
          '$'
            | '&'
            | '*'
            | '+'
            | '-'
            | '/'
            | '='
            | '>'
            | '@'
            | '^'
            | '|'
            | '%'
            | '<'
            | '!'
            | '.'
            | ':'
            | '?'
            | '~'
        ) {
          buf.push(self.ch());
          self.index += 1;
        }
      }
      '`' => {
        buf.clear();
        loop {
          match self.ch() {
            '`' => {
              let mut loc = loc;
              loc.index += 1;
              let tok = self.make_token(RawOp(self.arena.alloc_str(&buf)), loc);
              self.index += 1;
              return tok;
            }
            '\0' => return self.make_error(EarlyEof, self.get_loc()),
            _ => buf.push(self.ch()),
          }
          self.index += 1;
        }
      }
      _ => unreachable!(),
    }
    self.make_token(Op(self.arena.alloc_str(&buf)), loc)
  }

  pub fn next_with_err(&'b mut self) -> Result<Token<'a>> {
    let tok = self.next();
    if let TokenTag::Error(err) = tok.tag {
      Result::Err(anyhow::anyhow!(err))
    } else {
      Ok(tok)
    }
  }

  pub fn next(&'b mut self) -> Token<'a> {
    loop {
      let c = self.ch();
      let loc = self.get_loc();
      match c {
        '\0' => return self.eof(),
        ' ' | '\t' | '\r' => {
          self.index += 1;
          loop {
            match self.ch() {
              ' ' | '\t' | '\r' => self.index += 1,
              _ => break,
            }
          }
          continue;
        }
        '\n' => {
          self.index += 1;
          self.move_to_newline_begin();
          return self.make_token(Newline, loc);
        }
        '(' => {
          self.index += 1;
          if self.ch() == '*' {
            self.index += 1;
            if !self.skip_comment() {
              return self.make_error(EarlyEof, loc);
            }
          } else {
            return self.make_token(PairedOpen(Parenthesis), loc);
          }
        }
        ')' => {
          self.index += 1;
          return self.make_token(PairedClose(Parenthesis), loc);
        }
        '[' => {
          self.index += 1;
          return self.make_token(PairedOpen(Bracket), loc);
        }
        ']' => {
          self.index += 1;
          return self.make_token(PairedClose(Bracket), loc);
        }
        '{' => {
          self.index += 1;
          return self.make_token(PairedOpen(Brace), loc);
        }
        '}' => {
          self.index += 1;
          return self.make_token(PairedClose(Brace), loc);
        }
        '"' => return self.string_literal(loc),
        '0'..='9' => return self.integer_literal(loc, c, false),
        '+' | '-' => {
          let next_c = self.ch_at(self.index + 1);
          if next_c.is_ascii_digit() {
            self.index += 1;
            return self.integer_literal(loc, next_c, c == '-');
          }
          return self.operator(loc, c);
        }
        'a'..='z' | 'A'..='Z' | '_' => return self.ident(loc),
        ',' | ';' | ':' | '.' | '?' | '~' | '!' | '$' | '&' | '*' | '/' | '=' | '>' | '@' | '^'
        | '|' | '%' | '<' | '#' | '`' => {
          return self.operator(loc, c);
        }
        _ => return self.make_error(UnexpectedChar, loc),
      }
    }
  }
}

#[cfg(test)]
mod tests {
  pub use super::*;

  struct TestToken<'a> {
    tag: TokenTag<'a>,
    span: &'static str,
  }

  fn t<'a>(tag: TokenTag<'a>, span: &'static str) -> TestToken<'a> {
    TestToken { tag, span }
  }

  fn test_tokenize(input: &str, expected: &[TestToken]) {
    let mut arena = Bump::new();
    let mut tokenizer = Tokenizer::new(&mut arena, input);
    for expected_token in expected {
      let token = tokenizer.next();
      assert_eq!(token.tag, expected_token.tag);
      assert_eq!(token.span.to_string().as_str(), expected_token.span);
    }
    let last_token = tokenizer.next();
    assert_eq!(last_token.tag, TokenTag::Eof);
  }

  #[test]
  fn test_keywords() {
    test_tokenize("fn", &[t(Kw(Keyword::Fn), "fn")]);
    test_tokenize("let", &[t(Kw(Keyword::Let), "let")]);
    test_tokenize("rec", &[t(Kw(Keyword::Rec), "rec")]);
    test_tokenize("with", &[t(Kw(Keyword::With), "with")]);
    test_tokenize("if", &[t(Kw(Keyword::If), "if")]);
    test_tokenize("else", &[t(Kw(Keyword::Else), "else")]);
    test_tokenize("then", &[t(Kw(Keyword::Then), "then")]);
    test_tokenize("end", &[t(Kw(Keyword::End), "end")]);
  }

  #[test]
  fn test_operators() {
    test_tokenize("+", &[t(Op("+"), "+")]);
    test_tokenize("-", &[t(Op("-"), "-")]);
    test_tokenize("*", &[t(Op("*"), "*")]);
    test_tokenize("/", &[t(Op("/"), "/")]);
    test_tokenize("=", &[t(Op("="), "=")]);
    test_tokenize(">", &[t(Op(">"), ">")]);
    test_tokenize("<", &[t(Op("<"), "<")]);
    test_tokenize("!", &[t(Op("!"), "!")]);
    test_tokenize("&", &[t(Op("&"), "&")]);
    test_tokenize("|", &[t(Op("|"), "|")]);
    test_tokenize("^", &[t(Op("^"), "^")]);
    test_tokenize("@", &[t(Op("@"), "@")]);
    test_tokenize("!", &[t(Op("!"), "!")]);
    test_tokenize(":", &[t(Op(":"), ":")]);
    test_tokenize("#+", &[t(Op("#+"), "#+")]);
    test_tokenize("#-", &[t(Op("#-"), "#-")]);
    test_tokenize("?++", &[t(Op("?++"), "?++")]);
    test_tokenize("~--", &[t(Op("~--"), "~--")]);
    test_tokenize("`raw operator`", &[t(RawOp("raw operator"), "raw operator")]);
    test_tokenize("-->", &[t(Op("-->"), "-->")]);
    test_tokenize("->", &[t(Op("->"), "->")]);
  }

  #[test]
  fn test_identifier() {
    test_tokenize("a", &[t(Identifer, "a")]);
    test_tokenize("a0", &[t(Identifer, "a0")]);
    test_tokenize("a0_", &[t(Identifer, "a0_")]);
    test_tokenize("a0_!$%&*+-/:<=>?@^_~", &[t(Identifer, "a0_!$%&*+-/:<=>?@^_~")]);
    test_tokenize("a0_!$%&*+-/:<=>?@^_~b", &[t(Identifer, "a0_!$%&*+-/:<=>?@^_~b")]);
    test_tokenize("a0_!$%&*+-/:<=>?@^_~b ", &[t(Identifer, "a0_!$%&*+-/:<=>?@^_~b")]);
    test_tokenize(
      "a0_!$%&*+-/:<=>?@^_~b\n",
      &[t(Identifer, "a0_!$%&*+-/:<=>?@^_~b"), t(Newline, "\n")],
    );
    test_tokenize(
      "a0_!$%&*+-/:<=>?@^_~b\n ",
      &[t(Identifer, "a0_!$%&*+-/:<=>?@^_~b"), t(Newline, "\n")],
    );
  }

  #[test]
  fn test_integer_literals() {
    test_tokenize("0xff", &[t(IntLiteral(255), "0xff")]);
    test_tokenize("0o77", &[t(IntLiteral(63), "0o77")]);
    test_tokenize("0b11", &[t(IntLiteral(3), "0b11")]);
    test_tokenize("0", &[t(IntLiteral(0), "0")]);
    test_tokenize("1", &[t(IntLiteral(1), "1")]);
    test_tokenize("01234", &[t(IntLiteral(1234), "01234")]);
    test_tokenize("1234", &[t(IntLiteral(1234), "1234")]);
    test_tokenize("0x0", &[t(IntLiteral(0), "0x0")]);
    test_tokenize(
      "+0xffffffffffffffff",
      &[t(IntLiteral(0xffffffffffffffff), "+0xffffffffffffffff")],
    );
    test_tokenize("-1", &[t(IntLiteral(-1), "-1")]);
    test_tokenize(
      "-9223372036854775808",
      &[t(IntLiteral(-9223372036854775808), "-9223372036854775808")],
    );
    test_tokenize(
      "9223372036854775808",
      &[t(IntLiteral(9223372036854775808), "9223372036854775808")],
    );
    test_tokenize("0x", &[t(Error(InvalidIntLiteralPrefix), "0x")]);
    test_tokenize("0b2", &[t(Error(InvalidIntLiteralDigit), "0b2")]);
    test_tokenize("0o8", &[t(Error(InvalidIntLiteralDigit), "0o8")]);
  }

  #[test]
  fn test_string_literals() {
    test_tokenize("\"\"", &[t(StrLiteral(""), "\"\"")]);
    test_tokenize("\"a\"", &[t(StrLiteral("a"), "\"a\"")]);
    test_tokenize("\"\\n\"", &[t(StrLiteral("\n"), "\"\\n\"")]);
    test_tokenize("\"\\\\\"", &[t(StrLiteral("\\"), "\"\\\\\"")]);
    test_tokenize("\"\\", &[t(Error(EarlyEof), "\"\\")]);
    test_tokenize("\"\\a", &[t(Error(InvalidEscapeSequence), "\"\\a")]);
    test_tokenize("\"\\\"", &[t(Error(EarlyEof), "\"\\\"")]);
    test_tokenize("\"", &[t(Error(EarlyEof), "\"")]);
    test_tokenize("\"a", &[t(Error(EarlyEof), "\"a")]);
  }

  #[test]
  fn test_comments() {
    test_tokenize("(* *)\n", &[t(Newline, "\n")]);
    test_tokenize("(* (* *)\n", &[t(Error(EarlyEof), "(* (* *)\n")]);
    test_tokenize("(* (* *) *)\n", &[t(Newline, "\n")]);
  }
}
