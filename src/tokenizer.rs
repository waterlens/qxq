use crate::bytecode::FloatBits;
use crate::diagnostic::{Diagnostic, Result};
use std::rc::Rc;

use std::fmt;
use std::fmt::Display;
use std::hash::{Hash, Hasher};

/// A source position; `column` and `index` count bytes.
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
  pub fn from_span(span: TokenSpan<'a>) -> Self {
    TokenStr::new(span.0)
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

/// The source text of a token, a slice of the tokenizer's buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan<'a>(&'a str);

impl<'a> TokenSpan<'a> {
  const EMPTY: TokenSpan<'static> = TokenSpan("<empty>");
  pub fn new(s: &'a str) -> Self {
    TokenSpan(s)
  }
  pub const fn empty() -> Self {
    Self::EMPTY
  }
  pub fn is_empty(&self) -> bool {
    self.0 == Self::EMPTY.0
  }
}

impl Display for TokenSpan<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl AsRef<str> for TokenSpan<'_> {
  fn as_ref(&self) -> &str {
    self.0
  }
}

impl<'a> From<&'a str> for TokenSpan<'a> {
  fn from(s: &'a str) -> Self {
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
  FloatLiteral(FloatBits),
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
  InvalidFloatLiteral,
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
      InvalidFloatLiteral => write!(f, "invalid float literal"),
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
  Type,
  Struct,
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
      Type => write!(f, "type"),
      Struct => write!(f, "struct"),
    }
  }
}

impl std::fmt::Display for Token<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.tag {
      Eof => write!(f, "<end of file>"),
      Newline => write!(f, "<new line>"),
      Identifer => write!(f, "<identifier {}>", self.span.0),
      Kw(kw) => write!(f, "<keyword {kw}>"),
      Op(op) => write!(f, "<op {op}>"),
      RawOp(op) => write!(f, "<raw op: `{op}`>"),
      StrLiteral(s) => write!(f, "<string literal {s:?}>"),
      IntLiteral(i) => write!(f, "<int literal {i}>"),
      FloatLiteral(fl) => write!(f, "<float literal {fl}>"),
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

/// Scans the source as bytes.  Every token class is ASCII; bytes at or above
/// 0x80 only occur inside string literals and comments, where they are copied
/// or skipped whole, so multi-byte sequences are never cut.  Locations count
/// bytes.
pub struct Tokenizer<'a> {
  arena: &'a Bump,
  buffer: &'a str,
  index: usize,
  colstart: usize,
  line: u32,
  diag: Rc<Diagnostic>,
}

use Paired::*;
use TokenErr::*;
use TokenTag::*;

static KEYWORDS: phf::Map<&'static str, Keyword> = phf::phf_map! {
  "fn" => Keyword::Fn,
  "let" => Keyword::Let,
  "rec" => Keyword::Rec,
  "with" => Keyword::With,
  "and" => Keyword::And,
  "is" => Keyword::Is,
  "if" => Keyword::If,
  "else" => Keyword::Else,
  "then" => Keyword::Then,
  "end" => Keyword::End,
  "type" => Keyword::Type,
  "struct" => Keyword::Struct,
};

impl<'a, 'b> Tokenizer<'a>
where
  'a: 'b,
{
  pub fn new<I>(arena: &'a Bump, input: I, diag: Rc<Diagnostic>) -> Self
  where
    I: AsRef<str>,
  {
    // A NUL sentinel ends the buffer so the scanner never bounds-checks.
    let input = input.as_ref();
    let bytes: &'a [u8] = {
      let bytes = arena.alloc_slice_fill_copy(input.len() + 1, 0u8);
      bytes[..input.len()].copy_from_slice(input.as_bytes());
      bytes
    };
    let buffer = std::str::from_utf8(bytes).expect("a str followed by NUL is valid UTF-8");
    Tokenizer { arena, buffer, index: 0, colstart: 0, line: 1, diag }
  }

  fn move_to_newline_begin(&mut self) {
    self.colstart = self.index;
    self.line += 1;
  }

  fn get_loc(&self) -> Loc {
    Loc::new(self.line, (self.index - self.colstart + 1) as u32, self.index)
  }

  fn slice(&'b self, start: usize) -> &'a str {
    &self.buffer[start..self.index]
  }

  fn make_token(&'b self, tag: TokenTag<'a>, loc: Loc) -> Token<'a> {
    Token::new(tag, TokenSpan::new(self.slice(loc.index)), loc)
  }

  fn make_error(&'b self, err: TokenErr, loc: Loc) -> Token<'a> {
    self.make_token(Error(err), loc)
  }

  fn eof(&'b self) -> Token<'a> {
    Token::new(
      Eof,
      TokenSpan::empty(),
      Loc::new(self.line, (self.index - self.colstart) as u32, self.index),
    )
  }

  fn ch(&self) -> u8 {
    self.buffer.as_bytes()[self.index]
  }
  fn ch_at(&self, i: usize) -> u8 {
    self.buffer.as_bytes()[i]
  }

  fn string_literal(&'b mut self, loc: Loc) -> Token<'a> {
    self.index += 1;
    let start = self.index;
    // Only literals with escapes are copied; the rest are sliced from the buffer.
    let mut unescaped: Option<Vec<u8>> = None;
    loop {
      match self.ch() {
        0 => return self.make_error(EarlyEof, loc),
        b'\n' => {
          self.index += 1;
          self.move_to_newline_begin();
          return self.make_error(NewlineInLiteral, loc);
        }
        b'"' => break,
        b'\\' => {
          let buf =
            unescaped.get_or_insert_with(|| self.buffer.as_bytes()[start..self.index].to_vec());
          self.index += 1;
          match self.ch() {
            0 => return self.make_error(EarlyEof, loc),
            b'n' => buf.push(b'\n'),
            b'r' => buf.push(b'\r'),
            b't' => buf.push(b'\t'),
            c @ (b'\\' | b'"') => buf.push(c),
            _ => {
              self.index += 1;
              return self.make_error(InvalidEscapeSequence, loc);
            }
          }
        }
        c => {
          if let Some(buf) = &mut unescaped {
            buf.push(c);
          }
        }
      }
      self.index += 1;
    }
    let text = match unescaped {
      // Escapes only ever add ASCII, so the copy is still valid UTF-8.
      Some(buf) => self.arena.alloc_str(std::str::from_utf8(&buf).expect("valid UTF-8")),
      None => self.slice(start),
    };
    self.index += 1;
    self.make_token(StrLiteral(text), loc)
  }

  fn numeric_literal(&'b mut self, loc: Loc, first_c: u8, neg: bool) -> Token<'a> {
    use TokenErr::*;
    let start = self.index; // position of first_c in buffer
    self.index += 1;
    let mut base: u32 = 10;
    let mut value: i128 = (first_c - b'0').into();
    if neg {
      value = -value;
    }
    if first_c == b'0' {
      match self.ch() {
        b'b' | b'B' => {
          self.index += 1;
          base = 2;
        }
        b'o' | b'O' => {
          self.index += 1;
          base = 8;
        }
        b'x' | b'X' => {
          self.index += 1;
          base = 16;
        }
        b'0'..=b'9' | b'_' => {}
        b'.' | b'e' | b'E' => {} // handled below as float
        _ => return self.make_token(IntLiteral(value), loc),
      }
    }
    if base != 10 {
      match self.ch() {
        b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => {}
        _ => return self.make_error(InvalidIntLiteralPrefix, loc),
      }
    }
    // Parse integer digits (with underscore separators)
    loop {
      match self.ch() {
        b'_' => {
          self.index += 1;
          continue;
        }
        b'0'..=b'9' => {}
        b'a'..=b'f' | b'A'..=b'F' if base == 16 => {}
        _ => break,
      }
      let n = match (self.ch() as char).to_digit(base) {
        Some(n) => n as i128,
        None => {
          self.index += 1;
          return self.make_error(InvalidIntLiteralDigit, loc);
        }
      };
      value *= base as i128;
      value = if neg { value - n } else { value + n };
      self.index += 1;
    }
    // Float: only for decimal base, when followed by '.' + digit or 'e'/'E'
    if base == 10
      && (self.ch() == b'e'
        || self.ch() == b'E'
        || (self.ch() == b'.' && self.ch_at(self.index + 1).is_ascii_digit()))
    {
      return self.float_literal_tail(loc, start, neg);
    }
    self.make_token(IntLiteral(value), loc)
  }

  /// Continue parsing a float literal after the integer part has been consumed.
  /// `start` is the buffer index of the first digit. The caller has already
  /// advanced `self.index` past the integer digits.
  fn float_literal_tail(&'b mut self, loc: Loc, start: usize, neg: bool) -> Token<'a> {
    use TokenErr::*;
    // Fractional part
    if self.ch() == b'.' {
      self.index += 1;
      while self.ch().is_ascii_digit() || self.ch() == b'_' {
        self.index += 1;
      }
    }
    // Exponent part
    if self.ch() == b'e' || self.ch() == b'E' {
      self.index += 1;
      if self.ch() == b'+' || self.ch() == b'-' {
        self.index += 1;
      }
      if !self.ch().is_ascii_digit() {
        return self.make_error(InvalidFloatLiteral, loc);
      }
      while self.ch().is_ascii_digit() || self.ch() == b'_' {
        self.index += 1;
      }
    }
    // Parse from the source slice, stripping underscores into a stack buffer.
    let src = self.slice(start).as_bytes();
    let mut buf = [0u8; 64];
    let mut len = 0;
    if neg {
      buf[len] = b'-';
      len += 1;
    }
    for &c in src {
      if c != b'_' {
        buf[len] = c;
        len += 1;
      }
    }
    match std::str::from_utf8(&buf[..len]).unwrap().parse::<f64>() {
      Ok(f) => self.make_token(FloatLiteral(FloatBits(f)), loc),
      Err(_) => self.make_error(InvalidFloatLiteral, loc),
    }
  }

  fn ident(&'b mut self, loc: Loc) -> Token<'a> {
    self.index += 1;
    let mut symbol_like = false;
    loop {
      match self.ch() {
        b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' => (),
        b'!' | b'$' | b'%' | b'&' | b'*' | b'+' | b'-' | b'/' | b':' | b'<' | b'=' | b'>'
        | b'?' | b'@' | b'^' | b'~' => symbol_like |= true,
        0 | b' ' | b'\t' | b'\r' | b'\n' => break,
        b')' | b']' | b'}' | b',' | b';' => break,
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

    match KEYWORDS.get(self.slice(loc.index)) {
      Some(&kw) => self.make_token(Kw(kw), loc),
      None => self.make_token(Identifer, loc),
    }
  }

  fn skip_comment(&'b mut self) -> bool {
    self.index += 1;
    let mut level = 1;
    loop {
      match self.ch() {
        b'(' => {
          self.index += 1;
          if self.ch() == b'*' {
            self.index += 1;
            level += 1;
          }
        }
        b'*' => {
          self.index += 1;
          if self.ch() == b')' {
            self.index += 1;
            level -= 1;
            if level == 0 {
              return true;
            }
          }
        }
        0 => return false,
        b'\n' => {
          self.index += 1;
          self.move_to_newline_begin();
        }
        _ => self.index += 1,
      }
    }
  }

  fn operator(&'b mut self, loc: Loc, first_c: u8) -> Token<'a> {
    self.index += 1;
    match first_c {
      b',' | b';' | b':' | b'.' => return self.make_token(Op(self.slice(loc.index)), loc),
      b'?' | b'~' | b'!' => {
        match self.ch() {
          b'$' | b'&' | b'*' | b'+' | b'-' | b'/' | b'=' | b'>' | b'@' | b'^' | b'|' | b'%'
          | b'<' => {}
          _ => {
            if first_c != b'!' {
              return self.make_error(InvalidUnaryOperator, self.get_loc());
            }
          }
        }
        while matches!(
          self.ch(),
          b'$' | b'&' | b'*' | b'+' | b'-' | b'/' | b'=' | b'>' | b'@' | b'^' | b'|' | b'%' | b'<'
        ) {
          self.index += 1;
        }
      }
      b'$' | b'&' | b'*' | b'+' | b'-' | b'/' | b'=' | b'>' | b'@' | b'^' | b'|' | b'%' | b'<'
      | b'#' => {
        match self.ch() {
          b'$' | b'&' | b'*' | b'+' | b'-' | b'/' | b'=' | b'>' | b'@' | b'^' | b'|' | b'%'
          | b'<' | b'!' | b'.' | b':' | b'?' | b'~' => {}
          _ => {
            if first_c == b'#' {
              return self.make_error(InvalidBinaryOperator, self.get_loc());
            }
          }
        }
        while matches!(
          self.ch(),
          b'$'
            | b'&'
            | b'*'
            | b'+'
            | b'-'
            | b'/'
            | b'='
            | b'>'
            | b'@'
            | b'^'
            | b'|'
            | b'%'
            | b'<'
            | b'!'
            | b'.'
            | b':'
            | b'?'
            | b'~'
        ) {
          self.index += 1;
        }
      }
      b'`' => loop {
        match self.ch() {
          b'`' => {
            let mut loc = loc;
            loc.index += 1;
            let tok = self.make_token(RawOp(self.slice(loc.index)), loc);
            self.index += 1;
            return tok;
          }
          0 => return self.make_error(EarlyEof, self.get_loc()),
          _ => self.index += 1,
        }
      },
      _ => unreachable!(),
    }
    self.make_token(Op(self.slice(loc.index)), loc)
  }

  pub fn next_with_err(&'b mut self) -> Result<Token<'a>> {
    let tok = self.next_token();
    if let TokenTag::Error(err) = tok.tag { self.diag.fail(err.to_string()) } else { Ok(tok) }
  }

  pub fn next_token(&'b mut self) -> Token<'a> {
    loop {
      let c = self.ch();
      let loc = self.get_loc();
      match c {
        0 => return self.eof(),
        b' ' | b'\t' | b'\r' => {
          self.index += 1;
          while let b' ' | b'\t' | b'\r' = self.ch() {
            self.index += 1;
          }
          continue;
        }
        b'\n' => {
          self.index += 1;
          self.move_to_newline_begin();
          return self.make_token(Newline, loc);
        }
        b'(' => {
          self.index += 1;
          if self.ch() == b'*' {
            self.index += 1;
            if !self.skip_comment() {
              return self.make_error(EarlyEof, loc);
            }
          } else {
            return self.make_token(PairedOpen(Parenthesis), loc);
          }
        }
        b')' => {
          self.index += 1;
          return self.make_token(PairedClose(Parenthesis), loc);
        }
        b'[' => {
          self.index += 1;
          return self.make_token(PairedOpen(Bracket), loc);
        }
        b']' => {
          self.index += 1;
          return self.make_token(PairedClose(Bracket), loc);
        }
        b'{' => {
          self.index += 1;
          return self.make_token(PairedOpen(Brace), loc);
        }
        b'}' => {
          self.index += 1;
          return self.make_token(PairedClose(Brace), loc);
        }
        b'"' => return self.string_literal(loc),
        b'0'..=b'9' => return self.numeric_literal(loc, c, false),
        b'+' | b'-' => {
          let next_c = self.ch_at(self.index + 1);
          if next_c.is_ascii_digit() {
            self.index += 1;
            return self.numeric_literal(loc, next_c, c == b'-');
          }
          return self.operator(loc, c);
        }
        b'a'..=b'z' | b'A'..=b'Z' | b'_' => return self.ident(loc),
        b',' | b';' | b':' | b'.' | b'?' | b'~' | b'!' | b'$' | b'&' | b'*' | b'/' | b'='
        | b'>' | b'@' | b'^' | b'|' | b'%' | b'<' | b'#' | b'`' => {
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
    let arena = Bump::new();
    let diag = Rc::new(Diagnostic::new());
    let mut tokenizer = Tokenizer::new(&arena, input, diag);
    for expected_token in expected {
      let token = tokenizer.next_token();
      assert_eq!(token.tag, expected_token.tag);
      assert_eq!(token.span.to_string().as_str(), expected_token.span);
    }
    let last_token = tokenizer.next_token();
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
    test_tokenize("%", &[t(Op("%"), "%")]);
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
    test_tokenize("\"héllo\"", &[t(StrLiteral("héllo"), "\"héllo\"")]);
    test_tokenize("\"é\\n\"", &[t(StrLiteral("é\n"), "\"é\\n\"")]);
  }

  #[test]
  fn test_comments() {
    test_tokenize("(* *)\n", &[t(Newline, "\n")]);
    test_tokenize("(* (* *)\n", &[t(Error(EarlyEof), "(* (* *)\n")]);
    test_tokenize("(* (* *) *)\n", &[t(Newline, "\n")]);
    test_tokenize("(* 注释 *)\n", &[t(Newline, "\n")]);
  }
}
