use std::fmt::Debug;

use bumpalo::Bump;

use crate::codegen::CodeGenCtx;
use crate::sexp::{Sexp, SexpPool, ToSexp};
use crate::tokenizer::{Keyword, Paired, Token, TokenStr, TokenTag, Tokenizer};

use anyhow::Result;

pub struct TokenIndex(u32);

impl From<u32> for TokenIndex {
  fn from(i: u32) -> Self {
    TokenIndex(i)
  }
}

impl From<TokenIndex> for u32 {
  fn from(val: TokenIndex) -> Self {
    val.0
  }
}

pub struct TokenRef<'a>(&'a Token<'a>);

impl<'a> From<&'a Token<'a>> for TokenRef<'a> {
  fn from(t: &'a Token<'a>) -> Self {
    TokenRef(t)
  }
}

impl<'a> From<TokenRef<'a>> for &'a Token<'a> {
  fn from(val: TokenRef<'a>) -> Self {
    val.0
  }
}

struct Affinity;

impl Affinity {
  const NONE: u32 = u32::MAX;
  const POSTFIX_START: u32 = 3000;
  const PREFIX_START: u32 = 2000;
  const INFIX_START: u32 = 1000;
  const PREFIX: phf::Map<&'static str, (u32, u32)> = phf::phf_map! {
    "+" => (Self::NONE, Self::PREFIX_START + 1),
    "-" => (Self::NONE, Self::PREFIX_START + 1),
  };
  const POSTFIX: phf::Map<&'static str, (u32, u32)> = phf::phf_map! {
    "(" => (Self::POSTFIX_START + 1, Self::NONE),
    "[" => (Self::POSTFIX_START + 1, Self::NONE),
    "{" => (Self::POSTFIX_START + 1, Self::NONE),
  };
  const INFIX: phf::Map<&'static str, (u32, u32)> = phf::phf_map! {
    ":" => (Self::INFIX_START + 20, Self::INFIX_START + 19),
    "<" => (Self::INFIX_START + 1, Self::INFIX_START + 2),
    ">" => (Self::INFIX_START + 1, Self::INFIX_START + 2),
    "==" => (Self::INFIX_START + 1, Self::INFIX_START + 2),
    "!=" => (Self::INFIX_START + 1, Self::INFIX_START + 2),
    "<=" => (Self::INFIX_START + 1, Self::INFIX_START + 2),
    ">=" => (Self::INFIX_START + 1, Self::INFIX_START + 2),
    "@" => (Self::INFIX_START + 8, Self::INFIX_START + 7),
    "+" => (Self::INFIX_START + 3, Self::INFIX_START + 4),
    "-" => (Self::INFIX_START + 3, Self::INFIX_START + 4),
    "*" => (Self::INFIX_START + 5, Self::INFIX_START + 6),
    "/" => (Self::INFIX_START + 5, Self::INFIX_START + 6),
  };
  fn get_prefix(op: &str) -> Option<(u32, u32)> {
    Self::PREFIX.get(op).copied()
  }
  fn get_postfix(op: &str) -> Option<(u32, u32)> {
    Self::POSTFIX.get(op).copied()
  }
  fn get_infix(op: &str) -> Option<(u32, u32)> {
    Self::INFIX.get(op).copied()
  }
}

pub const BUILTIN_CTORS: phf::Map<&'static str, usize> = phf::phf_map!["true" => 0, "false" => 0];

pub type ExprRef<'a> = &'a Expr<'a>;
pub type ExprsRef<'a> = &'a [ExprRef<'a>];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr<'a> {
  IntLiteral(i128),
  StrLiteral(&'a str),
  Ident(TokenStr<'a>),
  Op(TokenStr<'a>),
  OpApply { op: ExprRef<'a>, pair: Option<Paired>, args: ExprsRef<'a> },
  Apply { func: ExprRef<'a>, pair: Option<Paired>, args: ExprsRef<'a> },
  Bind { rec: bool, name: TokenStr<'a>, expr: ExprRef<'a> },
  Fn { params: ExprsRef<'a>, body: ExprRef<'a> },
  Block(ExprsRef<'a>),
  If(ExprRef<'a>, ExprRef<'a>, ExprRef<'a>),
  Tuple(ExprsRef<'a>),
}

type ExprCon<'a> = Expr<'a>;

impl ToSexp for Expr<'_> {
  fn to_sexp<'pool>(&self, pool: &'pool SexpPool) -> Sexp<'pool> {
    use Expr::*;
    match self {
      IntLiteral(n) => pool.atom(n.to_string()),
      StrLiteral(s) => pool.atom(s),
      Ident(s) => pool.atom(s.as_ref()),
      Op(s) => pool.atom(s.as_ref()),
      OpApply { op, pair: _, args } => pool
        .non_empty_list(op.to_sexp(pool), args.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>()),
      Apply { func, pair: _, args } => pool.non_empty_list(
        func.to_sexp(pool),
        args.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>(),
      ),
      Bind { rec, name, expr } => pool.list(&[
        pool.atom(if *rec { "let-rec" } else { "let" }),
        pool.atom(name.as_ref()),
        expr.to_sexp(pool),
      ]),
      Fn { params, body } => pool.list(&[
        pool.atom("fn"),
        pool.list(params.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>()),
        body.to_sexp(pool),
      ]),
      Block(xs) => pool
        .non_empty_list(pool.atom("block"), xs.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>()),
      If(a, b, c) => {
        pool.list(&[pool.atom("if"), a.to_sexp(pool), b.to_sexp(pool), c.to_sexp(pool)])
      }
      Tuple(xs) => pool
        .non_empty_list(pool.atom("tuple"), xs.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>()),
    }
  }
}

impl std::fmt::Display for Expr<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let pool = SexpPool::new();
    let sexp = self.to_sexp(&pool);
    write!(f, "{sexp}")
  }
}

pub struct SynTree<'a> {
  pub root: ExprRef<'a>,
}

impl std::fmt::Display for SynTree<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let pool = SexpPool::new();
    let sexp = self.root.to_sexp(&pool);
    write!(f, "{sexp}")
  }
}

pub struct Parser<'a> {
  arena: &'a Bump,
  tokenizer: Tokenizer<'a>,
  token: Option<&'a Token<'a>>,
  ctx: &'a mut CodeGenCtx<'a>,
}

pub struct PeekResult<'a, T> {
  inner: &'a T,
}

type PeekToken<'a> = Result<PeekResult<'a, Token<'a>>>;
type PeekExpr<'a> = Result<PeekResult<'a, Expr<'a>>>;

impl<'a> Parser<'a> {
  pub fn new(arena: &'a Bump, src: &'a str, ctx: &'a mut CodeGenCtx<'a>) -> Self {
    let tokenizer: Tokenizer<'a> = Tokenizer::new(arena, src);
    Self { arena, tokenizer, token: None, ctx }
  }

  fn skip_token(&mut self) {
    self.token = None;
  }

  fn peek_token(&mut self) -> PeekToken<'a> {
    match self.token {
      Some(tok) => Ok(PeekResult { inner: tok }),
      None => {
        let tok = self.tokenizer.next_with_err()?;
        let tokref = self.arena.alloc(tok.clone());
        self.token = Some(self.arena.alloc(tok));
        Ok(PeekResult { inner: tokref })
      }
    }
  }

  fn next_token(&mut self) -> PeekToken<'a> {
    let tok = self.peek_token()?;
    self.token = None;
    Ok(tok)
  }

  fn expect_reach_eof(&mut self) -> Result<()> {
    match self.peek_token()? {
      x if x.inner.tag == TokenTag::Eof => Ok(()),
      x => Err(anyhow::anyhow!("expect eof but got {}", x.inner)),
    }
  }

  fn expect_paired_open(&mut self, po: Paired) -> PeekToken<'a> {
    while self.peek_newline() {
      self.skip_token();
    }
    let tok = self.next_token()?;
    if tok.inner.tag != TokenTag::PairedOpen(po) {
      return Err(anyhow::anyhow!("expected paired open {}", po));
    }
    Ok(tok)
  }

  fn expect_paired_close(&mut self, po: Paired, allow_newline: bool) -> PeekToken<'a> {
    if allow_newline {
      while self.peek_newline() {
        self.skip_token();
      }
    }
    let tok = self.next_token()?;
    if tok.inner.tag != TokenTag::PairedClose(po) {
      return Err(anyhow::anyhow!("expected paired close {}", po));
    }
    Ok(tok)
  }

  fn peek_keyword(&mut self, kw: Keyword, allow_newline: bool) -> bool {
    if allow_newline {
      while self.peek_newline() {
        self.skip_token();
      }
    }
    match self.peek_token() {
      Ok(x) => x.inner.tag == TokenTag::Kw(kw),
      Err(_) => false,
    }
  }

  fn peek_operator(&mut self, op: &str, allow_newline: bool) -> bool {
    if allow_newline {
      while self.peek_newline() {
        self.skip_token();
      }
    }
    match self.peek_token() {
      Ok(x) => matches!(x.inner.tag, TokenTag::Op(op2) | TokenTag::RawOp(op2) if op == op2),
      Err(_) => false,
    }
  }

  fn peek_paired_close(&mut self, po: Paired, allow_newline: bool) -> bool {
    if allow_newline {
      while self.peek_newline() {
        self.skip_token();
      }
    }
    match self.peek_token() {
      Ok(x) => matches!(x.inner.tag, TokenTag::PairedClose(po2) if po == po2),
      Err(_) => false,
    }
  }

  fn next_ident(&mut self, allow_newline: bool) -> PeekToken<'a> {
    if allow_newline {
      while self.peek_newline() {
        self.skip_token();
      }
    }
    let tok = self.next_token()?;
    if !matches!(tok.inner.tag, TokenTag::Identifer) {
      return Err(anyhow::anyhow!("expected identifier, but got {}", tok.inner));
    }
    Ok(tok)
  }

  fn peek_newline(&mut self) -> bool {
    match self.peek_token() {
      Ok(x) => matches!(x.inner.tag, TokenTag::Newline),
      Err(_) => false,
    }
  }

  fn expect_keyword(&mut self, kw: Keyword, allow_newline: bool) -> PeekToken<'a> {
    if allow_newline {
      while self.peek_newline() {
        self.skip_token();
      }
    }
    let tok = self.next_token()?;
    if tok.inner.tag != TokenTag::Kw(kw) {
      return Err(anyhow::anyhow!("expected keyword {}, but got {}", kw, tok.inner));
    }
    Ok(tok)
  }

  fn expect_operator(&mut self, op: &str, allow_newline: bool) -> PeekToken<'a> {
    if allow_newline {
      while self.peek_newline() {
        self.skip_token();
      }
    }
    let tok = self.next_token()?;
    if !matches!(tok.inner.tag, TokenTag::Op(op2) | TokenTag::RawOp(op2) if op == op2) {
      return Err(anyhow::anyhow!("expected operator {}", op));
    }
    Ok(tok)
  }

  fn parse_expr<'t>(&'t mut self) -> PeekExpr<'a> {
    self.parse_expr_with_affinity(0)
  }

  fn parse_ident_expr<'t>(&'t mut self, def_or_use: bool) -> PeekExpr<'a> {
    let arena = self.arena;
    let tok = self.next_ident(false)?;
    let name = TokenStr::from_span(arena, tok.inner.span);
    if def_or_use {
      self.ctx.emit_ident_def(name);
    } else {
      self.ctx.emit_ident_use(name);
    }
    Ok(PeekResult { inner: arena.alloc(ExprCon::Ident(name)) })
  }

  fn parse_expr_with_affinity<'t>(&'t mut self, minaff: u32) -> PeekExpr<'a> {
    use TokenTag::*;
    let arena = self.arena;
    let lhs_token = self.next_token()?;
    let mut lhs_op = None;
    let mut lhs: &Expr = match lhs_token.inner.tag {
      IntLiteral(n) => {
        self.ctx.emit_int_literal(n);
        arena.alloc(ExprCon::IntLiteral(n))
      }
      StrLiteral(s) => {
        self.ctx.emit_str_literal(s);
        arena.alloc(ExprCon::StrLiteral(s))
      }
      Identifer => {
        self.ctx.emit_ident_use_pre(TokenStr::from_span(arena, lhs_token.inner.span));
        arena.alloc(ExprCon::Ident(TokenStr::from_span(arena, lhs_token.inner.span)))
      }
      PairedOpen(po) => {
        let inner_token = self.peek_token()?;
        match inner_token.inner.tag {
          Op(op) | RawOp(op) => {
            self.skip_token();

            lhs_op = Some(op);

            let _ = self.expect_paired_close(po, false)?;

            self.ctx.emit_op_obj_pre(op);
            arena.alloc(ExprCon::Op(op.into()))
          }
          _ => {
            let expr = self.parse_expr()?;

            if self.peek_operator(",", false) {
              let mut exprs = vec![expr.inner];

              while self.peek_operator(",", false) {
                self.skip_token();

                while self.peek_newline() {
                  self.skip_token();
                }

                let expr = self.parse_expr()?;
                exprs.push(expr.inner);
              }

              let _ = self.expect_paired_close(po, false)?;

              self.ctx.emit_tuple_pre(exprs.len());
              arena.alloc(ExprCon::Tuple(arena.alloc_slice_copy(&exprs)))
            } else {
              let _ = self.expect_paired_close(po, false)?;

              arena.alloc(expr.inner.clone())
            }
          }
        }
      }
      RawOp(op) => {
        lhs_op = Some(op);
        self.ctx.emit_op_obj_pre(op);
        arena.alloc(ExprCon::Op(op.into()))
      }
      Op(op) => {
        let (_laff, raff) =
          Affinity::get_prefix(op).ok_or_else(|| anyhow::anyhow!("prefix operator expected"))?;
        let rhs_expr = self.parse_expr_with_affinity(raff)?;
        self.ctx.emit_prefix_op(op);
        arena.alloc(ExprCon::OpApply {
          op: arena.alloc(ExprCon::Op(op.into())),
          pair: None,
          args: arena.alloc_slice_clone(&[rhs_expr.inner]),
        })
      }
      Kw(kw) => match kw {
        Keyword::Fn => {
          let _ = self.expect_paired_open(Paired::Parenthesis)?;

          let mut params = vec![];

          if !self.peek_paired_close(Paired::Parenthesis, false) {
            let expr = self.parse_ident_expr(true)?;
            params.push(expr.inner);

            if self.peek_operator(",", false) {
              while self.peek_operator(",", false) {
                self.skip_token();

                while self.peek_newline() {
                  self.skip_token();
                }

                let expr = self.parse_ident_expr(true)?;
                params.push(expr.inner);
              }
            }
          }

          let _ = self.expect_paired_close(Paired::Parenthesis, false)?;

          while self.peek_newline() {
            self.skip_token();
          }

          let body = self.parse_exprs()?;

          let _ = self.expect_keyword(Keyword::End, true)?;

          self.ctx.emit_fn();
          arena.alloc(ExprCon::Fn { params: arena.alloc_slice_copy(&params), body: body.inner })
        }
        Keyword::Let => {
          let is_rec = self.peek_keyword(Keyword::Rec, false);
          if is_rec {
            self.skip_token();
          }

          let name = self.next_ident(false)?;
          let name = TokenStr::from_span(arena, name.inner.span);

          let _ = self.expect_operator("=", false)?;

          let body = self.parse_expr()?;

          arena.alloc(ExprCon::Bind { rec: is_rec, name, expr: body.inner })
        }
        Keyword::If => {
          let condition = self.parse_expr()?;

          let _ = self.expect_keyword(Keyword::Then, true)?;

          while self.peek_newline() {
            self.skip_token();
          }

          let then_branch = self.parse_expr()?;

          while self.peek_newline() {
            self.skip_token();
          }

          let _ = self.expect_keyword(Keyword::Else, true)?;

          while self.peek_newline() {
            self.skip_token();
          }

          let else_branch = self.parse_expr()?;

          let _ = self.expect_keyword(Keyword::End, true)?;

          arena.alloc(ExprCon::If(condition.inner, then_branch.inner, else_branch.inner))
        }
        _ => return Err(anyhow::anyhow!("unexpected keyword {}", lhs_token.inner)),
      },
      _ => return Err(anyhow::anyhow!("unexpected token {}", lhs_token.inner)),
    };

    loop {
      let op_token = self.peek_token()?;
      let op_str = match op_token.inner.tag {
        Op(op) | RawOp(op) => op,
        PairedOpen(po) => match po {
          Paired::Parenthesis => "(",
          Paired::Bracket => "[",
          Paired::Brace => "{",
        },
        PairedClose(_) => break,
        Eof | Newline => break,
        Kw(_) => break,
        _ => return Result::Err(anyhow::anyhow!("unexpected trailing token {}", op_token.inner)),
      };

      if let Some((laff, _)) = Affinity::get_postfix(op_str) {
        if laff < minaff {
          break;
        }

        self.skip_token();

        if let PairedOpen(po) = op_token.inner.tag {
          let expr = self.parse_expr()?;

          let mut exprs = vec![expr.inner];

          while self.peek_operator(",", false) {
            self.skip_token();

            while self.peek_newline() {
              self.skip_token();
            }

            let expr = self.parse_expr()?;
            exprs.push(expr.inner);
          }

          let _ = self.expect_paired_close(po, false)?;

          if let Some(op) = lhs_op {
            lhs = arena.alloc(ExprCon::OpApply {
              op: arena.alloc(ExprCon::Op(op.into())),
              pair: Some(po),
              args: arena.alloc_slice_clone(&exprs),
            });
          } else {
            lhs = arena.alloc(ExprCon::Apply {
              func: lhs,
              pair: Some(po),
              args: arena.alloc_slice_clone(&exprs),
            });
          }
        } else {
          let old_lhs: &Expr = lhs;
          lhs = arena.alloc(ExprCon::OpApply {
            op: arena.alloc(ExprCon::Op(op_str.into())),
            pair: None,
            args: arena.alloc_slice_clone(&[old_lhs]),
          });
        }

        lhs_op = None;

        continue;
      } else if let Some((laff, raff)) = Affinity::get_infix(op_str) {
        if laff < minaff {
          break;
        }

        self.skip_token();

        let rhs = self.parse_expr_with_affinity(raff)?;

        let old_lhs: &Expr = lhs;
        lhs = arena.alloc(ExprCon::OpApply {
          op: arena.alloc(ExprCon::Op(op_str.into())),
          pair: None,
          args: arena.alloc_slice_clone(&[old_lhs, rhs.inner]),
        });

        lhs_op = None;

        continue;
      }

      break;
    }

    if let Some(op) = lhs_op {
      lhs = arena.alloc(ExprCon::Op(op.into()));
    }

    Ok(PeekResult { inner: lhs })
  }

  fn parse_exprs<'t>(&'t mut self) -> PeekExpr<'a> {
    let arena = self.arena;
    while self.peek_newline() {
      self.skip_token();
    }

    let first_expr = self.parse_expr()?;
    if !self.peek_operator(";", false) {
      return Ok(first_expr);
    }

    while self.peek_newline() {
      self.skip_token();
    }

    let mut exprs = vec![first_expr.inner];

    while self.peek_operator(";", false) {
      self.skip_token();

      while self.peek_newline() {
        self.skip_token();
      }

      let next_expr = self.parse_expr()?;
      exprs.push(next_expr.inner);
    }

    while self.peek_newline() {
      self.skip_token();
    }

    Ok(PeekResult { inner: arena.alloc(ExprCon::Block(arena.alloc_slice_clone(&exprs))) })
  }

  pub fn parse(&mut self) -> Result<SynTree<'a>> {
    let root = self.parse_exprs()?;
    self.expect_reach_eof()?;
    Ok(SynTree { root: root.inner })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn test_parse_exprs(source: &str, expected_sexp_str: &str) {
    let arena = Bump::new();
    let mut ctx = CodeGenCtx::new(&arena);
    let mut parser = Parser::new(&arena, source, &mut ctx);
    let expr = parser.parse_exprs().unwrap();

    assert_eq!(expr.inner.to_sexp(&SexpPool::new()).to_string(), expected_sexp_str);

    let last = parser.next_token().unwrap();
    assert_eq!(last.inner.tag, TokenTag::Eof);
  }

  #[test]
  fn test_parse_expressions() {
    test_parse_exprs("1 + 2", "(+ 1 2)");
    test_parse_exprs("(1 + 2)", "(+ 1 2)");
    test_parse_exprs("(1 + 2) + 3", "(+ (+ 1 2) 3)");
    test_parse_exprs("1 + (2 + 3)", "(+ 1 (+ 2 3))");
    test_parse_exprs("1 + 2 * (3 + 4)", "(+ 1 (* 2 (+ 3 4)))");
    test_parse_exprs("1 + 2 + 3 + 4", "(+ (+ (+ 1 2) 3) 4)");
    test_parse_exprs("1 + 2 * 3 + 4", "(+ (+ 1 (* 2 3)) 4)");
    test_parse_exprs("1 * 2 * 3", "(* (* 1 2) 3)");
    test_parse_exprs("1 * (2 * 3)", "(* 1 (* 2 3))");
    test_parse_exprs("1 + 2 * 3", "(+ 1 (* 2 3))");
    test_parse_exprs("1 + 2 * (3)", "(+ 1 (* 2 3))");
    test_parse_exprs("1 + 2 * (3 + 4)", "(+ 1 (* 2 (+ 3 4)))");
    test_parse_exprs("1 + 2 * (3 * 4)", "(+ 1 (* 2 (* 3 4)))");
    test_parse_exprs("(1 + 2) * (3 * 4)", "(* (+ 1 2) (* 3 4))");
    test_parse_exprs("+ 1", "(+ 1)");
    test_parse_exprs("+ 1 + 2", "(+ (+ 1) 2)");
    test_parse_exprs("+1 + 2", "(+ 1 2)");
    test_parse_exprs("`rawop`(1)", "(rawop 1)");
    test_parse_exprs("(`rawop`)(1)", "(rawop 1)");
    test_parse_exprs("(+)(1)", "(+ 1)");
    test_parse_exprs("f @ g @ h", "(@ f (@ g h))");
    test_parse_exprs("f{x}[y](z)", "(((f x) y) z)");
    test_parse_exprs("f(x, y)(z)", "((f x y) z)");
    test_parse_exprs("(x, y)", "(tuple x y)");
  }

  #[test]
  fn test_parse_blocks() {
    test_parse_exprs("1; 2; 3", "(block 1 2 3)");
    test_parse_exprs("1 + 2; 3 + 4", "(block (+ 1 2) (+ 3 4))");
    test_parse_exprs(
      r#"
      let x = 1;
      let y = 2;
      x + y"#,
      "(block (let x 1) (let y 2) (+ x y))",
    );
  }

  #[test]
  fn test_let_bindings() {
    test_parse_exprs("let a = 10 + 2", "(let a (+ 10 2))");
    test_parse_exprs("let x = 10", "(let x 10)");
    test_parse_exprs("let rec x = 10 + 2", "(let-rec x (+ 10 2))");
  }

  #[test]
  fn test_functions() {
    test_parse_exprs("fn () unit end", "(fn () unit)");
    test_parse_exprs("fn (x) x end", "(fn (x) x)");
    test_parse_exprs("fn (x, y) x end", "(fn (x y) x)");
    test_parse_exprs("fn (x, y) x + y end", "(fn (x y) (+ x y))");
    test_parse_exprs("fn (x, y) x + y + 1 end", "(fn (x y) (+ (+ x y) 1))");
    test_parse_exprs(
      r#"
      let x = fn (x, y)
        let z = x + y;
        z + 1
      end"#,
      "(let x (fn (x y) (block (let z (+ x y)) (+ z 1))))",
    );
    test_parse_exprs(
      r#"
      let x = fn (x, y)
        if x == 0 then y + 1 else y - 1 end
      end;
      x(1, 2)"#,
      "(block (let x (fn (x y) (if (== x 0) (+ y 1) (- y 1)))) (x 1 2))",
    );
  }
}
