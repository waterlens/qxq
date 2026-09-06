use crate::bytecode::FloatBits;
use crate::diagnostic::{Diagnostic, Result};
use crate::sexp::{Sexp, SexpPool, ToSexp};
use crate::tokenizer::{Keyword, Paired, Token, TokenStr, TokenTag, Tokenizer};
use bumpalo::Bump;
use hashbrown::HashMap;
use indexmap::IndexSet;
use slotmap::SlotMap;
use std::fmt::Debug;
use std::rc::Rc;

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
    "." => (Self::POSTFIX_START + 1, Self::NONE),
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
    "%" => (Self::INFIX_START + 5, Self::INFIX_START + 6),
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

pub type ExprRef<'a, I> = &'a Expr<'a, I>;
pub type ExprsRef<'a, I> = &'a [ExprRef<'a, I>];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr<'a, I> {
  Unit(I),
  EmptyArray(I),
  EmptyMap(I),
  BoolLiteral(bool, I),
  IntLiteral(i64, I),
  FloatLiteral(FloatBits, I),
  StrLiteral(&'a str, I),
  Ident(TokenStr<'a>, I),
  Op(TokenStr<'a>, I),
  OpApply {
    op: ExprRef<'a, I>,
    pair: Option<Paired>,
    args: ExprsRef<'a, I>,
    info: I,
  },
  Apply {
    func: ExprRef<'a, I>,
    pair: Option<Paired>,
    args: ExprsRef<'a, I>,
    info: I,
  },
  Bind {
    rec: bool,
    name: TokenStr<'a>,
    expr: ExprRef<'a, I>,
    info: I,
  },
  Fn {
    name: Option<TokenStr<'a>>,
    params: &'a [TokenStr<'a>],
    body: ExprRef<'a, I>,
    info: I,
  },
  Block(ExprsRef<'a, I>, I),
  If(ExprRef<'a, I>, ExprRef<'a, I>, ExprRef<'a, I>, I),
  Tuple(ExprsRef<'a, I>, I),
  /// `type name = struct {fields} with fn ... end ... end`; every method is a
  /// named `Fn` whose first parameter is `self`.
  StructDecl {
    name: TokenStr<'a>,
    fields: &'a [TokenStr<'a>],
    methods: ExprsRef<'a, I>,
    info: I,
  },
  Construct {
    ty: ExprRef<'a, I>,
    inits: &'a [Init<'a, I>],
    info: I,
  },
  Member {
    receiver: ExprRef<'a, I>,
    member: TokenStr<'a>,
    info: I,
  },
  MemberApply {
    receiver: ExprRef<'a, I>,
    member: TokenStr<'a>,
    args: ExprsRef<'a, I>,
    info: I,
  },
  /// `receiver.n`: the n-th field of a tuple or struct, counted from 1.
  Index {
    receiver: ExprRef<'a, I>,
    index: u32,
    info: I,
  },
}

/// One constructor initializer: `label = expr`, or a positional `expr`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Init<'a, I> {
  pub label: Option<TokenStr<'a>>,
  pub expr: ExprRef<'a, I>,
}

impl<I> ToSexp for Init<'_, I> {
  fn to_sexp<'pool>(&self, pool: &'pool SexpPool) -> Sexp<'pool> {
    let label = self.label.map_or("_", |l| l.0);
    pool.list(&[pool.atom("init"), pool.atom(label), self.expr.to_sexp(pool)])
  }
}

fn method_to_sexp<'pool, I>(method: &Expr<'_, I>, pool: &'pool SexpPool) -> Sexp<'pool> {
  match method {
    Expr::Fn { name: Some(name), params, body, info: _ } => pool.list(&[
      pool.atom("method"),
      pool.atom(name.as_ref()),
      pool.list(params.iter().map(|x| pool.atom(x.as_ref())).collect::<Vec<_>>()),
      body.to_sexp(pool),
    ]),
    _ => method.to_sexp(pool),
  }
}

type ExprCon<'a, I> = Expr<'a, I>;

impl<I> ToSexp for Expr<'_, I> {
  fn to_sexp<'pool>(&self, pool: &'pool SexpPool) -> Sexp<'pool> {
    use Expr::*;
    match self {
      Unit(_) => pool.atom("()"),
      EmptyArray(_) => pool.atom("[]"),
      EmptyMap(_) => pool.atom("{}"),
      BoolLiteral(b, _) => pool.atom(if *b { "true" } else { "false" }),
      IntLiteral(n, _) => pool.atom(n.to_string()),
      FloatLiteral(n, _) => pool.atom(n.to_string()),
      StrLiteral(s, _) => pool.atom(format!("\"{}\"", s.escape_default())),
      Ident(s, _) => pool.atom(s.as_ref()),
      Op(s, _) => pool.atom(s.as_ref()),
      OpApply { op, pair: _, args, info: _ } => pool
        .non_empty_list(op.to_sexp(pool), args.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>()),
      Apply { func, pair: _, args, info: _ } => pool.non_empty_list(
        func.to_sexp(pool),
        args.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>(),
      ),
      Bind { rec, name, expr, info: _ } => pool.list(&[
        pool.atom(if *rec { "let-rec" } else { "let" }),
        pool.atom(name.as_ref()),
        expr.to_sexp(pool),
      ]),
      Fn { name, params, body, info: _ } => match name {
        Some(name) => pool.list(&[
          pool.atom("fn"),
          pool.atom(name),
          pool.list(params.iter().map(|x| pool.atom(x.as_ref())).collect::<Vec<_>>()),
          body.to_sexp(pool),
        ]),
        None => pool.list(&[
          pool.atom("fn"),
          pool.list(params.iter().map(|x| pool.atom(x.as_ref())).collect::<Vec<_>>()),
          body.to_sexp(pool),
        ]),
      },
      Block(xs, _) => pool
        .non_empty_list(pool.atom("block"), xs.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>()),
      If(a, b, c, _) => {
        pool.list(&[pool.atom("if"), a.to_sexp(pool), b.to_sexp(pool), c.to_sexp(pool)])
      }
      Tuple(xs, _) => pool
        .non_empty_list(pool.atom("tuple"), xs.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>()),
      StructDecl { name, fields, methods, info: _ } => {
        let fields = fields.iter().map(|x| pool.atom(x.as_ref())).collect::<Vec<_>>();
        let mut parts = vec![pool.atom("struct"), pool.non_empty_list(pool.atom("fields"), fields)];
        parts.extend(methods.iter().map(|m| method_to_sexp(m, pool)));
        pool.list(&[pool.atom("type"), pool.atom(name.as_ref()), pool.list(parts)])
      }
      Construct { ty, inits, info: _ } => {
        let mut parts = vec![pool.atom("construct"), ty.to_sexp(pool)];
        parts.extend(inits.iter().map(|x| x.to_sexp(pool)));
        pool.list(parts)
      }
      Member { receiver, member, info: _ } => {
        pool.list(&[pool.atom("member"), receiver.to_sexp(pool), pool.atom(member.as_ref())])
      }
      MemberApply { receiver, member, args, info: _ } => {
        let mut parts =
          vec![pool.atom("member-apply"), receiver.to_sexp(pool), pool.atom(member.as_ref())];
        parts.extend(args.iter().map(|x| x.to_sexp(pool)));
        pool.list(parts)
      }
      Index { receiver, index, info: _ } => {
        pool.list(&[pool.atom("index"), receiver.to_sexp(pool), pool.atom(index.to_string())])
      }
    }
  }
}

impl<I> std::fmt::Display for Expr<'_, I> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let pool = SexpPool::new();
    let sexp = self.to_sexp(&pool);
    write!(f, "{sexp}")
  }
}

impl<I> Expr<'_, I> {
  pub fn get_info(&self) -> &I {
    use Expr::*;
    match self {
      Unit(i)
      | EmptyArray(i)
      | EmptyMap(i)
      | BoolLiteral(_, i)
      | IntLiteral(_, i)
      | FloatLiteral(_, i)
      | StrLiteral(_, i)
      | Ident(_, i)
      | Op(_, i)
      | OpApply { info: i, .. }
      | Apply { info: i, .. }
      | Bind { info: i, .. }
      | Fn { info: i, .. }
      | Block(_, i)
      | If(.., i)
      | Tuple(_, i)
      | StructDecl { info: i, .. }
      | Construct { info: i, .. }
      | Member { info: i, .. }
      | MemberApply { info: i, .. }
      | Index { info: i, .. } => i,
    }
  }
}

pub struct InfoExpr<'a> {
  pub expr: ExprRef<'a, InfoKey>,
  pub map: &'a SlotMap<InfoKey, Info<'a>>,
}

impl<'a> ToSexp for InfoExpr<'a> {
  fn to_sexp<'pool>(&self, pool: &'pool SexpPool) -> Sexp<'pool> {
    use Expr::*;
    let mut parts = Vec::new();
    let is_atom = match &self.expr {
      Unit(_) => {
        parts.push(pool.atom("()"));
        true
      }
      EmptyArray(_) => {
        parts.push(pool.atom("[]"));
        true
      }
      EmptyMap(_) => {
        parts.push(pool.atom("{}"));
        true
      }
      BoolLiteral(b, _) => {
        parts.push(pool.atom(if *b { "true" } else { "false" }));
        true
      }
      IntLiteral(n, _) => {
        parts.push(pool.atom(n.to_string()));
        true
      }
      FloatLiteral(n, _) => {
        parts.push(pool.atom(n.to_string()));
        true
      }
      StrLiteral(s, _) => {
        parts.push(pool.atom(format!("\"{}\"", s.escape_default())));
        true
      }
      Ident(s, _) => {
        parts.push(pool.atom(s.as_ref()));
        true
      }
      Op(s, _) => {
        parts.push(pool.atom(s.as_ref()));
        true
      }
      OpApply { op, pair: _, args, info: _ } => {
        parts.push(InfoExpr { expr: op, map: self.map }.to_sexp(pool));
        parts.extend(args.iter().map(|x| InfoExpr { expr: x, map: self.map }.to_sexp(pool)));
        false
      }
      Apply { func, pair: _, args, info: _ } => {
        parts.push(InfoExpr { expr: func, map: self.map }.to_sexp(pool));
        parts.extend(args.iter().map(|x| InfoExpr { expr: x, map: self.map }.to_sexp(pool)));
        false
      }
      Bind { rec, name, expr, info: _ } => {
        parts.push(pool.atom(if *rec { "let-rec" } else { "let" }));
        parts.push(pool.atom(name.as_ref()));
        parts.push(InfoExpr { expr, map: self.map }.to_sexp(pool));
        false
      }
      Fn { name, params, body, info: _ } => {
        parts.push(pool.atom("fn"));
        if let Some(name) = name {
          parts.push(pool.atom(name.as_ref()));
        }
        parts.push(pool.list(params.iter().map(|x| pool.atom(x.as_ref())).collect::<Vec<_>>()));
        parts.push(InfoExpr { expr: body, map: self.map }.to_sexp(pool));
        false
      }
      Block(xs, _) => {
        parts.push(pool.atom("block"));
        parts.extend(xs.iter().map(|x| InfoExpr { expr: x, map: self.map }.to_sexp(pool)));
        false
      }
      If(a, b, c, _) => {
        parts.push(pool.atom("if"));
        parts.push(InfoExpr { expr: a, map: self.map }.to_sexp(pool));
        parts.push(InfoExpr { expr: b, map: self.map }.to_sexp(pool));
        parts.push(InfoExpr { expr: c, map: self.map }.to_sexp(pool));
        false
      }
      Tuple(xs, _) => {
        parts.push(pool.atom("tuple"));
        parts.extend(xs.iter().map(|x| InfoExpr { expr: x, map: self.map }.to_sexp(pool)));
        false
      }
      StructDecl { name, fields, methods, info: _ } => {
        parts.push(pool.atom("type"));
        parts.push(pool.atom(name.as_ref()));
        let fields = fields.iter().map(|x| pool.atom(x.as_ref())).collect::<Vec<_>>();
        let mut body = vec![pool.atom("struct"), pool.non_empty_list(pool.atom("fields"), fields)];
        body.extend(methods.iter().map(|m| InfoExpr { expr: m, map: self.map }.to_sexp(pool)));
        parts.push(pool.list(body));
        false
      }
      Construct { ty, inits, info: _ } => {
        parts.push(pool.atom("construct"));
        parts.push(InfoExpr { expr: ty, map: self.map }.to_sexp(pool));
        parts.extend(inits.iter().map(|x| {
          let label = x.label.map_or("_", |l| l.0);
          let expr = InfoExpr { expr: x.expr, map: self.map }.to_sexp(pool);
          pool.list(&[pool.atom("init"), pool.atom(label), expr])
        }));
        false
      }
      Member { receiver, member, info: _ } => {
        parts.push(pool.atom("member"));
        parts.push(InfoExpr { expr: receiver, map: self.map }.to_sexp(pool));
        parts.push(pool.atom(member.as_ref()));
        false
      }
      MemberApply { receiver, member, args, info: _ } => {
        parts.push(pool.atom("member-apply"));
        parts.push(InfoExpr { expr: receiver, map: self.map }.to_sexp(pool));
        parts.push(pool.atom(member.as_ref()));
        parts.extend(args.iter().map(|x| InfoExpr { expr: x, map: self.map }.to_sexp(pool)));
        false
      }
      Index { receiver, index, info: _ } => {
        parts.push(pool.atom("index"));
        parts.push(InfoExpr { expr: receiver, map: self.map }.to_sexp(pool));
        parts.push(pool.atom(index.to_string()));
        false
      }
    };

    let info_key = self.expr.get_info();
    if let Some(info) = self.map.get(*info_key)
      && !info.freevars.is_empty()
    {
      parts.push(info.to_sexp(pool));
    }

    if is_atom && parts.len() == 1 { parts.pop().unwrap() } else { pool.list(&parts) }
  }
}

#[derive(Debug, Clone, Default)]
pub struct Info<'a> {
  pub freevars: Vec<TokenStr<'a>>,
}

impl ToSexp for Info<'_> {
  fn to_sexp<'pool>(&self, pool: &'pool SexpPool) -> Sexp<'pool> {
    pool.non_empty_list(
      pool.atom("freevars"),
      self.freevars.iter().map(|s| pool.atom(s.as_ref())).collect::<Vec<_>>(),
    )
  }
}

pub struct SynTree<'a, I> {
  pub root: ExprRef<'a, I>,
  pub information: SlotMap<InfoKey, Info<'a>>,
}

impl<I> std::fmt::Display for SynTree<'_, I> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let pool = SexpPool::new();
    let sexp = self.root.to_sexp(&pool);
    write!(f, "{sexp}")
  }
}

pub type InfoKey = slotmap::DefaultKey;

pub struct Parser<'a> {
  arena: &'a Bump,
  tokenizer: Tokenizer<'a>,
  token: Option<&'a Token<'a>>,
  information: SlotMap<InfoKey, Info<'a>>,
  func_stack: Vec<FunctionCtx<'a>>,
  diag: Rc<Diagnostic>,
}

pub struct PeekResult<'a, T> {
  inner: &'a T,
}

type PeekToken<'a> = Result<PeekResult<'a, Token<'a>>>;
type PeekExpr<'a> = Result<PeekResult<'a, Expr<'a, InfoKey>>>;

#[derive(Default)]
struct FunctionCtx<'a> {
  scopes: Vec<IndexSet<TokenStr<'a>>>,
  local_counts: HashMap<TokenStr<'a>, u32>,
  freevars: IndexSet<TokenStr<'a>>,
  self_name: Option<TokenStr<'a>>,
}

impl<'a> Parser<'a> {
  pub fn new(arena: &'a Bump, diag: Rc<Diagnostic>, src: &'a str) -> Self {
    let tokenizer: Tokenizer<'a> = Tokenizer::new(arena, src, Rc::clone(&diag));
    let information: SlotMap<InfoKey, Info<'a>> = SlotMap::new();
    let mut parser =
      Self { arena, tokenizer, token: None, information, func_stack: Vec::with_capacity(4), diag };
    parser.enter_function(None);
    parser
  }

  fn enter_function(&mut self, name: Option<TokenStr<'a>>) {
    self.func_stack.push(FunctionCtx {
      scopes: vec![IndexSet::new()],
      local_counts: HashMap::new(),
      freevars: IndexSet::new(),
      self_name: name,
    });
  }

  fn pop_function(&mut self) -> Vec<TokenStr<'a>> {
    let ctx = self.func_stack.pop().expect("function stack underflow");
    let mut free_vec: Vec<_> = ctx.freevars.iter().cloned().collect();
    free_vec.sort();
    free_vec
  }

  // Free variables of a nested function are usages in its parent.
  fn propagate_freevars(&mut self, freevars: &[TokenStr<'a>]) {
    if let Some(parent) = self.func_stack.last_mut() {
      for fv in freevars {
        Self::use_var_in_ctx(parent, *fv);
      }
    }
  }

  fn enter_scope(&mut self) {
    self.func_stack.last_mut().expect("no active function").scopes.push(IndexSet::new());
  }

  fn leave_scope(&mut self) {
    let ctx = self.func_stack.last_mut().expect("no active function");
    if let Some(scope) = ctx.scopes.pop() {
      for name in scope {
        if let Some(count) = ctx.local_counts.get_mut(&name) {
          *count -= 1;
          if *count == 0 {
            ctx.local_counts.remove(&name);
          }
        }
      }
    }
  }

  fn declare_local(&mut self, name: TokenStr<'a>) {
    if let Some(ctx) = self.func_stack.last_mut()
      && let Some(scope) = ctx.scopes.last_mut()
      && scope.insert(name)
    {
      *ctx.local_counts.entry(name).or_insert(0) += 1;
    }
  }

  fn use_var(&mut self, name: TokenStr<'a>) {
    if let Some(ctx) = self.func_stack.last_mut() {
      Self::use_var_in_ctx(ctx, name);
    }
  }

  fn use_var_in_ctx(ctx: &mut FunctionCtx<'a>, name: TokenStr<'a>) {
    if matches!(ctx.self_name, Some(name2) if name2 == name) || ctx.local_counts.contains_key(&name)
    {
      return; // Bound locally
    }
    // Not bound locally, so it's free
    ctx.freevars.insert(name);
  }

  fn new_info(&mut self, freevars: Vec<TokenStr<'a>>) -> InfoKey {
    self.information.insert(Info { freevars })
  }

  fn new_empty_info(&mut self) -> InfoKey {
    self.information.insert(Info::default())
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
      x => self.diag.fail(format!("expect eof but got {}", x.inner)),
    }
  }

  fn expect_paired_open(&mut self, po: Paired) -> PeekToken<'a> {
    self.skip_newlines();
    let tok = self.next_token()?;
    if tok.inner.tag != TokenTag::PairedOpen(po) {
      return self.diag.fail(format!("expected paired open {}", po));
    }
    Ok(tok)
  }

  fn expect_paired_close(&mut self, po: Paired, allow_newline: bool) -> PeekToken<'a> {
    if allow_newline {
      self.skip_newlines();
    }
    let tok = self.next_token()?;
    if tok.inner.tag != TokenTag::PairedClose(po) {
      return self.diag.fail(format!("expected paired close {}", po));
    }
    Ok(tok)
  }

  fn peek_keyword(&mut self, kw: Keyword, allow_newline: bool) -> bool {
    if allow_newline {
      self.skip_newlines();
    }
    match self.peek_token() {
      Ok(x) => x.inner.tag == TokenTag::Kw(kw),
      Err(_) => false,
    }
  }

  fn peek_operator(&mut self, op: &str, allow_newline: bool) -> bool {
    if allow_newline {
      self.skip_newlines();
    }
    match self.peek_token() {
      Ok(x) => matches!(x.inner.tag, TokenTag::Op(op2) | TokenTag::RawOp(op2) if op == op2),
      Err(_) => false,
    }
  }

  fn peek_paired_open(&mut self, po: Paired) -> bool {
    matches!(self.peek_token(), Ok(x) if x.inner.tag == TokenTag::PairedOpen(po))
  }

  fn peek_paired_close(&mut self, po: Paired, allow_newline: bool) -> bool {
    if allow_newline {
      self.skip_newlines();
    }
    match self.peek_token() {
      Ok(x) => matches!(x.inner.tag, TokenTag::PairedClose(po2) if po == po2),
      Err(_) => false,
    }
  }

  fn next_ident(&mut self, allow_newline: bool) -> PeekToken<'a> {
    if allow_newline {
      self.skip_newlines();
    }
    let tok = self.next_token()?;
    if !matches!(tok.inner.tag, TokenTag::Identifer) {
      return self.diag.fail(format!("expected identifier, but got {}", tok.inner));
    }
    Ok(tok)
  }

  fn peek_newline(&mut self) -> bool {
    match self.peek_token() {
      Ok(x) => matches!(x.inner.tag, TokenTag::Newline),
      Err(_) => false,
    }
  }

  fn skip_newlines(&mut self) {
    while self.peek_newline() {
      self.skip_token()
    }
  }

  fn expect_keyword(&mut self, kw: Keyword, allow_newline: bool) -> PeekToken<'a> {
    if allow_newline {
      self.skip_newlines();
    }
    let tok = self.next_token()?;
    if tok.inner.tag != TokenTag::Kw(kw) {
      return self.diag.fail(format!("expected keyword {}, but got {}", kw, tok.inner));
    }
    Ok(tok)
  }

  fn expect_operator(&mut self, op: &str, allow_newline: bool) -> PeekToken<'a> {
    if allow_newline {
      self.skip_newlines();
    }
    let tok = self.next_token()?;
    if !matches!(tok.inner.tag, TokenTag::Op(op2) | TokenTag::RawOp(op2) if op == op2) {
      return self.diag.fail(format!("expected operator {}", op));
    }
    Ok(tok)
  }

  fn parse_expr<'t>(&'t mut self) -> PeekExpr<'a> {
    self.parse_expr_with_affinity(0)
  }

  fn parse_ident<'t>(&'t mut self, is_decl: bool) -> Result<TokenStr<'a>> {
    let tok = self.next_ident(false)?;
    let name = TokenStr::from_span(tok.inner.span);
    if is_decl {
      self.declare_local(name);
    } else {
      self.use_var(name);
    }
    Ok(name)
  }

  fn parse_function<'t>(&'t mut self, name: Option<TokenStr<'a>>) -> PeekExpr<'a> {
    let (params, body, freevars) = self.parse_function_parts(name)?;
    self.propagate_freevars(&freevars);
    let info = self.new_info(freevars);
    Ok(PeekResult { inner: self.arena.alloc(ExprCon::Fn { name, params, body, info }) })
  }

  /// Parses `(params) body end` in a fresh function context and returns the
  /// sorted free variables without propagating them to the parent.
  fn parse_function_parts<'t>(
    &'t mut self,
    name: Option<TokenStr<'a>>,
  ) -> Result<(&'a [TokenStr<'a>], ExprRef<'a, InfoKey>, Vec<TokenStr<'a>>)> {
    self.enter_function(name);
    let _ = self.expect_paired_open(Paired::Parenthesis)?;

    let mut params = vec![];
    if !self.peek_paired_close(Paired::Parenthesis, false) {
      loop {
        params.push(self.parse_ident(true)?);
        if !self.peek_operator(",", false) {
          break;
        }
        self.skip_token();
        self.skip_newlines();
      }
    }
    let _ = self.expect_paired_close(Paired::Parenthesis, false)?;
    self.skip_newlines();

    let body = self.parse_exprs()?;
    let _ = self.expect_keyword(Keyword::End, true)?;
    let freevars = self.pop_function();
    Ok((self.arena.alloc_slice_copy(&params), body.inner, freevars))
  }

  fn parse_struct_decl<'t>(&'t mut self) -> PeekExpr<'a> {
    let arena = self.arena;
    let name_tok = self.next_ident(false)?;
    let name = TokenStr::from_span(name_tok.inner.span);
    let _ = self.expect_operator("=", false)?;
    let _ = self.expect_keyword(Keyword::Struct, false)?;
    let _ = self.expect_paired_open(Paired::Brace)?;

    let mut fields = vec![];
    if !self.peek_paired_close(Paired::Brace, true) {
      loop {
        let tok = self.next_ident(true)?;
        fields.push(TokenStr::from_span(tok.inner.span));
        if !self.peek_operator(",", true) {
          break;
        }
        self.skip_token();
      }
    }
    let _ = self.expect_paired_close(Paired::Brace, true)?;

    // A method body may name a binder of its recursion group declared later,
    // so free variables are settled only once every method has been parsed.
    let mut methods = vec![];
    while self.peek_keyword(Keyword::With, true) {
      self.skip_token();
      let _ = self.expect_keyword(Keyword::Fn, false)?;
      let tok = self.next_ident(false)?;
      let mname = TokenStr::from_span(tok.inner.span);
      let (params, body, freevars) = self.parse_function_parts(None)?;
      if params.first().map(|p| p.0) != Some("self") {
        return self.diag.fail(format!("the first parameter of method {mname} must be `self`"));
      }
      methods.push((mname, params, body, freevars));
    }
    let _ = self.expect_keyword(Keyword::End, true)?;

    let rec_group: Vec<_> = methods.iter().map(|m| m.0).collect();
    let mut method_exprs = Vec::with_capacity(methods.len());
    for (mname, params, body, mut freevars) in methods {
      freevars.retain(|v| !rec_group.contains(v));
      self.propagate_freevars(&freevars);
      let info = self.new_info(freevars);
      method_exprs.push(&*arena.alloc(ExprCon::Fn { name: Some(mname), params, body, info }));
    }

    self.declare_local(name);
    let decl = arena.alloc(ExprCon::StructDecl {
      name,
      fields: arena.alloc_slice_copy(&fields),
      methods: arena.alloc_slice_clone(&method_exprs),
      info: self.new_empty_info(),
    });
    Ok(PeekResult { inner: decl })
  }

  /// Parses `expr, expr, ...` up to and including the closing token of `po`.
  fn parse_args<'t>(&'t mut self, po: Paired) -> Result<Vec<ExprRef<'a, InfoKey>>> {
    let mut exprs = vec![];
    if !self.peek_paired_close(po, false) {
      loop {
        exprs.push(self.parse_expr()?.inner);
        if !self.peek_operator(",", false) {
          break;
        }
        self.skip_token();
        self.skip_newlines();
      }
    }
    let _ = self.expect_paired_close(po, false)?;
    Ok(exprs)
  }

  fn parse_inits<'t>(&'t mut self) -> Result<Vec<Init<'a, InfoKey>>> {
    let mut inits = vec![];
    if !self.peek_paired_close(Paired::Brace, true) {
      loop {
        inits.push(self.parse_init()?);
        if !self.peek_operator(",", true) {
          break;
        }
        self.skip_token();
      }
    }
    let _ = self.expect_paired_close(Paired::Brace, true)?;
    Ok(inits)
  }

  // `label = expr` names a field rather than using `label` as a variable.
  fn parse_init<'t>(&'t mut self) -> Result<Init<'a, InfoKey>> {
    self.skip_newlines();
    let tok = self.next_token()?;
    if tok.inner.tag == TokenTag::Identifer && self.peek_operator("=", false) {
      self.skip_token();
      let label = Some(TokenStr::from_span(tok.inner.span));
      return Ok(Init { label, expr: self.parse_expr()?.inner });
    }
    Ok(Init { label: None, expr: self.parse_expr_from(tok, 0)?.inner })
  }

  fn parse_expr_with_affinity<'t>(&'t mut self, minaff: u32) -> PeekExpr<'a> {
    let lhs_token = self.next_token()?;
    self.parse_expr_from(lhs_token, minaff)
  }

  fn parse_expr_from<'t>(
    &'t mut self,
    lhs_token: PeekResult<'a, Token<'a>>,
    minaff: u32,
  ) -> PeekExpr<'a> {
    use TokenTag::*;
    let arena = self.arena;
    let mut lhs_op = None;
    let mut lhs: ExprRef<'_, InfoKey> = match lhs_token.inner.tag {
      IntLiteral(n) => arena.alloc(ExprCon::IntLiteral(n.try_into()?, self.new_empty_info())),
      FloatLiteral(f) => arena.alloc(ExprCon::FloatLiteral(f, self.new_empty_info())),
      StrLiteral(s) => arena.alloc(ExprCon::StrLiteral(s, self.new_empty_info())),
      Identifer => {
        let name = TokenStr::from_span(lhs_token.inner.span);
        match name.0 {
          "true" => arena.alloc(ExprCon::BoolLiteral(true, self.new_empty_info())),
          "false" => arena.alloc(ExprCon::BoolLiteral(false, self.new_empty_info())),
          _ => {
            self.use_var(name);
            arena.alloc(ExprCon::Ident(name, self.new_empty_info()))
          }
        }
      }
      PairedOpen(po) => {
        let inner_token = self.peek_token()?;
        match inner_token.inner.tag {
          PairedClose(pc) if pc == po => match po {
            Paired::Parenthesis => {
              self.skip_token();
              arena.alloc(ExprCon::Unit(self.new_empty_info()))
            }
            Paired::Bracket => {
              self.skip_token();
              arena.alloc(ExprCon::EmptyArray(self.new_empty_info()))
            }
            Paired::Brace => {
              self.skip_token();
              arena.alloc(ExprCon::EmptyMap(self.new_empty_info()))
            }
          },
          PairedClose(_) => {
            let _ = self.expect_paired_close(po, false)?;
            unreachable!("expect_paired_close should reject mismatched or unsupported empty pairs")
          }
          _ if po == Paired::Bracket => {
            return self.diag.fail("non-empty arrays have not been supported yet");
          }
          _ if po == Paired::Brace => {
            return self.diag.fail("non-empty maps have not been supported yet");
          }
          Op(op) | RawOp(op) => {
            self.skip_token();

            lhs_op = Some(op);

            let _ = self.expect_paired_close(po, false)?;

            arena.alloc(ExprCon::Op(op.into(), self.new_empty_info()))
          }
          _ => {
            let expr = self.parse_expr()?;

            if self.peek_operator(",", false) {
              let mut exprs = vec![expr.inner];

              while self.peek_operator(",", false) {
                self.skip_token();

                self.skip_newlines();

                let expr = self.parse_expr()?;
                exprs.push(expr.inner);
              }

              let _ = self.expect_paired_close(po, false)?;
              arena.alloc(ExprCon::Tuple(arena.alloc_slice_copy(&exprs), self.new_empty_info()))
            } else {
              let _ = self.expect_paired_close(po, false)?;

              arena.alloc(expr.inner.clone())
            }
          }
        }
      }
      RawOp(op) => {
        lhs_op = Some(op);
        arena.alloc(ExprCon::Op(op.into(), self.new_empty_info()))
      }
      Op(op) => {
        let (_laff, raff) =
          Affinity::get_prefix(op).ok_or_else(|| self.diag.error("prefix operator expected"))?;
        let rhs_expr = self.parse_expr_with_affinity(raff)?;

        arena.alloc(ExprCon::OpApply {
          op: arena.alloc(ExprCon::Op(op.into(), self.new_empty_info())),
          pair: None,
          args: arena.alloc_slice_clone(&[rhs_expr.inner]),
          info: self.new_empty_info(),
        })
      }
      Kw(kw) => match kw {
        Keyword::Fn => self.parse_function(None)?.inner,
        Keyword::Type => self.parse_struct_decl()?.inner,
        Keyword::Let => {
          let is_rec = self.peek_keyword(Keyword::Rec, false);
          if is_rec {
            self.skip_token();
          }
          let name_tok = self.next_ident(false)?;
          let name = TokenStr::from_span(name_tok.inner.span);

          let _ = self.expect_operator("=", false)?;

          if is_rec {
            self.declare_local(name);
          }

          let body = if is_rec {
            if self.peek_keyword(Keyword::Fn, false) {
              self.skip_token();
              self.parse_function(Some(name))?
            } else {
              return self.diag.fail("let rec is not followed by a function definiton");
            }
          } else {
            self.parse_expr()?
          };

          if !is_rec {
            self.declare_local(name);
          }

          arena.alloc(ExprCon::Bind {
            rec: is_rec,
            name,
            expr: body.inner,
            info: self.new_empty_info(),
          })
        }
        Keyword::If => {
          let condition = self.parse_expr()?;

          let _ = self.expect_keyword(Keyword::Then, true)?;

          self.skip_newlines();

          let then_branch = self.parse_expr()?;

          self.skip_newlines();

          let _ = self.expect_keyword(Keyword::Else, true)?;

          self.skip_newlines();

          let else_branch = self.parse_expr()?;

          let _ = self.expect_keyword(Keyword::End, true)?;

          arena.alloc(ExprCon::If(
            condition.inner,
            then_branch.inner,
            else_branch.inner,
            self.new_empty_info(),
          ))
        }
        _ => return self.diag.fail(format!("unexpected keyword {}", lhs_token.inner)),
      },
      _ => return self.diag.fail(format!("unexpected token {}", lhs_token.inner)),
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
        _ => return self.diag.fail(format!("unexpected trailing token {}", op_token.inner)),
      };

      if let Some((laff, _)) = Affinity::get_postfix(op_str) {
        if laff < minaff {
          break;
        }

        self.skip_token();

        if op_str == "." {
          let tok = self.next_token()?;
          let info = self.new_empty_info();
          lhs = match tok.inner.tag {
            IntLiteral(n) => {
              let index = u32::try_from(n).ok().filter(|&n| n > 0).ok_or_else(|| {
                self.diag.error(format!("fields are numbered from 1, but got {n}"))
              })?;
              arena.alloc(ExprCon::Index { receiver: lhs, index, info })
            }
            Identifer => {
              let member = TokenStr::from_span(tok.inner.span);
              if self.peek_paired_open(Paired::Parenthesis) {
                self.skip_token();
                let args = arena.alloc_slice_clone(&self.parse_args(Paired::Parenthesis)?);
                arena.alloc(ExprCon::MemberApply { receiver: lhs, member, args, info })
              } else {
                arena.alloc(ExprCon::Member { receiver: lhs, member, info })
              }
            }
            _ => return self.diag.fail(format!("expected a member, but got {}", tok.inner)),
          };
        } else if op_token.inner.tag == PairedOpen(Paired::Brace) {
          let inits = arena.alloc_slice_clone(&self.parse_inits()?);
          lhs = arena.alloc(ExprCon::Construct { ty: lhs, inits, info: self.new_empty_info() });
        } else if let PairedOpen(po) = op_token.inner.tag {
          let exprs = self.parse_args(po)?;

          if let Some(op) = lhs_op {
            lhs = arena.alloc(ExprCon::OpApply {
              op: arena.alloc(ExprCon::Op(op.into(), self.new_empty_info())),
              pair: Some(po),
              args: arena.alloc_slice_clone(&exprs),
              info: self.new_empty_info(),
            });
          } else {
            lhs = arena.alloc(ExprCon::Apply {
              func: lhs,
              pair: Some(po),
              args: arena.alloc_slice_clone(&exprs),
              info: self.new_empty_info(),
            });
          }
        } else {
          let old_lhs: ExprRef<'_, InfoKey> = lhs;

          lhs = arena.alloc(ExprCon::OpApply {
            op: arena.alloc(ExprCon::Op(op_str.into(), self.new_empty_info())),
            pair: None,
            args: arena.alloc_slice_clone(&[old_lhs]),
            info: self.new_empty_info(),
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

        let old_lhs: ExprRef<'_, InfoKey> = lhs;

        lhs = arena.alloc(ExprCon::OpApply {
          op: arena.alloc(ExprCon::Op(op_str.into(), self.new_empty_info())),
          pair: None,
          args: arena.alloc_slice_clone(&[old_lhs, rhs.inner]),
          info: self.new_empty_info(),
        });

        lhs_op = None;

        continue;
      }

      break;
    }

    Ok(PeekResult { inner: lhs })
  }

  fn parse_exprs<'t>(&'t mut self) -> PeekExpr<'a> {
    let arena = self.arena;
    self.enter_scope();
    self.skip_newlines();

    let first_expr = self.parse_expr();
    if let Err(e) = first_expr {
      self.leave_scope();
      return Err(e);
    }
    let first_expr = first_expr?;

    if !self.peek_operator(";", false) {
      self.leave_scope();
      return Ok(first_expr);
    }

    self.skip_newlines();

    let mut exprs = vec![first_expr.inner];

    while self.peek_operator(";", false) {
      self.skip_token();

      self.skip_newlines();

      let next_expr = self.parse_expr();
      if let Err(e) = next_expr {
        self.leave_scope();
        return Err(e);
      }
      exprs.push(next_expr?.inner);
    }

    self.skip_newlines();

    self.leave_scope();

    Ok(PeekResult {
      inner: arena.alloc(ExprCon::Block(arena.alloc_slice_clone(&exprs), self.new_empty_info())),
    })
  }

  pub fn parse(mut self) -> Result<SynTree<'a, InfoKey>> {
    let root = self.parse_exprs()?;
    self.skip_newlines();
    self.expect_reach_eof()?;
    let information = self.information;
    Ok(SynTree { root: root.inner, information })
  }
}

#[cfg(test)]
mod tests {
  use std::rc::Rc;

  use super::*;

  fn test_parse_exprs(source: &str, expected_sexp_str: &str) {
    let arena = Bump::new();
    let diag = Rc::new(Diagnostic::new());
    let parser = Parser::new(&arena, diag, source);
    let tree = parser.parse().unwrap();
    assert_eq!(tree.root.to_sexp(&SexpPool::new()).to_string(), expected_sexp_str);
  }

  fn parse_fails(source: &str) {
    let arena = Bump::new();
    let diag = Rc::new(Diagnostic::new());
    assert!(Parser::new(&arena, diag, source).parse().is_err(), "{source}");
  }

  fn test_parse_with_info(source: &str, expected_sexp_str: &str) {
    let arena = Bump::new();
    let diag = Rc::new(Diagnostic::new());
    let parser = Parser::new(&arena, diag, source);
    let tree = parser.parse().unwrap();
    let info_expr = InfoExpr { expr: tree.root, map: &tree.information };
    let pool = SexpPool::new();
    let sexp = info_expr.to_sexp(&pool);
    assert_eq!(sexp.to_string(), expected_sexp_str);
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
    test_parse_exprs("f()", "(f)");
    test_parse_exprs("f[]", "(f)");
    test_parse_exprs("f{}", "(construct f)");
    test_parse_exprs("f{x}[y](z)", "(((construct f (init _ x)) y) z)");
    test_parse_exprs("f(x, y)(z)", "((f x y) z)");
    test_parse_exprs("()", "()");
    test_parse_exprs("[]", "[]");
    test_parse_exprs("{}", "{}");
    test_parse_exprs("let x = (); x", "(block (let x ()) x)");
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
    test_parse_exprs("let rec x = fn (x) x end", "(let-rec x (fn x (x) x))");
  }

  #[test]
  fn test_functions() {
    test_parse_exprs("fn () unit end", "(fn () unit)");
    test_parse_exprs("fn () () end", "(fn () ())");
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

  #[test]
  fn test_structs() {
    test_parse_exprs("type P = struct {x, y} end", "(type P (struct (fields x y)))");
    test_parse_exprs(
      "type P = struct {} with fn f(self) 1 end with fn g(self, n) n end end",
      "(type P (struct (fields) (method f (self) 1) (method g (self n) n)))",
    );
    test_parse_exprs(
      "type P = struct {\n  x,\n  y\n}\n  with fn f(self) self.x end\nend",
      "(type P (struct (fields x y) (method f (self) (member self x))))",
    );
    test_parse_exprs("p.x", "(member p x)");
    test_parse_exprs("p.x.y", "(member (member p x) y)");
    test_parse_exprs("p.f(1, 2).g", "(member (member-apply p f 1 2) g)");
    test_parse_exprs("p.x + 1 * p.f()", "(+ (member p x) (* 1 (member-apply p f)))");
    test_parse_exprs("-p.x", "(- (member p x))");
    test_parse_exprs("P{1, y = 2}", "(construct P (init _ 1) (init y 2))");
    test_parse_exprs("P{}", "(construct P)");
    test_parse_exprs("P{\n  x = 1,\n  y = 2\n}", "(construct P (init x 1) (init y 2))");
    test_parse_exprs("P{x = 1}.x", "(member (construct P (init x 1)) x)");
    parse_fails("type P = struct {x} with fn f() 1 end end");
    parse_fails("type P = struct {x} with fn f(this) 1 end end");
  }

  #[test]
  fn test_index() {
    test_parse_exprs("t.1", "(index t 1)");
    test_parse_exprs("t.1.2", "(index (index t 1) 2)");
    test_parse_exprs("t.1(2)", "((index t 1) 2)");
    test_parse_exprs("p.x.1", "(index (member p x) 1)");
    parse_fails("t.0");
    parse_fails("t.-1");
  }

  #[test]
  fn test_method_freevars() {
    // Method names of the recursion group resolve through `self`, so they are not free.
    test_parse_with_info(
      "type P = struct {x} with fn f(self) g(k) + self.g(1) end with fn g(self, n) n end end",
      "(type P (struct (fields x) (fn f (self) (+ (g k) (member-apply self g 1)) (freevars k)) (fn g (self n) n)))",
    );
    test_parse_with_info(
      "type P = struct {} with fn f(self) fn () g() end end with fn g(self) 1 end end",
      "(type P (struct (fields) (fn f (self) (fn () (g) (freevars g))) (fn g (self) 1)))",
    );
    test_parse_with_info(
      "fn (k) type P = struct {} with fn f(self) g() + k end with fn g(self) 1 end end end",
      "(fn (k) (type P (struct (fields) (fn f (self) (+ (g) k) (freevars k)) (fn g (self) 1))))",
    );
  }

  #[test]
  fn test_freevars() {
    test_parse_with_info("fn (x) y + z end", "(fn (x) (+ y z) (freevars y z))");
    test_parse_with_info(
      "fn (x) fn (y) x + y + z end end",
      "(fn (x) (fn (y) (+ (+ x y) z) (freevars x z)) (freevars z))",
    );
    test_parse_with_info(
      "fn (x) let x = 1; x + y end",
      "(fn (x) (block (let x 1) (+ x y)) (freevars y))",
    );
    test_parse_with_info(
      "fn (a) fn (b) fn (c) a + b + c + d end end end",
      "(fn (a) (fn (b) (fn (c) (+ (+ (+ a b) c) d) (freevars a b d)) (freevars a d)) (freevars d))",
    );
    test_parse_with_info(
      "fn (y) let rec f = fn (x) f(x) + y + z end; f(1) end",
      "(fn (y) (block (let-rec f (fn f (x) (+ (+ (f x) y) z) (freevars y z))) (f 1)) (freevars z))",
    );
    test_parse_with_info(
      "fn (x) let y = 1; x + y + z end",
      "(fn (x) (block (let y 1) (+ (+ x y) z)) (freevars z))",
    );
  }
}
