use bumpalo::Bump;

pub struct SexpPool {
  arena: Bump,
}
pub trait ToSexp {
  fn to_sexp<'pool>(&self, pool: &'pool SexpPool) -> Sexp<'pool>;
}

impl<T: ToSexp> ToSexp for Option<T> {
  fn to_sexp<'pool>(&self, pool: &'pool SexpPool) -> Sexp<'pool> {
    match self {
      Some(x) => pool.list(&[pool.atom("?some"), x.to_sexp(pool)]),
      None => pool.atom("?none"),
    }
  }
}

impl<T: ToSexp> ToSexp for &[T] {
  fn to_sexp<'pool>(&self, pool: &'pool SexpPool) -> Sexp<'pool> {
    pool.list(self.iter().map(|x| x.to_sexp(pool)).collect::<Vec<_>>())
  }
}

impl<T: ToSexp> ToSexp for &T {
  fn to_sexp<'pool>(&self, pool: &'pool SexpPool) -> Sexp<'pool> {
    (*self).to_sexp(pool)
  }
}

#[derive(Debug, Clone)]
pub enum Sexp<'a> {
  Atom(&'a str),
  List(&'a [Sexp<'a>]),
}

impl Default for SexpPool {
    fn default() -> Self {
        Self::new()
    }
}

impl SexpPool {
  pub fn new() -> Self {
    SexpPool { arena: Bump::new() }
  }

  pub fn atom<T: AsRef<str>>(&self, s: T) -> Sexp {
    Sexp::Atom(self.arena.alloc_str(s.as_ref()))
  }

  pub fn list<'s, 'pool, T>(&'pool self, xs: T) -> Sexp<'pool>
  where
    T: AsRef<[Sexp<'s>]>,
    's: 'pool,
  {
    Sexp::List(self.arena.alloc_slice_clone(xs.as_ref()))
  }

  pub fn list_from_iter<'s, 'pool, I>(&'pool self, xs: I) -> Sexp<'pool>
  where
    I: IntoIterator<Item = Sexp<'s>>,
    I::IntoIter: ExactSizeIterator,
    's: 'pool,
  {
    Sexp::List(self.arena.alloc_slice_fill_iter(xs))
  }

  pub fn non_empty_list<'s1, 's2, 'pool, S>(&'pool self, x: Sexp<'s1>, xs: S) -> Sexp<'pool>
  where
    S: AsRef<[Sexp<'s2>]>,
    's1: 'pool,
    's2: 'pool,
  {
    let mut v = Vec::with_capacity(1 + xs.as_ref().len());
    v.push(x);
    v.extend_from_slice(xs.as_ref());
    Sexp::List(self.arena.alloc_slice_clone(v.as_ref()))
  }

  pub fn non_empty_list_from_iter<'s1, 'pool, I>(&'pool self, x: Sexp<'s1>, xs: I) -> Sexp<'pool>
  where
    I: IntoIterator<Item = Sexp<'s1>>,
    I::IntoIter: ExactSizeIterator,
    's1: 'pool,
  {
    let mut v = Vec::new();
    v.push(x);
    v.extend(xs);
    Sexp::List(self.arena.alloc_slice_fill_iter(v))
  }
}

impl std::fmt::Display for Sexp<'_> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    use Sexp::*;
    match self {
      Atom(s) => write!(f, "{s}"),
      List(xs) => {
        write!(f, "(")?;
        let mut iter = xs.iter();
        if let Some(x) = iter.next() {
          write!(f, "{x}")?;
          for x in iter {
            write!(f, " {x}")?;
          }
        }
        write!(f, ")")
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sexp_pool() {
    let pool = SexpPool::new();
    let sexp = pool.list(vec![pool.atom("a"), pool.atom("b"), pool.list(vec![pool.atom("c")])]);
    assert_eq!(sexp.to_string(), "(a b (c))");
  }
}
