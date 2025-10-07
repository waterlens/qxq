pub struct Diagnostic;

impl Default for Diagnostic {
  fn default() -> Self {
    Self::new()
  }
}

impl Diagnostic {
  pub fn new() -> Self {
    Self {}
  }

  pub fn error(&self, message: &str) -> ! {
    panic!("error: {message}");
  }
}
