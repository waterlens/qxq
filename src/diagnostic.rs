use anyhow::Context;
use anyhow::anyhow;

pub type Result<T> = anyhow::Result<T>;

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

  /// For internal compiler errors that should never happen (Compiler Bug).
  pub fn ice(&self, message: &str) -> ! {
    panic!("Internal Compiler Error: {message}");
  }

  /// For fatal user errors that stop compilation immediately.
  pub fn fatal(&self, message: &str) -> ! {
    panic!("Fatal Error: {message}");
  }

  /// For non-fatal reporting (just printing).
  pub fn report(&self, message: &str) {
    eprintln!("Error: {message}");
  }

  /// Unified entry point for printing anyhow errors.
  pub fn report_err(&self, err: &anyhow::Error) {
    eprintln!("Error: {err:?}");
  }

  /// Returns Err(anyhow!(...)), replacing anyhow::bail!
  pub fn fail<T>(&self, message: impl Into<String>) -> Result<T> {
    Err(anyhow!(message.into()))
  }

  /// Returns a raw anyhow::Error for use in closures (like ok_or_else).
  /// Replaces anyhow::anyhow!
  pub fn error(&self, message: impl Into<String>) -> anyhow::Error {
    anyhow!(message.into())
  }

  /// Adds context to a result, replacing .with_context()
  pub fn context<T, E: Into<anyhow::Error>>(
    &self,
    result: std::result::Result<T, E>,
    message: impl Into<String>,
  ) -> Result<T> {
    result.map_err(|e| e.into()).context(message.into())
  }
}
