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

  /// For internal compiler errors that should never happen.
  pub fn error(&self, message: &str) -> ! {
    panic!("error: {message}");
  }

  /// For non-fatal reporting.
  pub fn report(&self, message: &str) {
    eprintln!("error: {message}");
  }

  /// Unified entry point for printing anyhow errors.
  pub fn report_anyhow(&self, err: &anyhow::Error) {
    eprintln!("error: {err}");
  }

  /// Replaces anyhow::bail!
  pub fn fail<T>(&self, message: impl Into<String>) -> Result<T> {
    Err(anyhow!(message.into()))
  }

  /// Replaces .with_context()
  pub fn enrich<T, E: Into<anyhow::Error>>(
    &self,
    result: std::result::Result<T, E>,
    message: impl Into<String>,
  ) -> Result<T> {
    result.map_err(|e| e.into()).context(message.into())
  }
}
