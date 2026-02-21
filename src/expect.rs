use anyhow::{anyhow, Result};
use std::io::{self, BufRead};
use std::process::Command;

pub struct Expectation {
  pub content: String,
  pub start: usize,
  pub end: usize,
}

/// The "FileCheck" mode: reads from stdin and matches against a file.
pub fn run_check(file_path: &str) -> Result<()> {
  let content = std::fs::read_to_string(file_path)?;
  let (exps, _, err_pat) = extract_metadata(&content)?;

  let res = if exps.is_empty() {
    Err(anyhow!("No EXPECT: patterns found in {}", file_path))
  } else {
    verify_output(io::stdin().lock(), &exps)
  };

  handle_negative_test(res, err_pat)
}

/// The "Driver" mode: reads RUN:, executes it, and verifies output.
pub fn run_test_file(file_path: &str) -> Result<()> {
  let content = std::fs::read_to_string(file_path)?;
  let (exps, run_cmd, err_pat) = extract_metadata(&content)?;

  let res = run_test_file_core(file_path, &exps, run_cmd.as_deref(), err_pat.is_some());
  handle_negative_test(res, err_pat)
}

fn run_test_file_core(
  file_path: &str,
  exps: &[Expectation],
  run_cmd: Option<&str>,
  is_neg: bool,
) -> Result<()> {
  let cmd_raw = run_cmd.ok_or_else(|| anyhow!("No RUN: block found in {}", file_path))?;
  if exps.is_empty() && !is_neg {
    return Err(anyhow!("No EXPECT: patterns found in {}", file_path));
  }

  let cmd = resolve_run_line(cmd_raw, file_path)?;
  let out = execute_shell(&cmd)?;
  let combined =
    [String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)].join("\n");

  if !out.status.success() {
    return Err(anyhow!("RUN command failed: {}\nError: {}", cmd, combined));
  }

  verify_output(io::Cursor::new(out.stdout), exps)
    .map_err(|e| anyhow!("Test failed for {}: {}", file_path, e))
}

fn handle_negative_test(res: Result<()>, err_pat: Option<String>) -> Result<()> {
  match (res, err_pat) {
    (Ok(_), Some(pat)) => Err(anyhow!("Expected error matching '{}', but test passed", pat)),
    (Err(e), Some(pat)) => {
      let msg = e.to_string();
      if msg.contains(&pat) {
        Ok(())
      } else {
        Err(anyhow!("Expected error matching '{}', but got '{}'", pat, msg))
      }
    }
    (r, None) => r,
  }
}

pub enum UpdateStatus {
  Updated,
  Skipped,
}

pub fn update_expectations(file_path: &str, skip_multi: bool) -> Result<UpdateStatus> {
  let content = std::fs::read_to_string(file_path)?;
  let (exps, run_cmd, _) = extract_metadata(&content)?;

  if exps.len() > 1 {
    if skip_multi {
      println!("Skipping update for {} (multiple EXPECT: blocks)", file_path);
      return Ok(UpdateStatus::Skipped);
    }
    return Err(anyhow!("Cannot automatically update file with multiple EXPECT: blocks"));
  }

  let cmd_raw = run_cmd.ok_or_else(|| anyhow!("No RUN: block found in {}", file_path))?;
  let cmd = resolve_run_line(&cmd_raw, file_path)?;
  let out = execute_shell(&cmd)?;
  if !out.status.success() {
    return Err(anyhow!(
      "RUN command failed: {}\nError: {}",
      cmd,
      String::from_utf8_lossy(&out.stderr)
    ));
  }

  let new_output = String::from_utf8_lossy(&out.stdout);
  if let Some(exp) = exps.first() {
    let chars: Vec<char> = content.chars().collect();
    let mut new_chars = Vec::new();
    new_chars.extend_from_slice(&chars[..exp.start]);
    new_chars.push('\n');
    for line in new_output.trim_end().lines() {
      new_chars.extend("   ".chars());
      new_chars.extend(line.chars());
      new_chars.push('\n');
    }
    new_chars.extend_from_slice(&chars[exp.end..]);
    std::fs::write(file_path, new_chars.into_iter().collect::<String>())?;
    println!("Updated EXPECT block in {}", file_path);
    Ok(UpdateStatus::Updated)
  } else {
    Err(anyhow!("No EXPECT: block found to update in {}", file_path))
  }
}

fn verify_output<R: BufRead>(mut input: R, expectations: &[Expectation]) -> Result<()> {
  for (i, exp) in expectations.iter().enumerate() {
    for expected_line in exp.content.lines().map(|l| l.trim()).filter(|l| !l.is_empty()) {
      let mut found = false;
      let mut line = String::new();
      while input.read_line(&mut line)? > 0 {
        if line.contains(expected_line) {
          found = true;
          line.clear();
          break;
        }
        line.clear();
      }
      if !found {
        return Err(anyhow!(
          "expectation #{} line '{}' not found in the remaining output",
          i + 1,
          expected_line
        ));
      }
    }
  }
  Ok(())
}

fn execute_shell(cmd: &str) -> io::Result<std::process::Output> {
  if cfg!(target_os = "windows") {
    Command::new("cmd").args(["/C", cmd]).output()
  } else {
    Command::new("sh").args(["-c", cmd]).output()
  }
}

fn command_exists(cmd: &str) -> bool {
  let cmd = cmd.trim();
  if cmd.is_empty() {
    return false;
  }
  let check_cmd = if cfg!(target_os = "windows") { "where" } else { "which" };
  Command::new(check_cmd).arg(cmd).output().map(|o| o.status.success()).unwrap_or(false)
}

fn resolve_run_line(run_line: &str, file_path: &str) -> Result<String> {
  let mut s = Scanner::new(run_line);
  let mut result = String::new();
  while !s.is_eof() {
    match s.ch() {
      '{' => {
        s.advance();
        let mut preferred = String::new();
        let mut fallback = String::new();
        let mut in_fallback = false;
        let mut depth = 1;
        while !s.is_eof() && depth > 0 {
          match s.ch() {
            '{' => {
              depth += 1;
              if in_fallback {
                fallback.push('{');
              } else {
                preferred.push('{');
              }
              s.advance();
            }
            '}' => {
              depth -= 1;
              if depth > 0 {
                if in_fallback {
                  fallback.push('}');
                } else {
                  preferred.push('}');
                }
              }
              s.advance();
            }
            '|' if depth == 1 => {
              in_fallback = true;
              s.advance();
            }
            c => {
              if in_fallback {
                fallback.push(c);
              } else {
                preferred.push(c);
              }
              s.advance();
            }
          }
        }
        if in_fallback {
          if command_exists(preferred.trim()) {
            result.push_str(preferred.trim());
          } else {
            result.push_str(fallback.trim());
          }
        } else {
          result.push_str(preferred.trim());
        }
      }
      '%' => {
        s.advance();
        if s.ch() == 's' {
          result.push_str(file_path);
          s.advance();
        } else {
          result.push('%');
        }
      }
      c => {
        result.push(c);
        s.advance();
      }
    }
  }
  Ok(result)
}

struct Scanner {
  buffer: Vec<char>,
  index: usize,
}

impl Scanner {
  fn new(content: &str) -> Self {
    let mut buffer: Vec<char> = content.chars().collect();
    buffer.push('\0');
    Self { buffer, index: 0 }
  }
  fn ch(&self) -> char {
    self.buffer[self.index]
  }
  fn advance(&mut self) {
    if self.ch() != '\0' {
      self.index += 1;
    }
  }
  fn is_eof(&self) -> bool {
    self.ch() == '\0'
  }
}

fn extract_metadata(content: &str) -> Result<(Vec<Expectation>, Option<String>, Option<String>)> {
  let mut expectations = Vec::new();
  let mut run_cmd = None;
  let mut expected_error = None;
  let mut s = Scanner::new(content);

  while !s.is_eof() {
    match s.ch() {
      '(' => {
        s.advance();
        if s.ch() == '*' {
          s.advance();
          while s.ch().is_whitespace() {
            s.advance();
          }
          let start_after_ws = s.index;

          if try_match(&mut s, "EXPECT:") {
            let start = s.index;
            while !s.is_eof() {
              if s.ch() == '*' {
                let check_idx = s.index;
                s.advance();
                if s.ch() == ')' {
                  s.index = check_idx;
                  break;
                }
              } else {
                s.advance();
              }
            }
            expectations.push(Expectation {
              content: s.buffer[start..s.index].iter().collect(),
              start,
              end: s.index,
            });
            if s.ch() == '*' {
              s.advance();
              s.advance();
            }
          } else {
            s.index = start_after_ws;
            let is_err = try_match(&mut s, "RUN-EXPECT-ERROR:");
            if !is_err {
              s.index = start_after_ws;
            }
            let is_run = is_err || try_match(&mut s, "RUN:");

            if is_run {
              if run_cmd.is_some() {
                return Err(anyhow!("Multiple RUN: or RUN-EXPECT-ERROR: blocks found"));
              }
              while s.ch().is_whitespace() {
                s.advance();
              }
              let start = s.index;
              while !s.is_eof() {
                if s.ch() == '*' {
                  let check_idx = s.index;
                  s.advance();
                  if s.ch() == ')' {
                    s.index = check_idx;
                    break;
                  }
                } else {
                  s.advance();
                }
              }
              let raw: String = s.buffer[start..s.index].iter().collect();
              if is_err {
                let mut lines = raw.lines();
                run_cmd = lines.next().map(|l| l.trim().to_string());
                expected_error = Some(lines.collect::<Vec<_>>().join("\n").trim().to_string());
              } else {
                run_cmd = Some(raw.trim().to_string());
              }
              if s.ch() == '*' {
                s.advance();
                s.advance();
              }
            } else {
              s.index = start_after_ws;
              let mut level = 1;
              loop {
                match s.ch() {
                  '(' => {
                    s.advance();
                    if s.ch() == '*' {
                      s.advance();
                      level += 1;
                    }
                  }
                  '*' => {
                    s.advance();
                    if s.ch() == ')' {
                      s.advance();
                      level -= 1;
                      if level == 0 {
                        break;
                      }
                    }
                  }
                  '\0' => break,
                  _ => s.advance(),
                }
              }
            }
          }
        }
      }
      _ => s.advance(),
    }
  }
  Ok((expectations, run_cmd, expected_error))
}

fn try_match(s: &mut Scanner, prefix: &str) -> bool {
  let start_idx = s.index;
  for c in prefix.chars() {
    if s.ch() != c {
      s.index = start_idx;
      return false;
    }
    s.advance();
  }
  true
}
