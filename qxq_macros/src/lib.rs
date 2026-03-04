use proc_macro::TokenStream;
use quote::quote;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use syn::parse::{Parse, ParseStream};
use syn::{Expr, Ident, LitStr, Token, braced, parenthesized, parse_macro_input};

struct DisplayInfo {
  fmt: LitStr,
  name_expr: Expr,
  args: syn::punctuated::Punctuated<Expr, Token![,]>,
}

struct BytecodeEntry {
  name: Ident,
  op_info: OpInfo,
  fn_name: Ident,
  params: syn::punctuated::Punctuated<syn::FnArg, Token![,]>,
  construction: proc_macro2::TokenStream,
  display: DisplayInfo,
}

enum OpInfo {
  None,
  Tripple { op_enum: Ident, op_struct: Ident, op_var: Ident },
}

impl Parse for OpInfo {
  fn parse(input: ParseStream) -> syn::Result<Self> {
    let content;
    parenthesized!(content in input);
    let id: Ident = content.parse()?;
    if id == "N" {
      Ok(OpInfo::None)
    } else {
      content.parse::<Token![,]>()?;
      let op_struct: Ident = content.parse()?;
      content.parse::<Token![,]>()?;
      let op_var: Ident = content.parse()?;
      Ok(OpInfo::Tripple { op_enum: id, op_struct, op_var })
    }
  }
}

impl Parse for BytecodeEntry {
  fn parse(input: ParseStream) -> syn::Result<Self> {
    let name: Ident = input.parse()?;
    let op_info: OpInfo = input.parse()?;

    // Parse "fn name(args)"
    input.parse::<Token![fn]>()?;
    let fn_name: Ident = input.parse()?;

    let content_params;
    parenthesized!(content_params in input);
    let params = content_params.parse_terminated(syn::FnArg::parse, Token![,])?;

    let content_braces;
    braced!(content_braces in input);
    let construction: proc_macro2::TokenStream = content_braces.parse()?;

    input.parse::<Token![=>]>()?;

    let content_display;
    parenthesized!(content_display in input);
    let display_fmt: LitStr = content_display.parse()?;
    content_display.parse::<Token![,]>()?;
    let name_expr: Expr = content_display.parse()?;

    // Use parse_terminated to handle remaining arguments, but first check if there's a comma
    let mut args = syn::punctuated::Punctuated::new();
    if content_display.peek(Token![,]) {
      content_display.parse::<Token![,]>()?;
      args = content_display.parse_terminated(Expr::parse, Token![,])?;
    }

    // Consume optional trailing comma between entries
    if input.peek(Token![,]) {
      input.parse::<Token![,]>()?;
    }

    Ok(BytecodeEntry {
      name,
      op_info,
      fn_name,
      params,
      construction,
      display: DisplayInfo { fmt: display_fmt, name_expr, args },
    })
  }
}

struct BytecodeList {
  entries: Vec<BytecodeEntry>,
}

impl Parse for BytecodeList {
  fn parse(input: ParseStream) -> syn::Result<Self> {
    let mut entries = Vec::new();
    while !input.is_empty() {
      entries.push(input.parse()?);
    }
    Ok(BytecodeList { entries })
  }
}

#[proc_macro]
pub fn define_bytecode(input: TokenStream) -> TokenStream {
  let list = parse_macro_input!(input as BytecodeList);

  // Generate C header if feature enabled
  #[cfg(feature = "gen-c-bclist")]
  generate_c_header(&list.entries);

  // Generate Rust code
  let operator_variants: Vec<_> = list.entries.iter().map(|e| &e.name).collect();

  let constructors: Vec<_> = list
    .entries
    .iter()
    .map(|e| {
      let name = &e.name;
      let fn_name = &e.fn_name;
      let params = &e.params;
      let construction = &e.construction;
      match &e.op_info {
        OpInfo::None => quote! {
            pub fn #fn_name(#params) -> Self {
                Self(Operator::#name, Operands::N)
            }
        },
        OpInfo::Tripple { op_enum, op_struct, .. } => quote! {
            pub fn #fn_name(#params) -> Self {
                Self(
                    Operator::#name,
                    Operands::#op_enum(#op_struct { #construction })
                )
            }
        },
      }
    })
    .collect();

  let display_matches: Vec<_> = list
    .entries
    .iter()
    .map(|e| {
      let name = &e.name;
      let fmt = &e.display.fmt;
      let name_expr = &e.display.name_expr;
      let args = e.display.args.iter();
      match &e.op_info {
        OpInfo::None => quote! {
            (Operator::#name, Operands::N) => {
                write!(f, #fmt, #name_expr #(, #args)*)?;
            }
        },
        OpInfo::Tripple { op_enum, op_var, .. } => quote! {
            (Operator::#name, Operands::#op_enum(#op_var)) => {
                write!(f, #fmt, #name_expr #(, #args)*)?;
            }
        },
      }
    })
    .collect();

  let load_arms: Vec<_> = list
    .entries
    .iter()
    .map(|e| {
      let name = &e.name;
      match &e.op_info {
        OpInfo::None => quote! {
            code if code == Operator::#name as u8 => Ok(Self(Operator::#name, Operands::N)),
        },
        OpInfo::Tripple { op_enum, op_struct, .. } => quote! {
            code if code == Operator::#name as u8 => {
                Ok(Self(Operator::#name, Operands::#op_enum(#op_struct::load(&buf[1..])?)))
            }
        },
      }
    })
    .collect();

  let expanded = quote! {
      #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
      #[repr(u8)]
      pub enum Operator {
          #( #operator_variants, )*
      }

      impl Operator {
          pub fn opcode(&self) -> u8 {
              *self as u8
          }
      }

      impl Bytecode {
          #( #constructors )*
      }

      impl std::fmt::Display for Bytecode {
          fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
              match (self.0, &self.1) {
                  #( #display_matches )*
                  _ => unreachable!("Mismatched Operator/Operands combination"),
              }
              Ok(())
          }
      }

      impl BinaryRepr for Bytecode {
          fn dump(&self, buf: &mut [u8]) {
              match self.0 {
                  #(
                      Operator::#operator_variants => {
                          buf[0] = self.0.opcode();
                          self.1.dump(&mut buf[1..]);
                      }
                  )*
              }
          }

          fn load(buf: &[u8]) -> DResult<Self> {
              let opcode = buf.get(0).ok_or_else(|| anyhow::anyhow!("Buffer underflow while loading bytecode"))?;
              match *opcode {
                  #( #load_arms )*
                  unknown => Err(anyhow::anyhow!("Unknown opcode: {}", unknown))
              }
          }
      }
  };

  expanded.into()
}

#[cfg(feature = "gen-c-bclist")]
fn generate_c_header(entries: &[BytecodeEntry]) {
  let mut content = String::new();
  content.push_str("#define OPS(_) \\\n");

  for entry in entries {
    let name = entry.name.to_string();
    let print_name =
      if let Expr::Lit(syn::ExprLit { lit: syn::Lit::Str(s), .. }) = &entry.display.name_expr {
        s.value()
      } else {
        name.to_string().to_lowercase()
      };

    content.push_str(&format!("  _({:<10}, {:<12})\\\n", name, format!("\"{}\"", print_name),));
  }

  if content.ends_with(" \\\n") {
    content.pop();
    content.pop();
    content.pop();
    content.push('\n');
  }

  content.push_str("// clang-format on\n\n");

  // Locate the workspace root
  let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
  let mut path = PathBuf::from(manifest_dir);

  path.push("vm");
  path.push("src");
  path.push("bclist.def");

  match File::create(&path) {
    Ok(mut file) => {
      file.write_all(content.as_bytes()).unwrap();
    }
    Err(e) => {
      panic!("Failed to generate C bytecode list at {:?}: {}", path, e);
    }
  }
}
