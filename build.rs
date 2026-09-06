use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
  let profile = env::var("PROFILE").unwrap();
  let release = profile == "release";

  let mut build = cc::Build::new();

  build.flag("-std=c2x").flag("-Wall").flag("-Wextra");

  if release {
    build.flag("-O3").define("NDEBUG", None::<&str>);
  } else {
    build.flag("-O0");
  }

  // VM configuration macros, one file per architecture, shared with vm/Makefile
  // and vm/scripts/asm.py.
  let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
  let flags = PathBuf::from("vm/flags").join(&arch);
  let text = fs::read_to_string(&flags).unwrap_or_else(|_| panic!("missing {}", flags.display()));
  for flag in text.split_whitespace() {
    build.flag(flag);
  }
  println!("cargo:rerun-if-changed={}", flags.display());

  println!("cargo:rerun-if-changed=build.rs");

  let vm_src_dir = PathBuf::from("vm/src");

  if let Ok(entries) = fs::read_dir(&vm_src_dir) {
    for entry in entries.flatten() {
      let path = entry.path();
      if let Some(ext) = path.extension() {
        if ext == "c" {
          build.file(&path);
          println!("cargo:rerun-if-changed={}", path.display());
        } else if ext == "h" || ext == "def" {
          println!("cargo:rerun-if-changed={}", path.display());
        }
      }
    }
  }
  build.include(&vm_src_dir);
  build.compile("qxq_vm");

  let bindings_builder = bindgen::Builder::default()
    .header(vm_src_dir.join("vm.h").to_str().unwrap())
    .clang_arg("-Ivm/src")
    .clang_arg("-std=c2x")
    .allowlist_function("vm_entry")
    .allowlist_function("vm_exec_with")
    .allowlist_function("vm_thunk_alloc")
    .allowlist_function("vm_thunk_free")
    .allowlist_function("vm_type_alloc")
    .allowlist_function("vm_type_free")
    .allowlist_function("vm_const_from_i64")
    .allowlist_function("vm_const_from_f64")
    .allowlist_function("vm_heap_alloc")
    .allowlist_function("vm_heap_free")
    .allowlist_function("vm_const_from_str")
    .allowlist_function("vm_format_result")
    .allowlist_function("vm_object_size_for_fields")
    .allowlist_function("vm_status_name")
    .allowlist_type("state")
    .allowlist_type("fiber_segment")
    .allowlist_type("status_t")
    .allowlist_type("heap")
    .allowlist_type("thunk")
    .allowlist_type("type_desc")
    .allowlist_type("member_desc")
    .allowlist_var("dispatch")
    .rust_target(bindgen::RustTarget::stable(82, 0).unwrap())
    .raw_line("#![allow(non_upper_case_globals)]")
    .raw_line("#![allow(non_camel_case_types)]")
    .raw_line("#![allow(non_snake_case)]")
    .raw_line("#![allow(dead_code)]")
    .raw_line("#![allow(unused_imports)]")
    .raw_line("#![allow(unsafe_op_in_unsafe_fn)]");

  let bindings = bindings_builder.generate().expect("Unable to generate bindings");

  let bindings_path = PathBuf::from("src").join("generated").join("vm.rs");
  bindings.write_to_file(&bindings_path).expect("Couldn't write bindings!");
}
