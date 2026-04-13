use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
  let opt_level = env::var("OPT_LEVEL").unwrap();

  let mut build = cc::Build::new();

  build
    .flag("-std=c2x")
    .flag("-Wall")
    .flag("-Wextra")
    .define("JUMP_MODE", "0")
    .define("DECODE_MODE", "1");

  if opt_level != "0" {
    build.flag("-O3");
  } else {
    build.flag("-O0");
  }

  println!("cargo:rerun-if-changed=build.rs");

  let vm_src_dir = PathBuf::from("vm/src");

  if let Ok(entries) = fs::read_dir(&vm_src_dir) {
    for entry in entries.flatten() {
      let path = entry.path();
      if let Some(ext) = path.extension() {
        if ext == "c" {
          build.file(&path);
          println!("cargo:rerun-if-changed={}", path.display());
        } else if ext == "h" {
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
    .allowlist_function("vm_exec")
    .allowlist_function("vm_alloc_function")
    .allowlist_function("vm_free_function")
    .allowlist_function("vm_make_wrapper")
    .allowlist_function("vm_const_from_i64")
    .allowlist_function("vm_format_result")
    .allowlist_function("vm_status_name")
    .allowlist_type("state")
    .allowlist_type("status_t")
    .allowlist_type("heap")
    .allowlist_type("function")
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
