fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_CONFIG_FILE");
    reject_free_threaded_pyo3_config();
    if std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() == Some(std::ffi::OsStr::new("macos")) {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}

fn reject_free_threaded_pyo3_config() {
    let Some(config_path) = std::env::var_os("PYO3_CONFIG_FILE") else {
        return;
    };
    let Ok(config) = std::fs::read_to_string(&config_path) else {
        return;
    };
    let free_threaded = config.lines().any(|line| {
        line == "Py_GIL_DISABLED=1"
            || line == "gil_disabled=true"
            || line == "lib_name=python3.13t"
            || line == "lib_name=python3.14t"
    });
    if free_threaded {
        panic!(
            "medkit-python does not support free-threaded CPython with PyO3 0.22; use the repo .venv non-free-threaded Python or unset PYO3_CONFIG_FILE"
        );
    }
}
