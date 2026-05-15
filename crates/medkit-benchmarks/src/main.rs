#![forbid(unsafe_code)]

use std::{env, ffi::OsString, path::PathBuf, process};

use medkit_benchmarks::{
    fixtures::temp_fixture_root,
    macrobench::{run_cli_macrobench, MacrobenchConfig},
    Result,
};

fn main() {
    process::exit(match run(env::args_os()) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("{error}");
            2
        }
    });
}

fn run(args: impl IntoIterator<Item = OsString>) -> Result<()> {
    let mut args = args.into_iter();
    let _program = args.next();
    let Some(command) = args.next() else {
        return Err(usage().into());
    };
    match command.to_string_lossy().as_ref() {
        "run" => {
            let config = parse_run(args)?;
            let report = run_cli_macrobench(&config)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        "--help" | "-h" | "help" => {
            println!("{}", usage());
            Ok(())
        }
        other => Err(format!("unknown command: {other}\n\n{}", usage()).into()),
    }
}

fn parse_run(args: impl Iterator<Item = OsString>) -> Result<MacrobenchConfig> {
    let mut config = MacrobenchConfig::new(temp_fixture_root("macrobench"));
    let mut args = args.peekable();
    while let Some(flag) = args.next() {
        match flag.to_string_lossy().as_ref() {
            "--root" => config.synthetic.root = next_path(&mut args, "--root")?,
            "--cases" => {
                config.synthetic.cases =
                    parse_usize(&next_string(&mut args, "--cases")?, "--cases")?
            }
            "--shape" => {
                config.synthetic.shape =
                    parse_usize3(&next_string(&mut args, "--shape")?, "--shape")?
            }
            "--spacing" => {
                config.synthetic.spacing =
                    parse_f32_3(&next_string(&mut args, "--spacing")?, "--spacing")?
            }
            "--cache-shape" => {
                config.synthetic.cache_shape =
                    parse_usize3(&next_string(&mut args, "--cache-shape")?, "--cache-shape")?
            }
            "--resample-spacing" => {
                config.synthetic.resample_spacing = parse_f64_3(
                    &next_string(&mut args, "--resample-spacing")?,
                    "--resample-spacing",
                )?
            }
            "--patch" => {
                config.patch_size = parse_usize3(&next_string(&mut args, "--patch")?, "--patch")?
            }
            "--samples" => {
                config.samples = parse_usize(&next_string(&mut args, "--samples")?, "--samples")?
            }
            "--workers" => {
                config.workers = parse_usize(&next_string(&mut args, "--workers")?, "--workers")?
            }
            "--medkit-bin" => config.medkit_bin = Some(next_path(&mut args, "--medkit-bin")?),
            "--out" => config.out_path = Some(next_path(&mut args, "--out")?),
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unknown argument: {other}\n\n{}", usage()).into()),
        }
    }
    Ok(config)
}

fn next_path(
    args: &mut std::iter::Peekable<impl Iterator<Item = OsString>>,
    flag: &str,
) -> Result<PathBuf> {
    Ok(PathBuf::from(next_string(args, flag)?))
}

fn next_string(
    args: &mut std::iter::Peekable<impl Iterator<Item = OsString>>,
    flag: &str,
) -> Result<String> {
    args.next()
        .map(|value| value.to_string_lossy().into_owned())
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn parse_usize(value: &str, flag: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid integer for {flag}: {value}").into())
}

fn parse_usize3(value: &str, flag: &str) -> Result<[usize; 3]> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!("{flag} must be formatted as x,y,z, got {value}").into());
    }
    Ok([
        parse_usize(parts[0], flag)?,
        parse_usize(parts[1], flag)?,
        parse_usize(parts[2], flag)?,
    ])
}

fn parse_f32_3(value: &str, flag: &str) -> Result<[f32; 3]> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!("{flag} must be formatted as x,y,z, got {value}").into());
    }
    Ok([
        parse_f32(parts[0], flag)?,
        parse_f32(parts[1], flag)?,
        parse_f32(parts[2], flag)?,
    ])
}

fn parse_f64_3(value: &str, flag: &str) -> Result<[f64; 3]> {
    let parts = value.split(',').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!("{flag} must be formatted as x,y,z, got {value}").into());
    }
    Ok([
        parse_f64(parts[0], flag)?,
        parse_f64(parts[1], flag)?,
        parse_f64(parts[2], flag)?,
    ])
}

fn parse_f32(value: &str, flag: &str) -> Result<f32> {
    value
        .parse::<f32>()
        .map_err(|_| format!("invalid float for {flag}: {value}").into())
}

fn parse_f64(value: &str, flag: &str) -> Result<f64> {
    value
        .parse::<f64>()
        .map_err(|_| format!("invalid float for {flag}: {value}").into())
}

fn usage() -> String {
    "Usage:\n  medkit-benchmarks run [--root DIR] [--cases N] [--shape X,Y,Z] [--spacing X,Y,Z] [--cache-shape X,Y,Z] [--resample-spacing X,Y,Z] [--patch X,Y,Z] [--samples N] [--workers N] [--medkit-bin PATH] [--out report.json]\n\nExample:\n  cargo build -p medkit-cli -p medkit-benchmarks\n  cargo run -p medkit-benchmarks -- run --cases 4 --shape 64,64,64 --patch 32,32,32 --samples 256 --workers 4".to_string()
}
