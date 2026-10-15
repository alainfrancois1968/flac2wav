//! `flac2wav` command line.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use flac2wav::{decode_flac, encode_wav, worker_count, Error, Options};
use rayon::prelude::*;

#[derive(Parser)]
#[command(
    name = "flac2wav",
    version,
    about = "Convert FLAC to WAV with parallel decoding",
    arg_required_else_help = true
)]
struct Args {
    /// FLAC files to convert.
    #[arg(required = true, value_name = "FILE")]
    inputs: Vec<PathBuf>,

    /// Output WAV file, or a directory when converting more than one file.
    /// Defaults to each input name with a .wav extension.
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Decode threads, from 1 to 256. 0 uses one per logical CPU.
    #[arg(short = 'j', long, default_value_t = 0, value_name = "N")]
    jobs: usize,

    /// Replace existing WAV files.
    #[arg(short, long)]
    force: bool,

    /// Do not print per-file results.
    #[arg(short, long)]
    quiet: bool,

    /// Do not check the MD5 stored in STREAMINFO.
    #[arg(long)]
    ignore_md5: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("flac2wav: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), Error> {
    let jobs = worker_count(args.jobs);
    let pairs = destinations(&args.inputs, args.output.as_deref())?;
    ensure_outputs(&pairs, args.force)?;

    let started = Instant::now();
    let multiple = pairs.len() > 1;
    let check_md5 = !args.ignore_md5;
    let results: Vec<Result<String, Error>> = if multiple {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .map_err(|err| Error::Decode(format!("could not start worker threads: {err}")))?;
        pool.install(|| {
            pairs
                .par_iter()
                .map(|(input, output)| convert_file(input, output, 1, check_md5, false))
                .collect()
        })
    } else {
        let (input, output) = &pairs[0];
        vec![convert_file(input, output, jobs, check_md5, true)]
    };

    let mut failed = 0usize;
    for result in &results {
        match result {
            Ok(line) => {
                if !args.quiet {
                    println!("{line}");
                }
            }
            Err(err) => {
                failed += 1;
                eprintln!("flac2wav: {err}");
            }
        }
    }

    if multiple && !args.quiet {
        let ok = pairs.len() - failed;
        println!(
            "done  {ok} of {} files  {jobs} threads  {}",
            pairs.len(),
            format_elapsed(started.elapsed())
        );
    }

    if failed > 0 {
        let noun = if failed == 1 { "file" } else { "files" };
        return Err(Error::Invalid(format!("{failed} {noun} failed")));
    }
    Ok(())
}

fn convert_file(
    input: &Path,
    output: &Path,
    threads: usize,
    check_md5: bool,
    show_threads: bool,
) -> Result<String, Error> {
    let started = Instant::now();
    let bytes = fs::read(input)
        .map_err(|err| Error::Invalid(format!("cannot read {}: {err}", input.display())))?;
    let pcm = decode_flac(&bytes, &Options { threads, check_md5 })
        .map_err(|err| Error::Invalid(format!("{}: {err}", input.display())))?;
    let wav = encode_wav(&pcm)?;
    write_atomic(output, &wav)?;

    let name_in = file_name(input);
    let name_out = file_name(output);
    let threads_part = if show_threads {
        format!("  {threads} threads")
    } else {
        String::new()
    };
    Ok(format!(
        "ok  {name_in} -> {name_out}  {} Hz  {}  {}-bit  {}{threads_part}  {}",
        pcm.sample_rate,
        channel_label(pcm.channels),
        pcm.bits_per_sample,
        human_size(wav.len()),
        format_elapsed(started.elapsed())
    ))
}

fn destinations(
    inputs: &[PathBuf],
    output: Option<&Path>,
) -> Result<Vec<(PathBuf, PathBuf)>, Error> {
    if inputs.is_empty() {
        return Err(Error::Invalid("no input files".into()));
    }
    let multiple = inputs.len() > 1;
    let pairs = match output {
        Some(path) if !multiple && is_wav(path) => {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            vec![(inputs[0].clone(), path.to_path_buf())]
        }
        Some(path) if multiple && is_wav(path) && !path.is_dir() => {
            return Err(Error::Invalid(
                "-o must be a directory when converting more than one file".into(),
            ));
        }
        Some(dir) => {
            fs::create_dir_all(dir)?;
            inputs
                .iter()
                .map(|input| (input.clone(), dir.join(wav_file_name(input))))
                .collect()
        }
        None => inputs
            .iter()
            .map(|input| (input.clone(), input.with_extension("wav")))
            .collect(),
    };

    let mut seen = std::collections::HashSet::new();
    for (_, output) in &pairs {
        if !seen.insert(output.clone()) {
            return Err(Error::Invalid(format!(
                "multiple inputs would write {}",
                output.display()
            )));
        }
    }
    Ok(pairs)
}

fn ensure_outputs(pairs: &[(PathBuf, PathBuf)], force: bool) -> Result<(), Error> {
    for (input, output) in pairs {
        if !input.is_file() {
            return Err(Error::Invalid(format!("{} is not a file", input.display())));
        }
        if same_file(input, output) {
            return Err(Error::Invalid(format!(
                "{} would overwrite the input file",
                output.display()
            )));
        }
        if output.exists() && !force {
            return Err(Error::Invalid(format!(
                "{} already exists (pass --force to replace it)",
                output.display()
            )));
        }
    }
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(".partial");
    let tmp = PathBuf::from(tmp_name);
    fs::write(&tmp, bytes)?;
    if let Err(err) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(err.into());
    }
    Ok(())
}

fn same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

fn is_wav(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
}

fn wav_file_name(input: &Path) -> PathBuf {
    let mut name = input.file_stem().unwrap_or_default().to_os_string();
    name.push(".wav");
    PathBuf::from(name)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn channel_label(channels: u32) -> String {
    match channels {
        1 => "mono".to_string(),
        2 => "stereo".to_string(),
        n => format!("{n} ch"),
    }
}

fn human_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn format_elapsed(elapsed: std::time::Duration) -> String {
    format!("{:.1} ms", elapsed.as_secs_f64() * 1000.0)
}
