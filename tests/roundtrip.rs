//! Decoder output against PCM produced by an independent tool, plus the CLI.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use flac2wav::{decode_flac, flac_to_wav, probe, Options};

fn options(threads: usize) -> Options {
    Options {
        threads,
        check_md5: true,
    }
}

fn wav_payload(wav: &[u8]) -> &[u8] {
    assert!(wav.len() > 44, "WAV is shorter than a header");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(&wav[36..40], b"data");
    let data_len = u32::from_le_bytes(wav[40..44].try_into().unwrap()) as usize;
    assert_eq!(wav.len(), 44 + data_len);
    &wav[44..]
}

#[test]
fn stereo_matches_independent_pcm_on_one_and_many_threads() {
    let flac = include_bytes!("fixtures/noise_stereo16.flac");
    let expected = include_bytes!("fixtures/noise_stereo16.pcm");
    let single = flac_to_wav(flac, 1).unwrap();
    let parallel = flac_to_wav(flac, 8).unwrap();
    assert_eq!(wav_payload(&single), expected.as_slice());
    assert_eq!(single, parallel);

    let info = probe(flac).unwrap();
    assert_eq!(info.sample_rate, 44_100);
    assert_eq!(info.channels, 2);
    assert_eq!(info.bits_per_sample, 16);
    assert_eq!(u16::from_le_bytes(single[22..24].try_into().unwrap()), 2);
    assert_eq!(
        u32::from_le_bytes(single[24..28].try_into().unwrap()),
        44_100
    );
    assert_eq!(u16::from_le_bytes(single[34..36].try_into().unwrap()), 16);

    let pcm = decode_flac(flac, &options(4)).unwrap();
    assert!(pcm.ranges > 1);
    assert_eq!(pcm.interleaved.len(), 44_100 * 2 * 2);
}

#[test]
fn mono_24_bit_matches_independent_pcm() {
    let flac = include_bytes!("fixtures/sine_mono24.flac");
    let wav = flac_to_wav(flac, 4).unwrap();
    assert_eq!(
        wav_payload(&wav),
        include_bytes!("fixtures/sine_mono24.pcm").as_slice()
    );
    assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 48_000);
    assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 24);
    let pcm = decode_flac(flac, &options(1)).unwrap();
    assert_eq!(pcm.interleaved.len(), 48_000);
}

#[test]
fn tiny_file_stays_stable_with_extra_threads() {
    let flac = include_bytes!("fixtures/tiny_mono.flac");
    let one = decode_flac(flac, &options(1)).unwrap();
    let many = decode_flac(flac, &options(8)).unwrap();
    assert_eq!(one, many);
    assert_eq!(one.sample_rate, 44_100);
    assert_eq!(one.channels, 1);
}

#[test]
fn cli_writes_wav_refuses_clobber_and_batches() {
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!("flac2wav-{id}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    let bin = env!("CARGO_BIN_EXE_flac2wav");
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tiny = manifest.join("tests/fixtures/tiny_mono.flac");
    let noise = manifest.join("tests/fixtures/noise_stereo16.flac");
    let output = root.join("tiny.wav");

    let wrote = Command::new(bin)
        .arg("-j")
        .arg("4")
        .arg("-o")
        .arg(&output)
        .arg(&tiny)
        .status()
        .unwrap();
    assert!(wrote.success(), "convert failed");
    assert!(output.is_file());
    assert!(std::fs::metadata(&output).unwrap().len() > 44);

    let clobber = Command::new(bin)
        .arg("-o")
        .arg(&output)
        .arg(&tiny)
        .status()
        .unwrap();
    assert!(!clobber.success(), "overwrite should fail without --force");

    let forced = Command::new(bin)
        .args(["-f", "-q", "-o"])
        .arg(&output)
        .arg(&tiny)
        .status()
        .unwrap();
    assert!(forced.success(), "force should replace the WAV");

    let batch = root.join("batch");
    let batched = Command::new(bin)
        .args(["-j", "2", "-o"])
        .arg(&batch)
        .arg(&tiny)
        .arg(&noise)
        .status()
        .unwrap();
    assert!(batched.success(), "batch convert failed");
    assert!(batch.join("tiny_mono.wav").is_file());
    assert!(batch.join("noise_stereo16.wav").is_file());

    let help = Command::new(bin).arg("--help").status().unwrap();
    assert!(help.success());

    let _ = std::fs::remove_dir_all(&root);
}

static NEXT: AtomicU64 = AtomicU64::new(1);
