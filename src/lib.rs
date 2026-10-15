//! Parallel FLAC to WAV conversion.
//!
//! A file is split into FLAC frame ranges and those ranges are decoded on a
//! Rayon pool. Sample rate, channel count, and bit depth are preserved.
//! Vorbis comments and pictures are not copied into the WAV.

#![forbid(unsafe_code)]

mod decode;
mod wav;

pub use decode::{decode_flac, probe, worker_count, Error, FlacInfo, Options, Pcm};
pub use wav::encode_wav;

/// Decode `bytes` and wrap the PCM in a WAV file.
///
/// `threads == 0` uses one worker per logical CPU, capped at 256.
/// Any other value is clamped to `1..=256`.
pub fn flac_to_wav(bytes: &[u8], threads: usize) -> Result<Vec<u8>, Error> {
    convert(
        bytes,
        &Options {
            threads,
            check_md5: true,
        },
    )
}

/// Decode `bytes` with [`Options`] and wrap the PCM in a WAV file.
pub fn convert(bytes: &[u8], options: &Options) -> Result<Vec<u8>, Error> {
    encode_wav(&decode_flac(bytes, options)?)
}
