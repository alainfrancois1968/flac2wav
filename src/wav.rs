//! Little-endian PCM WAV writer. The header is the 44-byte canonical form.

use crate::decode::{Error, Pcm};

pub fn encode_wav(pcm: &Pcm) -> Result<Vec<u8>, Error> {
    if pcm.sample_rate == 0 || pcm.channels == 0 {
        return Err(Error::Invalid(
            "PCM is missing a sample rate or channel count".into(),
        ));
    }
    if pcm.channels > u16::MAX as u32 {
        return Err(Error::Unsupported(
            "too many channels for a WAV header".into(),
        ));
    }
    if !pcm.interleaved.len().is_multiple_of(pcm.channels as usize) {
        return Err(Error::Invalid(
            "PCM sample count is not divisible by the channel count".into(),
        ));
    }

    let width = byte_width(pcm.bits_per_sample)?;
    let channels = pcm.channels as u16;
    let block_align = channels as usize * width;
    let data_len = pcm
        .interleaved
        .len()
        .checked_mul(width)
        .ok_or_else(|| Error::Unsupported("WAV data is larger than memory can hold".into()))?;
    if data_len > u32::MAX as usize || data_len.saturating_add(36) > u32::MAX as usize {
        return Err(Error::Unsupported(
            "WAV would exceed the 4 GiB RIFF size limit".into(),
        ));
    }
    let byte_rate = (pcm.sample_rate as usize)
        .checked_mul(block_align)
        .ok_or_else(|| Error::Invalid("WAV byte rate overflowed".into()))?;
    if byte_rate > u32::MAX as usize {
        return Err(Error::Invalid(
            "WAV byte rate does not fit in the header".into(),
        ));
    }

    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&pcm.sample_rate.to_le_bytes());
    out.extend_from_slice(&(byte_rate as u32).to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&(pcm.bits_per_sample as u16).to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());

    let unsigned_8 = width == 1;
    for &sample in &pcm.interleaved {
        push_sample(sample, width, unsigned_8, &mut out);
    }
    debug_assert_eq!(out.len(), 44 + data_len);
    Ok(out)
}

pub(crate) fn byte_width(bits_per_sample: u32) -> Result<usize, Error> {
    match bits_per_sample {
        1..=8 => Ok(1),
        9..=16 => Ok(2),
        17..=24 => Ok(3),
        25..=32 => Ok(4),
        _ => Err(Error::Unsupported(format!(
            "{bits_per_sample}-bit audio is not supported"
        ))),
    }
}

/// Append one sample. `unsigned_8` is the WAV rule for 8-bit PCM only.
/// MD5 checks of the FLAC audio must pass `false` so 8-bit stays signed.
pub(crate) fn push_sample(sample: i32, width: usize, unsigned_8: bool, out: &mut Vec<u8>) {
    match width {
        1 => {
            let byte = sample as u8;
            out.push(if unsigned_8 {
                byte.wrapping_add(128)
            } else {
                byte
            });
        }
        2 => out.extend_from_slice(&(sample as i16).to_le_bytes()),
        3 => {
            out.push(sample as u8);
            out.push((sample >> 8) as u8);
            out.push((sample >> 16) as u8);
        }
        _ => out.extend_from_slice(&sample.to_le_bytes()),
    }
}
