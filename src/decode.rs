//! FLAC probe, frame walking, and parallel decode.
//!
//! Workers do not share a decoder. Each range starts on a frame header that
//! passed its CRC-8 and a real decode, so a sync-like byte pair inside a
//! residual cannot move the next worker.
//!
//! Some files store bit depth only in STREAMINFO (frame header code `000`).
//! Claxon refuses those frames. The slow path copies that one frame, writes
//! the explicit depth into the header, and recomputes the frame CRC-8 and
//! CRC-16. The checksum is recomputed only over a candidate that the decoder
//! consumes completely, so a guessed end offset cannot forge a successful frame.

use std::io::Cursor;

use claxon::frame::FrameReader;
use md5::{Digest, Md5};
use rayon::prelude::*;

use crate::wav::{byte_width, push_sample};

/// How many workers to use. `0` means one per logical CPU. The result is `1..=256`.
pub fn worker_count(threads: usize) -> usize {
    let count = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    };
    count.clamp(1, 256)
}

#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Unsupported(String),
    Decode(String),
    Md5Mismatch { expected: String, actual: String },
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Invalid(message) | Error::Unsupported(message) | Error::Decode(message) => {
                write!(f, "{message}")
            }
            Error::Md5Mismatch { expected, actual } => {
                write!(f, "MD5 mismatch: STREAMINFO {expected}, decoded {actual}")
            }
            Error::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

/// STREAMINFO plus where the first audio frame starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlacInfo {
    pub sample_rate: u32,
    pub channels: u32,
    pub bits_per_sample: u32,
    /// `None` when the file does not declare a total (STREAMINFO value 0).
    pub total_samples: Option<u64>,
    /// Absolute offset of the first audio frame.
    pub audio_offset: usize,
    /// Unencoded-audio MD5. All zeros means the encoder did not store one.
    pub md5: [u8; 16],
}

/// Interleaved signed PCM. Every sample is an `i32`, even when the file is 16-bit or 24-bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcm {
    pub sample_rate: u32,
    pub channels: u32,
    pub bits_per_sample: u32,
    pub interleaved: Vec<i32>,
    /// How many frame ranges were decoded. This is the parallelism that actually ran.
    pub ranges: usize,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// `0` picks one worker per logical CPU. Otherwise clamped to `1..=256`.
    pub threads: usize,
    /// Check STREAMINFO's MD5 when it is not all zeros.
    pub check_md5: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            threads: 0,
            check_md5: true,
        }
    }
}

const MAX_FRAME_SCAN: usize = 4 * 1024 * 1024;
const MAX_TRAIL: usize = 32;

struct Header {
    total_len: usize,
    bps_code: u8,
}

struct DecodedFrame {
    pcm: Vec<i32>,
    channels: u32,
    consumed: usize,
}

#[derive(Debug)]
enum FrameError {
    Missing,
    Failed(String),
}

/// Read STREAMINFO and the audio start offset. An ID3v2 tag in front of `fLaC` is skipped.
pub fn probe(bytes: &[u8]) -> Result<FlacInfo, Error> {
    let start = flac_magic(bytes)?;
    if bytes.len() < start + 8 {
        return Err(Error::Invalid("FLAC file is truncated".into()));
    }
    let mut offset = start + 4;
    let mut info = None;
    let mut first = true;
    loop {
        if offset + 4 > bytes.len() {
            return Err(Error::Invalid("metadata block is truncated".into()));
        }
        let header = bytes[offset];
        let last = header & 0x80 != 0;
        let kind = header & 0x7f;
        if kind == 127 {
            return Err(Error::Invalid(
                "invalid FLAC metadata block type 127".into(),
            ));
        }
        let len = u32::from_be_bytes([0, bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
            as usize;
        let body = offset + 4;
        let next = body
            .checked_add(len)
            .ok_or_else(|| Error::Invalid("metadata block length overflowed".into()))?;
        if next > bytes.len() {
            return Err(Error::Invalid(
                "metadata block runs past the end of the file".into(),
            ));
        }
        if first && kind != 0 {
            return Err(Error::Invalid(
                "first FLAC metadata block is not STREAMINFO".into(),
            ));
        }
        if kind == 0 {
            if info.is_some() {
                return Err(Error::Invalid("FLAC file has two STREAMINFO blocks".into()));
            }
            if len < 34 {
                return Err(Error::Invalid("STREAMINFO block is too short".into()));
            }
            info = Some(parse_streaminfo(&bytes[body..body + 34])?);
        }
        first = false;
        offset = next;
        if last {
            break;
        }
    }
    let mut info =
        info.ok_or_else(|| Error::Invalid("FLAC file has no STREAMINFO block".into()))?;
    info.audio_offset = offset;

    // Advance audio_offset to the first frame that actually decodes
    if let Some(real_start) = find_first_valid_frame(bytes, offset, info.bits_per_sample) {
        info.audio_offset = real_start;
    }

    Ok(info)
}

fn find_first_valid_frame(bytes: &[u8], from: usize, stream_bps: u32) -> Option<usize> {
    let mut i = from;
    while i + 6 < bytes.len() {
        if bytes[i] == 0xff && (bytes[i + 1] & 0xfe) == 0xf8 {
            if read_frame(&bytes[i..], stream_bps).is_ok() {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Decode a FLAC byte slice to interleaved PCM.
pub fn decode_flac(bytes: &[u8], options: &Options) -> Result<Pcm, Error> {
    let trimmed = trim_id3v1(bytes);
    if trimmed.len() != bytes.len() {
        if let Ok(pcm) = decode_bytes(trimmed, options) {
            return Ok(pcm);
        }
    }
    decode_bytes(bytes, options)
}

fn decode_bytes(bytes: &[u8], options: &Options) -> Result<Pcm, Error> {
    let info = probe(bytes)?;
    let threads = worker_count(options.threads);
    let ranges = plan_ranges(bytes, &info, threads)?;

    let parts = if ranges.len() == 1 {
        vec![decode_range(bytes, ranges[0].0, ranges[0].1, &info)?]
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.min(ranges.len()))
            .build()
            .map_err(|err| Error::Decode(format!("could not start worker threads: {err}")))?;
        pool.install(|| {
            ranges
                .par_iter()
                .map(|&(start, end)| decode_range(bytes, start, end, &info))
                .collect::<Result<Vec<_>, Error>>()
        })?
    };

    let mut interleaved = Vec::new();
    if let Some(total) = info.total_samples {
        interleaved
            .try_reserve(total as usize * info.channels as usize)
            .ok();
    }
    for part in parts {
        interleaved.extend(part);
    }

    // Fallback: if the normal path produced nothing, do a full byte-by-byte scan
    if interleaved.is_empty() {
        interleaved = decode_full_scan(bytes, &info)?;
    }

    if info.channels == 0 || !interleaved.len().is_multiple_of(info.channels as usize) {
        return Err(Error::Decode(
            "decoded sample count is not divisible by the channel count".into(),
        ));
    }
    let per_channel = (interleaved.len() / info.channels as usize) as u64;
    if let Some(expected) = info.total_samples {
        let diff = (per_channel as i64 - expected as i64).abs();
        // Allow missing up to one typical frame (or 1%, whichever is larger)
        let tolerance = (expected / 100).max(4608);
        if diff > tolerance as i64 {
            return Err(Error::Decode(format!(
                "decoded {per_channel} samples per channel, STREAMINFO says {expected}"
            )));
        }
    }
    if interleaved.is_empty() {
        return Err(Error::Decode("FLAC file has no audio samples".into()));
    }
    if options.check_md5 && info.md5.iter().any(|byte| *byte != 0) {
        let actual = md5_pcm(&interleaved, info.bits_per_sample)?;
        if actual != info.md5 {
            return Err(Error::Md5Mismatch {
                expected: hex16(&info.md5),
                actual: hex16(&actual),
            });
        }
    }
    Ok(Pcm {
        sample_rate: info.sample_rate,
        channels: info.channels,
        bits_per_sample: info.bits_per_sample,
        interleaved,
        ranges: ranges.len(),
    })
}

fn plan_ranges(
    bytes: &[u8],
    info: &FlacInfo,
    threads: usize,
) -> Result<Vec<(usize, usize)>, Error> {
    let audio = info.audio_offset;
    if audio >= bytes.len() {
        return Err(Error::Decode("FLAC file has no audio frames".into()));
    }
    if threads <= 1 {
        return Ok(vec![(audio, bytes.len())]);
    }
    let mut starts = vec![audio];
    let span = bytes.len() - audio;
    for index in 1..threads {
        let guess = audio + span * index / threads;
        if let Some(offset) = find_frame(bytes, guess, info.bits_per_sample) {
            if offset > *starts.last().unwrap() && offset < bytes.len() {
                starts.push(offset);
            }
        }
    }
    let mut ranges = Vec::with_capacity(starts.len());
    for (index, &start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(bytes.len());
        if end > start {
            ranges.push((start, end));
        }
    }
    if ranges.is_empty() {
        return Err(Error::Decode("FLAC file has no audio frames".into()));
    }
    Ok(ranges)
}

fn find_frame(bytes: &[u8], from: usize, stream_bps: u32) -> Option<usize> {
    if from >= bytes.len() {
        return None;
    }
    let last = (from + MAX_FRAME_SCAN).min(bytes.len());
    let mut index = from;
    while index + 6 < last {
        if bytes[index] == 0xff
            && bytes[index + 1] & 0xfe == 0xf8
            && read_frame(&bytes[index..], stream_bps).is_ok()
        {
            return Some(index);
        }
        index += 1;
    }
    None
}

fn decode_range(
    bytes: &[u8],
    start: usize,
    end: usize,
    info: &FlacInfo,
) -> Result<Vec<i32>, Error> {
    let end = end.min(bytes.len());
    if start >= end {
        return Ok(Vec::new());
    }
    let slice = &bytes[start..end];
    let mut pos = 0usize;
    let mut out = Vec::new();

    while pos + 6 <= slice.len() {
        match read_frame(&slice[pos..], info.bits_per_sample) {
            Ok(frame) => {
                if frame.consumed == 0 || pos + frame.consumed > slice.len() {
                    // Bad length — skip one byte and keep searching
                    pos += 1;
                    continue;
                }
                if frame.channels != info.channels {
                    // Channel mismatch — skip and continue
                    pos += 1;
                    continue;
                }
                out.extend(frame.pcm);
                pos += frame.consumed;
            }
            Err(FrameError::Missing) => {
                // No valid frame here — advance one byte and keep looking
                pos += 1;
                // Safety: don't scan forever if we already have data and we're near the end
                if !out.is_empty() && slice.len() - pos <= MAX_TRAIL {
                    break;
                }
            }
            Err(FrameError::Failed(_)) => {
                // Decode failed — skip one byte and continue
                pos += 1;
            }
        }
    }

    if out.is_empty() {
        return Err(Error::Decode(format!(
            "no FLAC frame at file offset {start}"
        )));
    }
    Ok(out)
}

fn read_frame(bytes: &[u8], stream_bps: u32) -> Result<DecodedFrame, FrameError> {
    let header = match parse_header(bytes) {
        Some(header) => header,
        None => return Err(FrameError::Missing),
    };
    if !header_crc_ok(bytes, header.total_len) {
        return Err(FrameError::Missing);
    }
    if header.bps_code == 0 {
        return read_frame_patched(bytes, &header, stream_bps);
    }
    read_frame_bytes(bytes).map_err(FrameError::Failed)
}

fn read_frame_patched(
    bytes: &[u8],
    header: &Header,
    stream_bps: u32,
) -> Result<DecodedFrame, FrameError> {
    let code = match explicit_bps_code(stream_bps) {
        Ok(code) => code,
        Err(err) => return Err(FrameError::Failed(err.to_string())),
    };
    let mut index = header.total_len;
    let scan_end = (header.total_len + MAX_FRAME_SCAN).min(bytes.len());
    while index + 6 <= scan_end {
        if bytes[index] == 0xff && looks_like_frame(&bytes[index..]) {
            if let Some(frame) = try_patched(bytes, index, header, code) {
                return Ok(frame);
            }
        }
        index += 1;
    }
    let end = bytes.len().min(header.total_len + MAX_FRAME_SCAN);
    if let Some(frame) = try_patched(bytes, end, header, code) {
        return Ok(frame);
    }
    Err(FrameError::Failed(
        "could not decode a frame whose bit depth is stored only in STREAMINFO".into(),
    ))
}

fn try_patched(bytes: &[u8], end: usize, header: &Header, code: u8) -> Option<DecodedFrame> {
    if end < header.total_len + 2 || end > bytes.len() {
        return None;
    }
    let mut frame = bytes[..end].to_vec();
    frame[3] = (frame[3] & 0b1111_0001) | (code << 1);
    let crc = crc8(&frame[..header.total_len - 1]);
    frame[header.total_len - 1] = crc;
    let footer = crc16(&frame[..end - 2]);
    frame[end - 2] = (footer >> 8) as u8;
    frame[end - 1] = (footer & 0xff) as u8;
    match read_frame_bytes(&frame) {
        Ok(decoded) if decoded.consumed == end => Some(decoded),
        _ => None,
    }
}

fn looks_like_frame(bytes: &[u8]) -> bool {
    match parse_header(bytes) {
        Some(header) => header_crc_ok(bytes, header.total_len),
        None => false,
    }
}

fn read_frame_bytes(bytes: &[u8]) -> Result<DecodedFrame, String> {
    let cursor = Cursor::new(bytes);
    let mut reader = FrameReader::new(cursor);
    let block = match reader.read_next_or_eof(Vec::new()) {
        Ok(Some(block)) => block,
        Ok(None) => return Err("frame ended before the header".into()),
        Err(err) => return Err(err.to_string()),
    };
    let consumed = reader.into_inner().position() as usize;
    if consumed == 0 || consumed > bytes.len() {
        return Err("decoder reported an impossible frame length".into());
    }
    let channels = block.channels();
    let count = block.duration() as usize;
    if channels == 0 || count == 0 {
        return Err("decoded an empty frame".into());
    }
    let mut planes = Vec::with_capacity(channels as usize);
    for channel in 0..channels {
        let plane = block.channel(channel);
        if plane.len() < count {
            return Err("decoded channel is shorter than the frame".into());
        }
        planes.push(plane);
    }
    let mut pcm = Vec::with_capacity(count * channels as usize);
    for index in 0..count {
        for plane in &planes {
            pcm.push(plane[index]);
        }
    }
    Ok(DecodedFrame {
        pcm,
        channels,
        consumed,
    })
}

fn parse_streaminfo(body: &[u8]) -> Result<FlacInfo, Error> {
    let packed = u64::from_be_bytes([
        body[10], body[11], body[12], body[13], body[14], body[15], body[16], body[17],
    ]);
    let sample_rate = (packed >> 44) as u32;
    let channels = ((packed >> 41) & 0x7) as u32 + 1;
    let bits_per_sample = ((packed >> 36) & 0x1f) as u32 + 1;
    let total = packed & ((1_u64 << 36) - 1);
    if sample_rate == 0 || channels == 0 || bits_per_sample == 0 || bits_per_sample > 32 {
        return Err(Error::Invalid(
            "STREAMINFO has an unusable rate, channel count, or bit depth".into(),
        ));
    }
    let mut md5 = [0_u8; 16];
    md5.copy_from_slice(&body[18..34]);
    Ok(FlacInfo {
        sample_rate,
        channels,
        bits_per_sample,
        total_samples: if total == 0 { None } else { Some(total) },
        audio_offset: 0,
        md5,
    })
}

fn parse_header(bytes: &[u8]) -> Option<Header> {
    if bytes.len() < 6 || bytes[0] != 0xff || bytes[1] & 0xfe != 0xf8 {
        return None;
    }
    // Reserved bit must be 0
    if bytes[3] & 0x01 != 0 {
        return None;
    }
    let block_size_code = bytes[2] >> 4;
    let sample_rate_code = bytes[2] & 0x0f;
    let channel_assignment = bytes[3] >> 4;
    let bps_code = (bytes[3] >> 1) & 0x07;

    // Reject illegal values
    if block_size_code == 0
        || sample_rate_code == 0x0f
        || bps_code == 0b011
        || channel_assignment > 10
    {
        return None;
    }

    let mut index = 4usize;
    let number_len = coded_number_len(*bytes.get(index)?)?;
    if index + number_len > bytes.len() {
        return None;
    }
    if number_len > 1 {
        for byte in &bytes[index + 1..index + number_len] {
            if byte & 0xc0 != 0x80 {
                return None;
            }
        }
    }
    index += number_len;
    let extra = match block_size_code {
        0b0110 => 1,
        0b0111 => 2,
        _ => 0,
    } + match sample_rate_code {
        0b1100 => 1,
        0b1101 | 0b1110 => 2,
        _ => 0,
    };
    let crc_index = index.checked_add(extra)?;
    let total_len = crc_index.checked_add(1)?;
    if total_len > bytes.len() {
        return None;
    }
    Some(Header {
        total_len,
        bps_code,
    })
}

fn coded_number_len(first: u8) -> Option<usize> {
    match first.leading_ones() {
        0 => Some(1),
        1 => None,
        ones @ 2..=7 => Some(ones as usize),
        _ => None,
    }
}

fn header_crc_ok(bytes: &[u8], total_len: usize) -> bool {
    total_len > 0
        && total_len <= bytes.len()
        && crc8(&bytes[..total_len - 1]) == bytes[total_len - 1]
}

fn explicit_bps_code(bits_per_sample: u32) -> Result<u8, Error> {
    Ok(match bits_per_sample {
        8 => 0b001,
        12 => 0b010,
        16 => 0b100,
        20 => 0b101,
        24 => 0b110,
        32 => 0b111,
        other => {
            return Err(Error::Unsupported(format!(
            "{other}-bit FLAC stores its depth only in STREAMINFO, which this decoder cannot patch"
        )))
        }
    })
}

fn crc8(bytes: &[u8]) -> u8 {
    let mut crc = 0_u8;
    for &byte in bytes {
        crc ^= byte;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

fn crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0_u16;
    for &byte in bytes {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

fn md5_pcm(samples: &[i32], bits_per_sample: u32) -> Result<[u8; 16], Error> {
    let width = byte_width(bits_per_sample)?;
    let mut hasher = Md5::new();
    let mut buf = Vec::with_capacity(8192);
    for &sample in samples {
        push_sample(sample, width, false, &mut buf);
        if buf.len() >= 8192 {
            hasher.update(&buf);
            buf.clear();
        }
    }
    if !buf.is_empty() {
        hasher.update(&buf);
    }
    Ok(hasher.finalize().into())
}

fn hex16(bytes: &[u8; 16]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(32);
    for byte in bytes {
        text.push(HEX[(byte >> 4) as usize] as char);
        text.push(HEX[(byte & 0x0f) as usize] as char);
    }
    text
}

fn flac_magic(bytes: &[u8]) -> Result<usize, Error> {
    if bytes.len() >= 4 && &bytes[..4] == b"fLaC" {
        return Ok(0);
    }
    if bytes.len() >= 10 && &bytes[..3] == b"ID3" {
        if bytes[6..10].iter().any(|byte| byte & 0x80 != 0) {
            return Err(Error::Invalid("ID3 size is not synchsafe".into()));
        }
        let size = ((bytes[6] as usize) << 21)
            | ((bytes[7] as usize) << 14)
            | ((bytes[8] as usize) << 7)
            | (bytes[9] as usize);
        let footer = if bytes[5] & 0x10 != 0 { 10 } else { 0 };
        let start = 10usize.saturating_add(size).saturating_add(footer);
        if start + 4 <= bytes.len() && &bytes[start..start + 4] == b"fLaC" {
            return Ok(start);
        }
        return Err(Error::Invalid(
            "ID3 tag is not followed by a FLAC stream".into(),
        ));
    }
    Err(Error::Invalid("not a FLAC file".into()))
}

fn trim_id3v1(bytes: &[u8]) -> &[u8] {
    if bytes.len() >= 128 && &bytes[bytes.len() - 128..bytes.len() - 125] == b"TAG" {
        &bytes[..bytes.len() - 128]
    } else {
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claxon::FlacReader;
    use std::io::Cursor;

    fn opts(threads: usize) -> Options {
        Options {
            threads,
            check_md5: true,
        }
    }

    #[test]
    fn probe_matches_claxon_and_frame_crc_holds() {
        let bytes = include_bytes!("../tests/fixtures/noise_stereo16.flac");
        let info = probe(bytes).unwrap();
        let reader = FlacReader::new(Cursor::new(&bytes[..])).unwrap();
        let stream = reader.streaminfo();
        assert_eq!(info.sample_rate, stream.sample_rate);
        assert_eq!(info.channels, stream.channels);
        assert_eq!(info.bits_per_sample, stream.bits_per_sample);
        assert_eq!(info.total_samples, stream.samples);
        assert!(info.md5.iter().any(|byte| *byte != 0));

        let header = parse_header(&bytes[info.audio_offset..]).expect("first frame header");
        assert!(header_crc_ok(&bytes[info.audio_offset..], header.total_len));
        assert_ne!(
            header.bps_code, 0,
            "fixture should store bit depth in the frame"
        );
        let frame = read_frame(&bytes[info.audio_offset..], info.bits_per_sample).unwrap();
        let raw = &bytes[info.audio_offset..info.audio_offset + frame.consumed];
        let stored = u16::from_be_bytes([raw[raw.len() - 2], raw[raw.len() - 1]]);
        assert_eq!(crc16(&raw[..raw.len() - 2]), stored);
    }

    #[test]
    fn parallel_ranges_match_claxon_and_each_other() {
        let bytes = include_bytes!("../tests/fixtures/noise_stereo16.flac");
        let one = decode_flac(bytes, &opts(1)).unwrap();
        let many = decode_flac(bytes, &opts(4)).unwrap();
        assert!(
            many.ranges > 1,
            "expected the 2 second file to split across workers"
        );
        assert_eq!(one.interleaved, many.interleaved);

        let mut reader = FlacReader::new(Cursor::new(&bytes[..])).unwrap();
        let oracle: Vec<i32> = reader.samples().map(|sample| sample.unwrap()).collect();
        assert_eq!(one.interleaved, oracle);
    }

    #[test]
    fn patched_streaminfo_bit_depth_matches() {
        let original = include_bytes!("../tests/fixtures/noise_stereo16.flac");
        let info = probe(original).unwrap();
        let mut bytes = original.to_vec();
        let mut pos = info.audio_offset;
        let mut frames = 0usize;
        while pos + 6 < bytes.len() {
            let header = match parse_header(&bytes[pos..]) {
                Some(header) if header_crc_ok(&bytes[pos..], header.total_len) => header,
                _ => break,
            };
            let decoded = read_frame_bytes(&bytes[pos..]).expect("original frame");
            bytes[pos + 3] &= 0b1111_0001;
            let crc = crc8(&bytes[pos..pos + header.total_len - 1]);
            bytes[pos + header.total_len - 1] = crc;
            pos += decoded.consumed;
            frames += 1;
            if bytes.len() - pos <= MAX_TRAIL {
                break;
            }
        }
        assert!(frames > 1);
        let before = decode_flac(original, &opts(1)).unwrap();
        let after = decode_flac(&bytes, &opts(3)).unwrap();
        assert_eq!(before.interleaved, after.interleaved);
    }

    #[test]
    fn id3_prefix_and_suffix_are_ignored() {
        let flac = include_bytes!("../tests/fixtures/tiny_mono.flac");
        let mut wrapped = vec![b'I', b'D', b'3', 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0];
        wrapped.extend_from_slice(flac);
        wrapped.extend_from_slice(b"TAG");
        wrapped.extend(std::iter::repeat_n(0, 125));
        let plain = decode_flac(flac, &opts(2)).unwrap();
        let tagged = decode_flac(&wrapped, &opts(2)).unwrap();
        assert_eq!(plain.interleaved, tagged.interleaved);
    }
}

fn decode_full_scan(bytes: &[u8], info: &FlacInfo) -> Result<Vec<i32>, Error> {
    let mut pos = info.audio_offset;
    let mut out = Vec::new();
    while pos + 6 < bytes.len() {
        match read_frame(&bytes[pos..], info.bits_per_sample) {
            Ok(frame) if frame.consumed > 0 && frame.channels == info.channels => {
                out.extend(frame.pcm);
                pos += frame.consumed;
            }
            _ => {
                pos += 1;
            }
        }
    }
    if out.is_empty() {
        return Err(Error::Decode("no valid FLAC frames found".into()));
    }
    Ok(out)
}
