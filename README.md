# flac2wav

Convert FLAC to WAV on all of your cores. The audio stays at the same sample rate, channel count, and bit depth. This writes PCM only: titles, pictures, and other FLAC tags are not copied.

## Build on a Mac

Install Rust from [rustup](https://rustup.rs) if you do not already have it, then:

```sh
cargo test
cargo build --release
```

The binary is `target/release/flac2wav`. Intel and Apple Silicon both work. You do not need FFmpeg or a C compiler. `cargo test` uses the short FLAC files in `tests/fixtures` and does not need network access after the crates are downloaded.

## Usage

```sh
# One file, eight threads, exact output path.
flac2wav -j 8 -o track.wav track.flac

# Many files at once. -o is the directory. -j is how many files run together.
flac2wav -j 8 -o ~/Music/wav ~/Music/flac/*.flac

# Write song.wav next to song.flac. -j 0 means one worker per CPU.
flac2wav song.flac
```

| Flag | Meaning |
| --- | --- |
| `-j`, `--jobs` | Threads, 1 through 256. `0` (the default) uses one per logical CPU. |
| `-o`, `--output` | A `.wav` path for one input, or a directory for several. |
| `-f`, `--force` | Replace a WAV that is already there. |
| `-q`, `--quiet` | Only print errors. |
| `--ignore-md5` | Skip the checksum stored in the FLAC header. |

The default is to refuse to overwrite. A failed file does not delete the others in a batch; the command exits non-zero if any file failed.

## Library

```rust
use flac2wav::{decode_flac, flac_to_wav, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let flac = std::fs::read("song.flac")?;
    let wav = flac_to_wav(&flac, 0)?; // 0 = one thread per CPU
    std::fs::write("song.wav", wav)?;

    let pcm = decode_flac(
        &flac,
        &Options {
            threads: 4,
            check_md5: true,
        },
    )?;
    println!("{} channels, {} ranges", pcm.channels, pcm.ranges);
    Ok(())
}
```

`Pcm::interleaved` is one `i32` per sample, left then right, already sign-extended. `Pcm::ranges` is how many frame ranges were actually decoded.

## How parallelism works

One file is split by byte range. Each worker scans forward to the next real FLAC frame. A candidate has to match the frame sync, pass the header CRC-8, and decode. A look-alike sync code inside compressed audio is rejected, so the next worker cannot drift.

Several files share one thread pool. Each file is then decoded on a single thread so eight files do not also split into eight frame ranges each. One file uses the whole pool for its frame ranges.

Most 16-bit and 24-bit files store the bit depth in every frame. Some store it only in the stream header. Those frames are rewritten in a private copy (depth filled in, CRC-8 and CRC-16 recomputed) because the decoder crate will not read that form. The rewrite is checked by a test that flips a normal file into that form and requires the same PCM.

## Limits

- 8, 12, 16, 20, 24, and 32-bit are the depths a frame header can name. A depth outside that list, stored only in the stream header, is an error.
- 32-bit files work only when the decoder accepts them. If it does not, you get an error instead of a bad WAV.
- WAV output larger than 4 GiB is rejected. RIFF cannot hold it.
- An optional MD5 in the FLAC header is checked against the decoded samples. `--ignore-md5` turns that off.
- A trailing ID3v1 tag or a leading ID3v2 tag is skipped. Other junk around the stream is an error.

## Layout

```text
src/main.rs     command line
src/lib.rs      public API
src/decode.rs   metadata, frame split, parallel decode
src/wav.rs      44-byte PCM WAV writer
tests/          fixtures and round-trip tests
```

The PCM fixtures were exported from the FLAC files by a separate decoder. `cargo test` checks that one thread and many threads match those bytes.
