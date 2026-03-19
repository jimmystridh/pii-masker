# `pii-masker`

[crates.io](https://crates.io/crates/pii-masker) | [docs.rs](https://docs.rs/pii-masker) | [GitHub](https://github.com/jimmystridh/pii-masker)

Rust port of the HydroXai `pii-masker` library, built around the same DeBERTa token-classification model and packaged as both:

- a reusable Rust library
- a stdin-friendly CLI

This crate is model-only. It does not add a regex fallback layer on top of the model output.

Install from crates.io:

```bash
cargo add pii-masker
```

## What Is Bundled

The crate embeds the small model artifacts it needs at compile time:

- `config.json`
- `tokenizer.json`

The large weight file is not embedded in the binary by default:

- `model.safetensors`

That keeps the compiled binary reasonably small and makes container deployment much cleaner.

## Model Resolution

When you call `PiiMasker::new()`, the crate resolves weights in this order:

1. `PII_MASKER_MODEL_WEIGHTS`
2. `model/model.safetensors` inside the repo or deployed app directory
3. Hugging Face download from `hydroxai/pii_model_weight`

If you want deterministic deployment behavior, set the weights path explicitly with the builder or set `PII_MASKER_MODEL_WEIGHTS`.

## Library Usage

Add the crate as a dependency:

```toml
[dependencies]
pii-masker = "0.1.0"
```

Basic usage:

```rust
use pii_masker::PiiMasker;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let masker = PiiMasker::builder()
        .weights_path("/app/model/model.safetensors")
        .build()?;

    let result = masker.mask("John Doe lives at 1234 Elm St.")?;

    println!("{}", result.masked_text);
    println!("{:#?}", result.pii);
    Ok(())
}
```

Example output:

```text
John Doe lives at [ADDRESS].
```

The main library types are:

- `PiiMasker`
- `PiiMaskerBuilder`
- `MaskResult`
- `PiiEntity`

## CLI Usage

Run the CLI in release mode:

```bash
cargo run --release --bin pii-mask -- --json
```

Pipe text on stdin:

```bash
printf 'John Doe lives at 1234 Elm St.' | cargo run --release --bin pii-mask -- --json
```

Or pass text as an argument:

```bash
cargo run --release --bin pii-mask -- "John Doe lives at 1234 Elm St."
```

Use a specific weights file:

```bash
printf 'John Doe lives at 1234 Elm St.' \
  | cargo run --release --bin pii-mask -- \
    --model-weights /app/model/model.safetensors \
    --json
```

Or place the weights at:

```text
model/model.safetensors
```

and let `PiiMasker::new()` or the CLI pick them up automatically.

Plain-text output:

```text
John Doe lives at [ADDRESS].
```

JSON output:

```json
{
  "masked_text": "John Doe lives at [ADDRESS].",
  "pii": {
    "ADDRESS": ["1234 Elm St"]
  }
}
```

## Performance Notes

Use `--release` for anything real.

`cargo run --bin pii-mask` uses the debug profile, and model startup is much slower there. The model is also loaded fresh for each process invocation, so short-lived CLI runs pay the startup cost every time.

For services or web apps:

- construct `PiiMasker` once at process startup
- reuse it for all requests
- avoid rebuilding the model for each call

## Container Deployment

The recommended deployment shape is:

- compile your app or CLI in a builder stage
- copy the binary into a slim runtime image
- copy `model.safetensors` into the image filesystem
- set `PII_MASKER_MODEL_WEIGHTS`

Example Dockerfile:

```dockerfile
FROM rust:1.86-bookworm AS builder
WORKDIR /src

COPY pii-masker ./pii-masker
WORKDIR /src/pii-masker
RUN cargo build --release --bin pii-mask

FROM debian:bookworm-slim
WORKDIR /app

COPY --from=builder /src/pii-masker/target/release/pii-mask /usr/local/bin/pii-mask
COPY model.safetensors /app/model/model.safetensors

ENV PII_MASKER_MODEL_WEIGHTS=/app/model/model.safetensors

ENTRYPOINT ["pii-mask"]
```

For local development, a simple layout is:

```text
pii-masker/
  model/
    model.safetensors
```

For a server that uses the library instead of the CLI, the pattern is the same:

- copy your server binary into the image
- copy `model.safetensors`
- set `PII_MASKER_MODEL_WEIGHTS`
- initialize `PiiMasker` once on startup

## Single-Binary Deployment

If you absolutely need a single self-contained binary, you can embed `model.safetensors` with `include_bytes!` and switch to a buffered safetensors loader.

That is possible, but it is usually a bad tradeoff:

- the binary grows by roughly 700 MB
- cold start gets worse
- rebuilds get slower
- container layering gets less efficient

Bundling the model as a file in the image is the better default.

## Testing

Run the library and CLI tests:

```bash
cargo test
```

The test suite includes:

- pure unit tests for label normalization and span trimming
- optional model-backed smoke tests when `PII_MASKER_TEST_MODEL_WEIGHTS` or `model/model.safetensors` is available

## Publishing

This repo follows the same basic crates.io flow used in several of the other Rust projects in `~/code/rust`:

- `ci.yml` runs format, clippy, tests, and release build on pushes to `main` and pull requests
- on pushes to `main`, the workflow checks whether the current version already exists on crates.io
- if the version does not exist and `CARGO_REGISTRY_TOKEN` is configured in GitHub Actions secrets, the crate is published automatically
- `release.yml` creates a GitHub Release when you push a `vX.Y.Z` tag

To enable crates.io publishing in GitHub Actions, add this repository secret:

- `CARGO_REGISTRY_TOKEN`

Example release flow:

1. Bump the version in `Cargo.toml`.
2. Merge to `main`.
3. CI publishes to crates.io if that version is not already published.
4. Push a tag like `v0.1.0` to create the GitHub release page.

## Current Behavior

This crate mirrors the model behavior closely. If the underlying model misses or truncates an entity, the Rust port will do the same.

For example, phone-number masking is currently weaker than email or address masking because the model itself is weaker there. This crate does not paper over that with regexes.
