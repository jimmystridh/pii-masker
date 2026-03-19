use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use pii_masker::PiiMasker;
use serde_json::to_string_pretty;

#[derive(Debug, Parser)]
#[command(name = "pii-mask")]
#[command(about = "Mask PII from stdin using the HydroXai DeBERTa model.")]
struct Args {
    #[arg(long)]
    json: bool,

    #[arg(long, value_name = "PATH")]
    model_weights: Option<PathBuf>,

    #[arg(value_name = "TEXT")]
    text: Vec<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("pii-mask: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input = read_input(&args.text)?;

    let masker = match args.model_weights {
        Some(path) => PiiMasker::builder().weights_path(path).build()?,
        None => PiiMasker::new()?,
    };

    let result = masker.mask(&input)?;
    if args.json {
        println!("{}", to_string_pretty(&result)?);
    } else {
        println!("{}", result.masked_text);
    }

    Ok(())
}

fn read_input(parts: &[String]) -> Result<String, Box<dyn std::error::Error>> {
    if !parts.is_empty() {
        return Ok(parts.join(" "));
    }

    let stdin = io::stdin();
    if stdin.is_terminal() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "no input provided; pipe text on stdin or pass it as an argument",
        )
        .into());
    }

    let mut input = String::new();
    stdin.lock().read_to_string(&mut input)?;
    if input.trim().is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "stdin was empty").into());
    }

    Ok(input.trim_end_matches('\n').to_string())
}
