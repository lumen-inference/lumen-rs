//! `lumen` CLI binary (`docs/RU/api.md` Section 2):
//! `inspect`, `run`, `chat`, `serve`, `probe`, `bench`, `pack`, `quantize`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use lumen_config::RuntimeConfig;
use lumen_core::{ComponentTree, LumenError};
use lumen_format::LtqContainer;
use lumen_models::{architecture_from_config, architecture_from_header, known_components};

#[derive(Parser, Debug)]
#[command(
    name = "lumen",
    version,
    about = "Hybrid GPU/CPU/NVMe LLM and multimodal inference engine (LTQ)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Inspect the resolved PlacementPlan, access modes, and rule provenance for a JSON config.
    Inspect {
        /// Path to the JSON configuration file (e.g., examples/example-moe.json).
        #[arg(short, long)]
        config: PathBuf,

        /// Treat out-of-range layer selectors as hard errors instead of warnings.
        #[arg(long)]
        strict: bool,

        /// Take the model architecture from this HuggingFace `config.json` instead of opening
        /// the `.ltq` file named by `model.path` (lets you inspect a placement before the model
        /// has been quantized).
        #[arg(long, value_name = "CONFIG_JSON")]
        model_config: Option<PathBuf>,
    },

    /// Single-prompt generation in the terminal.
    Run {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(short, long)]
        prompt: String,
    },

    /// Interactive multi-turn terminal chat with prefix caching.
    Chat {
        #[arg(short, long)]
        config: PathBuf,
    },

    /// Start the OpenAI / Anthropic compatible HTTP SSE server.
    Serve {
        #[arg(short, long)]
        config: PathBuf,
    },

    /// Measure sustained VRAM, DRAM, PCIe, and NVMe bandwidths and write hw-profile.json.
    Probe {
        #[arg(short, long, default_value = "hw-profile.json")]
        output: PathBuf,
    },

    /// Benchmark prompt processing (pp), token generation (tg), and hardware efficiency (eta).
    Bench {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(long)]
        trace: Option<PathBuf>,
    },

    /// Repack .ltq weights into backend-specific tile layouts (CpuBand16 / GpuWarp) on disk.
    Pack {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Quantize a BF16 safetensors model checkpoint into an .ltq v1 container.
    Quantize {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = 3.5)]
        bpw: f64,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run_cmd(cli.command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

type CliResult = Result<(), Box<dyn std::error::Error>>;

fn component_tree_for(
    cfg: &RuntimeConfig,
    model_config: Option<&PathBuf>,
) -> Result<ComponentTree, LumenError> {
    match model_config {
        Some(path) => {
            let text = std::fs::read_to_string(path)?;
            let json = serde_json::from_str(&text).map_err(|e| {
                LumenError::Format(format!("invalid JSON in '{}': {e}", path.display()))
            })?;
            Ok(architecture_from_config(&json)?.components())
        }
        None => {
            let container = LtqContainer::open(&cfg.model_path)?;
            Ok(architecture_from_header(&container.header)?.components())
        }
    }
}

fn run_cmd(cmd: Commands) -> CliResult {
    match cmd {
        Commands::Inspect {
            config,
            strict,
            model_config,
        } => {
            let cfg = RuntimeConfig::from_file(&config)?;
            let tree = component_tree_for(&cfg, model_config.as_ref())?;
            let plan = cfg.resolve_plan(tree, &known_components(), strict)?;
            print!("{}", plan.format_inspect(&cfg));
            Ok(())
        }
        // Not implemented yet: fail loudly so scripts do not mistake a no-op for success.
        other => {
            let name = format!("{other:?}");
            let name = name
                .split([' ', '{'])
                .next()
                .unwrap_or("command")
                .to_lowercase();
            Err(format!(
                "'{name}' is not implemented yet (see docs/EN/roadmap.md for the phase that adds it)"
            )
            .into())
        }
    }
}
