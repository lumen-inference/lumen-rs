//! `lumen` CLI binary (`docs/RU/api.md` Section 2):
//! `inspect`, `run`, `chat`, `serve`, `probe`, `bench`, `pack`, `quantize`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use lumen_config::RuntimeConfig;

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

        /// Optional architecture preset override (dense, moe, qwen3_next, qwen4_exp, gemma4).
        #[arg(long)]
        preset: Option<String>,

        /// Optional number of hidden layers override when inspecting without a .ltq model file.
        #[arg(long)]
        layers: Option<usize>,

        /// Treat out-of-range layer selectors as hard errors instead of warnings.
        #[arg(long)]
        strict: bool,
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
    if let Err(err) = run_cmd(cli.command) {
        eprintln!("error: {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run_cmd(cmd: Commands) -> lumen_core::Result<()> {
    match cmd {
        Commands::Inspect {
            config,
            preset,
            layers,
            strict,
        } => {
            let cfg = RuntimeConfig::from_file(&config)?;
            let tree = cfg.infer_tree_for_inspect(preset.as_deref(), layers)?;
            let plan = cfg.resolve_plan(tree, strict)?;
            print!("{}", plan.format_inspect(&cfg));
            Ok(())
        }
        other => {
            eprintln!(
                "command '{other:?}' will be wired in upcoming phases (see docs/RU/roadmap.md)"
            );
            Ok(())
        }
    }
}
