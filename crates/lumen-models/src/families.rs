//! Model families: each one turns a `config.json` into a [`Layout`] using the shared modules in
//! [`crate::blocks`]. This file (together with `lib.rs`) is the only place that names model
//! families (architecture rule 1).

use lumen_core::Result;

use crate::blocks::{AttentionSpec, Ffn, LayerSpec, Layout, LinearProjections, Mixer};
use crate::hf::{HfConfig, err};

/// Parses a config into a [`Layout`].
pub(crate) type ParseFn = fn(&HfConfig<'_>) -> Result<Layout>;

/// A registered family.
pub(crate) struct Family {
    /// `model_type` values handled by this family.
    pub model_types: &'static [&'static str],
    pub name: &'static str,
    pub parse: ParseFn,
}

pub(crate) const FAMILIES: &[Family] = &[
    Family {
        model_types: &["llama", "mistral", "qwen2"],
        name: "dense",
        parse: dense_plain,
    },
    Family {
        model_types: &["qwen3"],
        name: "dense",
        parse: dense_qk_norm,
    },
    Family {
        model_types: &["qwen3_moe"],
        name: "moe",
        parse: qwen3_moe,
    },
    Family {
        model_types: &["mixtral"],
        name: "moe",
        parse: mixtral,
    },
    Family {
        model_types: &["gpt_oss"],
        name: "gpt-oss",
        parse: gpt_oss,
    },
    Family {
        model_types: &["qwen3_next"],
        name: "qwen3_next",
        parse: qwen3_next,
    },
    Family {
        model_types: &["qwen3_5"],
        name: "qwen3_5_dense",
        parse: qwen3_5_dense,
    },
    Family {
        model_types: &["qwen3_5_moe"],
        name: "qwen3_5_moe",
        parse: qwen3_5_moe,
    },
    Family {
        model_types: &["qwen4_exp"],
        name: "qwen4_exp",
        parse: qwen4_exp,
    },
    Family {
        model_types: &["gemma4", "gemma4_text"],
        name: "gemma4",
        parse: gemma4,
    },
];

fn layout(name: &str, layers: Vec<LayerSpec>) -> Layout {
    Layout {
        name: name.to_string(),
        layers,
        hyper_connections: false,
        ple: false,
        vision_tower: false,
        audio_tower: false,
        mtp: false,
    }
}

fn attention_layer(qk_norm: bool, ffn: Ffn) -> LayerSpec {
    LayerSpec {
        mixer: Mixer::Attention(AttentionSpec::standard(qk_norm)),
        ffn,
    }
}

// ---------------------------------------------------------------------------------------------
// Plain dense and MoE transformers
// ---------------------------------------------------------------------------------------------

fn dense_with(cfg: &HfConfig<'_>, name: &str, qk_norm: bool) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    Ok(layout(name, vec![attention_layer(qk_norm, Ffn::Dense); n]))
}

fn dense_plain(cfg: &HfConfig<'_>) -> Result<Layout> {
    dense_with(cfg, "dense", false)
}

fn dense_qk_norm(cfg: &HfConfig<'_>) -> Result<Layout> {
    dense_with(cfg, "dense", true)
}

/// Layers that are MoE under the `mlp_only_layers` / `decoder_sparse_step` convention.
fn sparse_step_ffn(cfg: &HfConfig<'_>, n: usize, shared_experts: bool) -> Result<Vec<Ffn>> {
    let has_experts = cfg.usize_or("num_experts", 1)? > 0;
    let step = cfg.usize_or("decoder_sparse_step", 1)?.max(1);
    let dense_only = cfg.layer_index_list("mlp_only_layers", n)?;
    Ok((0..n)
        .map(|l| {
            if has_experts && !dense_only.contains(&l) && (l + 1) % step == 0 {
                Ffn::Moe { shared_experts }
            } else {
                Ffn::Dense
            }
        })
        .collect())
}

fn qwen3_moe(cfg: &HfConfig<'_>) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    let layers = sparse_step_ffn(cfg, n, false)?
        .into_iter()
        .map(|ffn| attention_layer(true, ffn))
        .collect();
    Ok(layout("moe", layers))
}

fn mixtral(cfg: &HfConfig<'_>) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    Ok(layout(
        "moe",
        vec![
            attention_layer(
                false,
                Ffn::Moe {
                    shared_experts: false
                }
            );
            n
        ],
    ))
}

fn gpt_oss(cfg: &HfConfig<'_>) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    if let Some(types) = cfg.layer_types(n)? {
        for t in types {
            if !matches!(t, "sliding_attention" | "full_attention") {
                return Err(err(format!("unknown layer type '{t}'")));
            }
        }
    }
    Ok(layout(
        "gpt-oss",
        vec![
            attention_layer(
                false,
                Ffn::Moe {
                    shared_experts: false
                }
            );
            n
        ],
    ))
}

// ---------------------------------------------------------------------------------------------
// Hybrid GatedDeltaNet + gated attention families
// ---------------------------------------------------------------------------------------------

/// Which layers are full attention in a 3:1 hybrid: `layer_types` when present, otherwise every
/// `full_attention_interval`-th layer (default 4).
fn hybrid_full_attention(cfg: &HfConfig<'_>, n: usize) -> Result<Vec<bool>> {
    if let Some(types) = cfg.layer_types(n)? {
        return types
            .into_iter()
            .map(|t| match t {
                "full_attention" => Ok(true),
                "linear_attention" => Ok(false),
                other => Err(err(format!("unknown layer type '{other}'"))),
            })
            .collect();
    }
    let interval = cfg.usize_or("full_attention_interval", 4)?;
    if interval == 0 {
        return Err(err("'full_attention_interval' must be greater than 0"));
    }
    Ok((0..n).map(|l| (l + 1) % interval == 0).collect())
}

fn hybrid(
    cfg: &HfConfig<'_>,
    name: &str,
    projections: LinearProjections,
    ffn: &[Ffn],
    indexer: bool,
) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    let full = hybrid_full_attention(cfg, n)?;
    let layers = (0..n)
        .map(|l| {
            let mixer = if full[l] {
                Mixer::Attention(AttentionSpec {
                    indexer,
                    ..AttentionSpec::standard(true)
                })
            } else {
                Mixer::LinearAttention(projections)
            };
            LayerSpec { mixer, ffn: ffn[l] }
        })
        .collect();
    let mut out = layout(name, layers);
    out.vision_tower = cfg.root_has("vision_config");
    // The family ships a multi-token-prediction draft layer (docs/EN/models/qwen3_next.md).
    out.mtp = true;
    Ok(out)
}

fn qwen3_next(cfg: &HfConfig<'_>) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    let ffn = sparse_step_ffn(cfg, n, true)?;
    hybrid(cfg, "qwen3_next", LinearProjections::Fused, &ffn, false)
}

fn qwen3_5_dense(cfg: &HfConfig<'_>) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    hybrid(
        cfg,
        "qwen3_5_dense",
        LinearProjections::Split,
        &vec![Ffn::Dense; n],
        false,
    )
}

fn qwen3_5_moe(cfg: &HfConfig<'_>) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    let ffn = vec![
        Ffn::Moe {
            shared_experts: true
        };
        n
    ];
    hybrid(cfg, "qwen3_5_moe", LinearProjections::Split, &ffn, false)
}

fn qwen4_exp(cfg: &HfConfig<'_>) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    let ffn = vec![
        Ffn::Moe {
            shared_experts: true
        };
        n
    ];
    let mut out = hybrid(cfg, "qwen4_exp", LinearProjections::Split, &ffn, true)?;
    out.hyper_connections = true;
    out.ple = true;
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Gemma 4
// ---------------------------------------------------------------------------------------------

fn gemma4(cfg: &HfConfig<'_>) -> Result<Layout> {
    let n = cfg.num_hidden_layers()?;
    let types = cfg
        .layer_types(n)?
        .ok_or_else(|| err("missing 'layer_types' (sliding_attention / full_attention)"))?;
    let k_eq_v = cfg.bool_or("attention_k_eq_v", false)?;
    let shared_kv = cfg.usize_or("num_kv_shared_layers", 0)?;
    if shared_kv >= n {
        return Err(err(format!(
            "'num_kv_shared_layers' = {shared_kv} must be < {n}"
        )));
    }
    let ffn = if cfg.usize_or("num_experts", 0)? > 0 {
        // Dense MLP evaluated in parallel with the MoE block (26B-A4B).
        Ffn::DenseAndMoe {
            shared_experts: false,
        }
    } else {
        Ffn::Dense
    };

    let mut layers = Vec::with_capacity(n);
    for (l, t) in types.into_iter().enumerate() {
        let global = match t {
            "full_attention" => true,
            "sliding_attention" => false,
            other => return Err(err(format!("unknown layer type '{other}'"))),
        };
        // The last `num_kv_shared_layers` layers own no K/V projections and no KV cache; they read
        // the cache of an earlier layer of the same type.
        let shares_kv = l >= n - shared_kv;
        let spec = AttentionSpec {
            k_proj: !shares_kv,
            v_proj: !shares_kv && !(global && k_eq_v),
            qk_norm: true,
            kv_cache: !shares_kv,
            indexer: false,
        };
        layers.push(LayerSpec {
            mixer: Mixer::Attention(spec),
            ffn,
        });
    }

    let mut out = layout("gemma4", layers);
    out.ple = cfg.usize_or("hidden_size_per_layer_input", 0)? > 0;
    out.vision_tower = cfg.root_has("vision_config");
    out.audio_tower = cfg.root_has("audio_config");
    Ok(out)
}
