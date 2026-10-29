//! Component trees of the supported families, and the example configs resolved against them.

use std::path::Path;

use lumen_config::RuntimeConfig;
use lumen_core::{AccessMode, ComponentKind, ComponentPath, ComponentTree, Device, MemLoc};
use lumen_models::{architecture_from_config, known_components, supported_model_types};
use serde_json::{Value, json};

fn tree(config: Value) -> ComponentTree {
    architecture_from_config(&config).unwrap().components()
}

fn layer_set(t: &ComponentTree, path: &str) -> Vec<usize> {
    t.layer_leaves
        .get(&ComponentPath::new(path))
        .map(|l| l.active_layers.iter().collect())
        .unwrap_or_default()
}

fn has_global(t: &ComponentTree, path: &str) -> bool {
    t.global_leaves.contains_key(&ComponentPath::new(path))
}

fn hybrid_types(n: usize) -> Vec<&'static str> {
    (0..n)
        .map(|l| {
            if (l + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect()
}

fn qwen3_5_27b() -> Value {
    json!({ "model_type": "qwen3_5", "vision_config": {},
            "text_config": { "model_type": "qwen3_5_text", "num_hidden_layers": 64,
                             "layer_types": hybrid_types(64) } })
}

fn qwen4_exp_48() -> Value {
    json!({ "model_type": "qwen4_exp", "vision_config": {},
            "num_hidden_layers": 48, "full_attention_interval": 4, "num_experts": 512 })
}

fn gemma_types(n: usize, every: usize) -> Vec<&'static str> {
    (0..n)
        .map(|l| {
            if (l + 1) % every == 0 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect()
}

fn gemma4_e4b() -> Value {
    json!({ "model_type": "gemma4", "vision_config": {}, "audio_config": {},
            "text_config": { "model_type": "gemma4_text", "num_hidden_layers": 42,
                             "layer_types": gemma_types(42, 6), "num_kv_shared_layers": 18,
                             "hidden_size_per_layer_input": 256 } })
}

fn gemma4_26b_a4b() -> Value {
    json!({ "model_type": "gemma4", "vision_config": {},
            "text_config": { "model_type": "gemma4_text", "num_hidden_layers": 30,
                             "layer_types": gemma_types(30, 6), "attention_k_eq_v": true,
                             "num_experts": 128 } })
}

fn all_samples() -> Vec<(&'static str, Value)> {
    vec![
        (
            "llama",
            json!({ "model_type": "llama", "num_hidden_layers": 32 }),
        ),
        (
            "qwen3",
            json!({ "model_type": "qwen3", "num_hidden_layers": 28 }),
        ),
        (
            "qwen3_moe",
            json!({ "model_type": "qwen3_moe", "num_hidden_layers": 24, "num_experts": 128 }),
        ),
        (
            "mixtral",
            json!({ "model_type": "mixtral", "num_hidden_layers": 32 }),
        ),
        (
            "gpt_oss",
            json!({ "model_type": "gpt_oss", "num_hidden_layers": 24,
                           "layer_types": gemma_types(24, 2) }),
        ),
        (
            "qwen3_next",
            json!({ "model_type": "qwen3_next", "num_hidden_layers": 48, "num_experts": 512 }),
        ),
        ("qwen3_5", qwen3_5_27b()),
        (
            "qwen3_5_moe",
            json!({ "model_type": "qwen3_5_moe", "num_hidden_layers": 40 }),
        ),
        ("qwen4_exp", qwen4_exp_48()),
        ("gemma4 E4B", gemma4_e4b()),
        ("gemma4 26B", gemma4_26b_a4b()),
    ]
}

// -------------------------------------------------------------------------------------------
// Families
// -------------------------------------------------------------------------------------------

#[test]
fn llama_is_plain_dense() {
    let t = tree(json!({ "model_type": "llama", "num_hidden_layers": 4 }));
    let all: Vec<usize> = (0..4).collect();
    assert_eq!(layer_set(&t, "attention/q"), all);
    assert_eq!(layer_set(&t, "attention/kv-cache"), all);
    assert_eq!(layer_set(&t, "dense-mlp/gate"), all);
    // No recurrent state, no experts, no QK-norm, no adapters.
    assert!(layer_set(&t, "linear-attention/state").is_empty());
    assert!(layer_set(&t, "moe/router").is_empty());
    assert!(layer_set(&t, "attention/qk-norm").is_empty());
    assert!(!has_global(&t, "ple/table") && !has_global(&t, "mtp"));
    assert!(layer_set(&t, "hyper-connections").is_empty());

    let q3 = tree(json!({ "model_type": "qwen3", "num_hidden_layers": 4 }));
    assert_eq!(layer_set(&q3, "attention/qk-norm"), all);
}

#[test]
fn qwen3_5_dense_is_a_3_to_1_hybrid_without_experts() {
    let t = tree(qwen3_5_27b());
    let attn: Vec<usize> = (0..64).filter(|l| (l + 1) % 4 == 0).collect();
    let gdn: Vec<usize> = (0..64).filter(|l| (l + 1) % 4 != 0).collect();
    assert_eq!((attn.len(), gdn.len()), (16, 48));
    assert_eq!(layer_set(&t, "attention/q"), attn);
    assert_eq!(layer_set(&t, "attention/kv-cache"), attn);
    assert_eq!(layer_set(&t, "linear-attention/state"), gdn);
    // Split projections (not the fused qkvz/ba of qwen3_next).
    assert_eq!(layer_set(&t, "linear-attention/projections/qkv"), gdn);
    assert!(layer_set(&t, "linear-attention/projections/qkvz").is_empty());
    assert_eq!(layer_set(&t, "dense-mlp/gate").len(), 64);
    assert!(layer_set(&t, "moe/router").is_empty());
    assert!(has_global(&t, "vision-tower") && has_global(&t, "mtp"));
    assert!(!has_global(&t, "ple/table"));
}

#[test]
fn qwen3_next_uses_fused_projections_and_moe_with_shared_expert() {
    let t = tree(
        json!({ "model_type": "qwen3_next", "num_hidden_layers": 8, "num_experts": 16,
                         "mlp_only_layers": [0], "decoder_sparse_step": 1 }),
    );
    assert_eq!(layer_set(&t, "linear-attention/projections/qkvz").len(), 6);
    assert!(layer_set(&t, "linear-attention/projections/qkv").is_empty());
    assert_eq!(layer_set(&t, "moe/router"), vec![1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(layer_set(&t, "dense-mlp/gate"), vec![0]);
    assert_eq!(
        layer_set(&t, "moe/shared-experts/gate"),
        vec![1, 2, 3, 4, 5, 6, 7]
    );
    assert!(
        !has_global(&t, "vision-tower"),
        "text-only without vision_config"
    );
}

#[test]
fn qwen4_exp_adds_qsa_hyper_connections_and_ngram_table() {
    let t = tree(qwen4_exp_48());
    let attn: Vec<usize> = (0..48).filter(|l| (l + 1) % 4 == 0).collect();
    assert_eq!(attn.len(), 12);
    assert_eq!(layer_set(&t, "linear-attention/state").len(), 36);
    assert_eq!(
        layer_set(&t, "attention/indexer/keys"),
        attn,
        "QSA indexer on full layers only"
    );
    assert_eq!(layer_set(&t, "hyper-connections").len(), 48);
    assert_eq!(layer_set(&t, "moe/routed-experts/projections").len(), 48);
    assert!(has_global(&t, "ple/table") && has_global(&t, "mtp") && has_global(&t, "vision-tower"));
    assert_eq!(
        t.global_leaves[&ComponentPath::new("ple/table")].kind,
        ComponentKind::RowTable
    );
}

#[test]
fn gemma4_e4b_shares_kv_on_last_layers_and_has_ple() {
    let t = tree(gemma4_e4b());
    let own_kv: Vec<usize> = (0..24).collect();
    assert_eq!(
        layer_set(&t, "attention/kv-cache"),
        own_kv,
        "last 18 of 42 layers share KV"
    );
    assert_eq!(layer_set(&t, "attention/k"), own_kv);
    assert_eq!(layer_set(&t, "attention/q").len(), 42);
    assert!(layer_set(&t, "linear-attention/state").is_empty());
    assert!(layer_set(&t, "moe/router").is_empty());
    assert!(has_global(&t, "ple/table") && has_global(&t, "audio-tower"));
}

#[test]
fn gemma4_26b_has_parallel_dense_and_moe_and_k_equals_v_on_global_layers() {
    let t = tree(gemma4_26b_a4b());
    let global: Vec<usize> = (0..30).filter(|l| (l + 1) % 6 == 0).collect();
    assert_eq!(global.len(), 5);
    let with_v: Vec<usize> = (0..30).filter(|l| !global.contains(l)).collect();
    assert_eq!(
        layer_set(&t, "attention/v"),
        with_v,
        "no v_proj on K = V global layers"
    );
    assert_eq!(layer_set(&t, "attention/k").len(), 30);
    assert_eq!(layer_set(&t, "dense-mlp/gate").len(), 30);
    assert_eq!(layer_set(&t, "moe/router").len(), 30);
    assert!(layer_set(&t, "moe/shared-experts/gate").is_empty());
    assert!(!has_global(&t, "ple/table") && !has_global(&t, "audio-tower"));
}

#[test]
fn gpt_oss_has_experts_but_no_shared_expert_or_hybrid_modules() {
    let t = tree(json!({ "model_type": "gpt_oss", "num_hidden_layers": 4,
                         "layer_types": ["sliding_attention", "full_attention",
                                         "sliding_attention", "full_attention"] }));
    assert_eq!(layer_set(&t, "moe/routed-experts/projections").len(), 4);
    assert!(layer_set(&t, "moe/shared-experts/projections").is_empty());
    assert!(layer_set(&t, "linear-attention/state").is_empty());
    assert!(layer_set(&t, "attention/indexer/keys").is_empty());
    assert!(!has_global(&t, "ple/table"));
}

#[test]
fn qwen3_moe_respects_sparse_step() {
    let t = tree(
        json!({ "model_type": "qwen3_moe", "num_hidden_layers": 6, "num_experts": 8,
                         "decoder_sparse_step": 2 }),
    );
    assert_eq!(layer_set(&t, "moe/router"), vec![1, 3, 5]);
    assert_eq!(layer_set(&t, "dense-mlp/gate"), vec![0, 2, 4]);
}

// -------------------------------------------------------------------------------------------
// Errors and invariants
// -------------------------------------------------------------------------------------------

#[test]
fn unknown_or_malformed_configs_are_errors_not_guesses() {
    let err = |c: Value| {
        architecture_from_config(&c)
            .err()
            .expect("must fail")
            .to_string()
    };

    assert!(
        err(json!({ "model_type": "some_future_model", "num_hidden_layers": 4 }))
            .contains("unsupported model_type 'some_future_model'")
    );
    assert!(err(json!({ "num_hidden_layers": 4 })).contains("<missing>"));
    assert!(err(json!([1, 2])).contains("JSON object"));
    assert!(err(json!({ "model_type": "llama" })).contains("num_hidden_layers"));
    assert!(err(json!({ "model_type": "llama", "num_hidden_layers": 0 })).contains("outside"));
    assert!(
        err(json!({ "model_type": "llama", "num_hidden_layers": 1_000_000 })).contains("outside")
    );
    assert!(err(json!({ "model_type": "llama", "num_hidden_layers": "4" })).contains("integer"));
    assert!(
        err(json!({ "model_type": "qwen3_next", "num_hidden_layers": 4,
                        "layer_types": ["full_attention"] }))
        .contains("entries")
    );
    assert!(
        err(json!({ "model_type": "qwen3_next", "num_hidden_layers": 2,
                        "layer_types": ["full_attention", "mamba"] }))
        .contains("unknown layer type")
    );
    assert!(
        err(json!({ "model_type": "qwen3_next", "num_hidden_layers": 2,
                        "full_attention_interval": 0 }))
        .contains("greater than 0")
    );
    assert!(err(json!({ "model_type": "gemma4", "num_hidden_layers": 2 })).contains("layer_types"));
    assert!(
        err(json!({ "model_type": "gemma4", "num_hidden_layers": 2,
                        "layer_types": ["full_attention", "full_attention"],
                        "num_kv_shared_layers": 2 }))
        .contains("num_kv_shared_layers")
    );
    assert!(
        err(json!({ "model_type": "qwen3_moe", "num_hidden_layers": 2,
                        "mlp_only_layers": [7] }))
        .contains("mlp_only_layers")
    );
}

#[test]
fn model_with_more_than_128_layers_keeps_every_layer() {
    let t = tree(json!({ "model_type": "llama", "num_hidden_layers": 200 }));
    assert_eq!(layer_set(&t, "attention/q"), (0..200).collect::<Vec<_>>());
}

#[test]
fn every_component_of_every_family_is_in_the_known_namespace_with_the_same_kind() {
    let known = known_components();
    for (name, config) in all_samples() {
        let t = tree(config);
        for (path, leaf) in &t.global_leaves {
            let k = known
                .global_leaves
                .get(path)
                .unwrap_or_else(|| panic!("{name}: {path} unknown"));
            assert_eq!(k.kind, leaf.kind, "{name}: kind of {path}");
        }
        for (path, leaf) in &t.layer_leaves {
            let k = known
                .layer_leaves
                .get(path)
                .unwrap_or_else(|| panic!("{name}: {path} unknown"));
            assert_eq!(k.kind, leaf.kind, "{name}: kind of {path}");
            assert!(
                !leaf.active_layers.is_empty(),
                "{name}: {path} registered but never active"
            );
        }
    }
}

#[test]
fn registry_lists_every_supported_model_type() {
    let types = supported_model_types();
    for needed in [
        "llama",
        "qwen3",
        "qwen3_moe",
        "mixtral",
        "gpt_oss",
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen4_exp",
        "gemma4",
    ] {
        assert!(types.contains(&needed), "{needed}");
    }
}

// -------------------------------------------------------------------------------------------
// The example configs shipped in `examples/`, resolved against the architectures they document
// -------------------------------------------------------------------------------------------

fn example(name: &str) -> RuntimeConfig {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(name);
    RuntimeConfig::from_file(path).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn example_dense_resolves_on_qwen3_5_27b() {
    let cfg = example("example-dense.json");
    let arch = architecture_from_config(&qwen3_5_27b()).unwrap();
    let plan = cfg
        .resolve_plan(arch.components(), &known_components(), true)
        .unwrap();
    assert!(
        plan.provenance.skipped_rules.is_empty(),
        "{:?}",
        plan.provenance.skipped_rules
    );

    let input = plan.global[&ComponentPath::new("input-layer")];
    assert_eq!(
        (input.store, input.compute),
        (MemLoc::Ram(0), Device::Cuda(0))
    );
    assert_eq!(
        input.decode_access_mode(ComponentKind::RowTable),
        AccessMode::ZeroCopy
    );

    let gate = ComponentPath::new("dense-mlp/gate");
    let l0 = plan.layers[0][&gate];
    assert_eq!((l0.compute, l0.store), (Device::Cpu(0), MemLoc::Ram(0)));
    assert_eq!(l0.prefill_compute, Some(Device::Cuda(0)));
    assert_eq!(
        l0.decode_access_mode(ComponentKind::DenseWeight),
        AccessMode::Direct
    );
    assert_eq!(
        l0.prefill_access_mode(ComponentKind::DenseWeight),
        AccessMode::Streamed
    );

    let l36 = plan.layers[36][&gate];
    assert_eq!((l36.compute, l36.store), (Device::Cuda(0), MemLoc::Vram(0)));

    // Hybrid: KV cache only on the 16 full-attention layers, recurrent state on the other 48.
    assert_eq!(plan.kv.len(), 16);
    assert_eq!(plan.state.len(), 48);
    assert!(
        plan.kv
            .iter()
            .all(|k| (k.layer + 1) % 4 == 0 && k.k_bits == Some(8))
    );
}

#[test]
fn example_moe_resolves_on_qwen4_exp() {
    let cfg = example("example-moe.json");
    let arch = architecture_from_config(&qwen4_exp_48()).unwrap();
    let plan = cfg
        .resolve_plan(arch.components(), &known_components(), true)
        .unwrap();
    assert!(
        plan.provenance.skipped_rules.is_empty(),
        "{:?}",
        plan.provenance.skipped_rules
    );

    let table = plan.global[&ComponentPath::new("ple/table")];
    assert_eq!(table.store, MemLoc::Disk);
    assert_eq!(
        table.decode_access_mode(ComponentKind::RowTable),
        AccessMode::FetchH2d
    );

    let experts = ComponentPath::new("moe/routed-experts/projections");
    let l0 = plan.layers[0][&experts];
    assert_eq!((l0.compute, l0.store), (Device::Cpu(0), MemLoc::Ram(0)));
    assert_eq!(l0.prefill_compute, Some(Device::Cuda(0)));
    assert_eq!(
        l0.decode_access_mode(ComponentKind::ExpertBank),
        AccessMode::Direct
    );
    assert_eq!(
        l0.prefill_access_mode(ComponentKind::ExpertBank),
        AccessMode::Streamed
    );
    let l32 = plan.layers[32][&experts];
    assert_eq!((l32.compute, l32.store), (Device::Cuda(0), MemLoc::Vram(0)));

    // KV cache and indexer keys live in RAM on the full-attention layers.
    assert_eq!(plan.kv.len(), 12);
    assert!(plan.kv.iter().all(|k| k.place.store == MemLoc::Ram(0)));
    let keys = ComponentPath::new("attention/indexer/keys");
    assert_eq!(plan.layers[3][&keys].store, MemLoc::Ram(0));
    assert!(
        !plan.layers[0].contains_key(&keys),
        "no indexer on linear-attention layers"
    );
}

#[test]
fn rules_for_modules_a_model_lacks_are_skipped_not_rejected() {
    let cfg = example("example-moe.json");
    // The same MoE example on a dense model: moe/ple/indexer rules do not apply.
    let arch = architecture_from_config(&json!({ "model_type": "llama", "num_hidden_layers": 48 }))
        .unwrap();
    let plan = cfg
        .resolve_plan(arch.components(), &known_components(), true)
        .unwrap();
    assert!(!plan.provenance.skipped_rules.is_empty());
    assert!(
        plan.provenance
            .skipped_rules
            .iter()
            .any(|s| s.contains("routed-experts"))
    );
}

#[test]
fn typos_are_still_rejected_with_a_suggestion() {
    let cfg = RuntimeConfig::from_json_str(
        r#"{"model": {"path": "m", "compute": "cuda:0", "store": "vram:0",
            "hidden-layers": { "moee": { "compute": "cpu:0" } } }}"#,
    )
    .unwrap();
    let arch = architecture_from_config(&qwen4_exp_48()).unwrap();
    let err = cfg
        .resolve_plan(arch.components(), &known_components(), true)
        .unwrap_err();
    assert!(err.to_string().contains("did you mean 'moe'?"), "{err}");
}
