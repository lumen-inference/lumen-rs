//! Shared architectural modules ("block builders") and the layer layout they are composed into.
//!
//! An architecture parses its `config.json` into a [`Layout`]: for every decoder layer, which
//! sequence mixer and which channel mixer it uses, plus the model-level adapters. Everything
//! downstream (the component tree used by `lumen-config`, later the operator graph) is derived
//! from that single description, so model families differ only in how they *fill in* a layout,
//! never in how a tree is built.

use lumen_core::{ComponentKind, ComponentTree, LayerMask, LumenError, Result};

/// Self-attention module (full or sliding window: the config tree does not distinguish them).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionSpec {
    /// Has its own `k_proj` (false on layers that reuse another layer's KV cache).
    pub k_proj: bool,
    /// Has its own `v_proj` (false when `K = V`, and on KV-sharing layers).
    pub v_proj: bool,
    /// Per-head `q_norm` / `k_norm`.
    pub qk_norm: bool,
    /// Owns a paged KV cache (false on KV-sharing layers).
    pub kv_cache: bool,
    /// Sparse-attention block indexer (`QSA`) with its own key cache.
    pub indexer: bool,
}

impl AttentionSpec {
    /// Ordinary attention: own K, V projections and KV cache.
    pub fn standard(qk_norm: bool) -> Self {
        Self {
            k_proj: true,
            v_proj: true,
            qk_norm,
            kv_cache: true,
            indexer: false,
        }
    }
}

/// How the `GatedDeltaNet` input projections are stored.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LinearProjections {
    /// Two fused projections `in_proj_qkvz` and `in_proj_ba`.
    Fused,
    /// Four split projections `in_proj_{qkv,z,b,a}`.
    Split,
}

/// Sequence mixer of a layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Mixer {
    Attention(AttentionSpec),
    /// `GatedDeltaNet` linear attention with a recurrent state.
    LinearAttention(LinearProjections),
}

/// Channel mixer (feed-forward block) of a layer.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Ffn {
    Dense,
    Moe {
        shared_experts: bool,
    },
    /// Dense MLP and MoE evaluated in parallel from the same input.
    DenseAndMoe {
        shared_experts: bool,
    },
}

impl Ffn {
    fn has_dense(self) -> bool {
        matches!(self, Self::Dense | Self::DenseAndMoe { .. })
    }

    fn has_moe(self) -> bool {
        !matches!(self, Self::Dense)
    }

    fn shared_experts(self) -> bool {
        matches!(
            self,
            Self::Moe {
                shared_experts: true
            } | Self::DenseAndMoe {
                shared_experts: true
            }
        )
    }
}

/// One decoder layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerSpec {
    pub mixer: Mixer,
    pub ffn: Ffn,
}

/// Complete structural description of a model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layout {
    /// Architecture name shown by `lumen inspect`.
    pub name: String,
    pub layers: Vec<LayerSpec>,
    /// Multi-stream residuals (`HyperConnections`) in every layer.
    pub hyper_connections: bool,
    /// Per-layer-embedding / n-gram lookup table and its projections.
    pub ple: bool,
    pub vision_tower: bool,
    pub audio_tower: bool,
    /// Multi-token-prediction draft layer.
    pub mtp: bool,
}

impl Layout {
    /// Checks the layout for combinations the component tree cannot express yet.
    ///
    /// A layer's KV cache is read densely by ordinary attention and by gather when a sparse
    /// indexer selects blocks (`QSA`). Both live at the path `attention/kv-cache`, whose access
    /// kind is a property of the path, so a model mixing the two is rejected rather than being
    /// given the wrong kind for half of its layers.
    pub fn validate(&self) -> Result<()> {
        let (mut sparse, mut dense) = (false, false);
        for layer in &self.layers {
            if let Mixer::Attention(a) = &layer.mixer
                && a.kv_cache
            {
                sparse |= a.indexer;
                dense |= !a.indexer;
            }
        }
        if sparse && dense {
            return Err(LumenError::Format(
                "model config: layers with and without a sparse-attention indexer both own a KV \
                 cache; mixed dense/sparse KV caches are not supported"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Builds the component tree (the config/placement namespace) for this layout.
    ///
    /// Only components that exist somewhere in the model are added; a path that exists in
    /// [`known_components`](crate::known_components) but not here is "absent from this model".
    pub fn components(&self) -> ComponentTree {
        use ComponentKind::{
            ComputeOnly, DenseWeight, ExpertBank, IndexerCache, KvCache, RecurrentState, RowTable,
            SparseKvCache,
        };
        let n = self.layers.len();
        let mut tree = ComponentTree::new(self.name.clone(), n);

        // Model-level components.
        tree.add_global("input-layer", RowTable);
        tree.add_global("output-layer/norm", DenseWeight);
        tree.add_global("output-layer/lm-head", DenseWeight);
        if self.ple {
            tree.add_global("ple/table", RowTable);
            tree.add_global("ple/projections", DenseWeight);
        }
        if self.vision_tower {
            tree.add_global("vision-tower", DenseWeight);
        }
        if self.audio_tower {
            tree.add_global("audio-tower", DenseWeight);
        }
        if self.mtp {
            tree.add_global("mtp", DenseWeight);
        }

        // Per-layer components: each group is active on the layers that satisfy its predicate.
        let mut add = |path: &str, kind: ComponentKind, pred: &dyn Fn(&LayerSpec) -> bool| {
            let mask = LayerMask::from_fn(n, |l| pred(&self.layers[l]));
            if !mask.is_empty() {
                tree.add_layer(path, kind, mask);
            }
        };

        let attn = |l: &LayerSpec| matches!(l.mixer, Mixer::Attention(_));
        let attn_with = |l: &LayerSpec, f: fn(&AttentionSpec) -> bool| match &l.mixer {
            Mixer::Attention(a) => f(a),
            Mixer::LinearAttention(_) => false,
        };
        let linear = |l: &LayerSpec, p: Option<LinearProjections>| match &l.mixer {
            Mixer::LinearAttention(have) => p.is_none_or(|p| p == *have),
            Mixer::Attention(_) => false,
        };

        add("norms/input-norm", DenseWeight, &|_| true);
        add("norms/post-attention-norm", DenseWeight, &|_| true);

        // Attention
        for proj in ["attention/q", "attention/o"] {
            add(proj, DenseWeight, &attn);
        }
        add("attention/k", DenseWeight, &|l| attn_with(l, |a| a.k_proj));
        add("attention/v", DenseWeight, &|l| attn_with(l, |a| a.v_proj));
        add("attention/qk-norm", DenseWeight, &|l| {
            attn_with(l, |a| a.qk_norm)
        });
        add("attention/rope", ComputeOnly, &attn);
        // Dense or sparse (QSA) access is a property of the path; `validate` rules out mixtures.
        let sparse_kv = self
            .layers
            .iter()
            .any(|l| matches!(&l.mixer, Mixer::Attention(a) if a.kv_cache && a.indexer));
        let kv_kind = if sparse_kv { SparseKvCache } else { KvCache };
        add("attention/kv-cache", kv_kind, &|l| {
            attn_with(l, |a| a.kv_cache)
        });
        add("attention/indexer/keys", IndexerCache, &|l| {
            attn_with(l, |a| a.indexer)
        });

        // Linear attention (GatedDeltaNet)
        let gdn_always = [
            "linear-attention/projections/out",
            "linear-attention/conv",
            "linear-attention/norm",
        ];
        for path in gdn_always {
            add(path, DenseWeight, &|l| linear(l, None));
        }
        for path in [
            "linear-attention/projections/qkvz",
            "linear-attention/projections/ba",
        ] {
            add(path, DenseWeight, &|l| {
                linear(l, Some(LinearProjections::Fused))
            });
        }
        for path in [
            "linear-attention/projections/qkv",
            "linear-attention/projections/z",
            "linear-attention/projections/b",
            "linear-attention/projections/a",
        ] {
            add(path, DenseWeight, &|l| {
                linear(l, Some(LinearProjections::Split))
            });
        }
        add("linear-attention/state", RecurrentState, &|l| {
            linear(l, None)
        });

        add("hyper-connections", DenseWeight, &|_| {
            self.hyper_connections
        });

        // Channel mixers
        for path in ["dense-mlp/gate", "dense-mlp/up", "dense-mlp/down"] {
            add(path, DenseWeight, &|l| l.ffn.has_dense());
        }
        add("moe/router", DenseWeight, &|l| l.ffn.has_moe());
        add("moe/routed-experts/projections", ExpertBank, &|l| {
            l.ffn.has_moe()
        });
        for path in ["moe/shared-experts/gate", "moe/shared-experts/projections"] {
            add(path, DenseWeight, &|l| l.ffn.shared_experts());
        }

        tree
    }
}

/// A layout that contains every module in every variant: the namespace of all component paths
/// any architecture can produce.
pub(crate) fn superset_layout() -> Layout {
    let everything = AttentionSpec {
        k_proj: true,
        v_proj: true,
        qk_norm: true,
        kv_cache: true,
        indexer: true,
    };
    let ffn = Ffn::DenseAndMoe {
        shared_experts: true,
    };
    Layout {
        name: "known-components".to_string(),
        layers: vec![
            LayerSpec {
                mixer: Mixer::Attention(everything),
                ffn,
            },
            LayerSpec {
                mixer: Mixer::LinearAttention(LinearProjections::Fused),
                ffn,
            },
            LayerSpec {
                mixer: Mixer::LinearAttention(LinearProjections::Split),
                ffn,
            },
        ],
        hyper_connections: true,
        ple: true,
        vision_tower: true,
        audio_tower: true,
        mtp: true,
    }
}
