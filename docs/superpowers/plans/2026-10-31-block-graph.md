# Composable blocks and first CPU execution

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute a small decoder built from reusable blocks, with explicit storage ownership and numerically checked prefill/decode, without adding model-specific paths to the executor.

**Architecture:** `lumen-models` describes a validated logical graph. Its values, weights, and sequence stores have distinct identities. Reusable blocks expand into that graph. `lumen-exec` validates placement and capacities, lowers the graph to backend operations and reserves buffers. `lumen-ref` executes those operations through explicit bindings; it does not depend on model descriptions.

**Tech Stack:** Rust 2024, MSRV 1.89, existing workspace dependencies; Python standard library for independent numerical fixtures.

**Spec:** `docs/EN/architecture.md`, particularly sections 1, 4, 7 and 9, and the user's requirement that shared mechanisms express computation rather than model-family special cases.

## Global constraints

- No family-name branches in backends, memory or execution preparation.
- No new crate. Preserve existing dependency direction and CUDA-free build.
- Q/K/V are logical inputs. Physical sequence storage has no privileged K/V role. Sharing a handle never allocates a second store.
- Validate dimensions, identities, placement, capacities and required weights before computation.
- State changes commit only after a successful step. Appended but uncommitted rows can be overwritten on retry; existing committed rows remain intact.
- Initial supported path: one sequence, dense f32 weights, causal full/sliding attention, grouped query heads, rotary embedding, gated MLP and residual composition. Other algorithms and data formats fail explicitly instead of pretending to execute.
- This delivers reusable graph APIs and an executable decoder example; checkpoint/tokenizer integration and the production `lumen run` interface remain subsequent work.

## Review focus

1. Shared storage must be counted once, including when both attention inputs use it or another layer reads it.
2. Cross-graph handles, mismatched dimensions, duplicate writes and missing weights must fail before state mutation.
3. Full prefill, chunked prefill and repeated decode must agree numerically, including rotary positions and grouped heads.
4. Failed steps and capacity errors must not advance committed sequence length or corrupt earlier rows.
5. Memory reports must include all actual prepared weight, state and temporary tensor payloads; no concealed full-history copies.

## Tasks

### 1. Validated graph and storage descriptions

Files: `crates/lumen-models/src/graph/`, `crates/lumen-models/tests/graph.rs`, common mathematical specifications in `lumen-tensor`.

- [x] Write tests for shape/identity errors, explicit shared stores and conflicting writes; watch them fail.
- [x] Implement typed handles, dynamic row domains, weight/store declarations and checked graph operations. Keep immutable graph accessors for preparation.
- [x] Derive the placement component tree from declarations; retain legacy config-only Layout until family adapters are migrated.

### 2. Reusable compositions

Files: `crates/lumen-models/src/blocks/`, tests in `crates/lumen-models/tests/graph.rs`.

- [x] Test compositions that reuse linear/norm/activation/attention and share stores without assuming two caches.
- [x] Implement a build-time Block contract, linear and norm blocks, gated MLP, projected attention and pre-norm residual composition.
- [x] Expose plain attention separately from projection and storage choices. Build a small decoder from these blocks.

### 3. Preparation and buffer bindings

Files: `crates/lumen-exec/src/`, `crates/lumen-backend/src/`, `crates/lumen-exec/tests/execution.rs`.

- [x] Test unsupported placement, memory/capacity overflow, missing/malformed weights and shared storage accounting.
- [x] Lower logical operations to the existing backend interface, replacing implicit layer/K=V state with explicit operands and append operations.
- [x] Reserve bounded host buffers once. Bind sequence reads directly to their store; expose payload accounting.
- [x] Pass actual step sizes/positions separately from the immutable schedule. Define reset/commit behavior.

### 4. Numerical execution and independent fixture

Files: `crates/lumen-ref/src/`, `crates/lumen-exec/tests/fixtures/`, `crates/lumen-exec/examples/`.

- [x] Produce independent Python reference logits for a tiny dense decoder; compare all input partitions. Minimal shared-history execution tests were run red before dispatch; the full decoder fixture was added after dispatch.
- [x] Implement checked scalar f32 dispatch for the supported operations; retain explicit errors for unsupported variants.
- [x] Test full/chunked/token decode, shared/derived state, failed-step recovery, and a runnable example.

### 5. Review and validation

- [x] Run targeted tests while developing, then workspace tests, fmt, clippy with warnings denied, and MSRV build.
- [x] Document actual supported scope and the Block API with a concrete example; preserve existing inspect behavior.
- [x] Review the final diff for ownership, mathematical semantics and unsupported capability handling. No automatic merge or push.

## Implementation record

- Graph and composition tests went from missing APIs to passing; shared-state execution tests initially failed with UnsupportedOp, then passed after reference dispatch was connected.
- Public binding role validation and quantized-state rejection were each reproduced with failing tests before their fixes.
- The independent fixture covers two layers, full and windowed grouped-query attention, partial rotary transforms, unequal key/value widths, norms, gated MLP and residual connections. All 32 input partitions are compared to its logits with absolute tolerance 2e-6.
- Memory tests cover one physical history shared across roles/layers, transforms deriving either K or V with explicitly counted full-history scratch, and a tied embedding/output weight. Error tests exercise failure after append, retry, reset, capacity, malformed weights and placement.
- Final independent review found two P2 issues: inconsistent binding axes and colliding serialized component keys. Both were reproduced RED, fixed, and verified GREEN. Binding axes now match their row domains; the global hidden-layers namespace is reserved.
- Scope remains the first synchronous single-sequence dense-f32 CPU path. Sliding-window reads retain full history; intermediate buffers are reserved independently rather than reused. Existing config-family adapters still require migration to computation graphs.
- Final validation: 102 workspace tests passed; `cargo fmt --all --check`, `git diff --check`, warnings-denied workspace/all-targets clippy and Rust 1.89 workspace/all-targets build passed. Fixture regeneration produced identical bytes. `dense_decode` ran successfully and reported 6096 bytes of reserved payload (1488 weights, 672 state, 3912 activations, 24 input).
