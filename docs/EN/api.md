# External Interfaces: Chat Layer, CLI, and HTTP Server

**Status: interface proposal.** Only `inspect` is operational, reporting resolved placement, access classifications, and rule provenance. Runtime memory reports, hardware ceilings, generation, chat, serving, probing, packing, and quantization are not implemented. The command and protocol descriptions below are planned behavior, not a ready public API.

The external interface layer on top of [`Engine`](architecture.md#8-generation-runtime-engine-speculative-decoding-and-multimodality) is divided into three crates:
- **`lumen-chat`** — formats conversational messages (text, images, audio, system prompts, and tool definitions) into tokens and parses generated token streams into typed events.
- **`lumen-cli`** — the `lumen` binary for running generation, inspecting placement plans, benchmarking hardware, and quantizing models.
- **`lumen-server`** — an `axum` HTTP server compatible with the **OpenAI API** and **Anthropic Messages API** protocols.

---

## 1. Chat Layer and Output Parsing (`lumen-chat`)

All metadata required for prompt formatting is stored directly inside the `.ltq` file header ([see container specification](ltq-format.md#6-row-slices-and-the-ltq-container)): the tokenizer vocabulary (`tokenizer`), chat template (`chat_template`), and media preprocessor configuration (`processor_config`).

1. **Tokenization and template rendering:**
   Tokenization is handled by the `tokenizers` crate. The chat template is rendered via `minijinja` with `minijinja-contrib` (`pycompat` for Python string methods used in HuggingFace templates), plus `raise_exception` and `strftime_now`.
2. **Template arguments (`kwargs`):**
   Default values are read from the config's `chat-template.kwargs` section and merged with per-request parameters (`reasoning_effort` in OpenAI API or `thinking` in Anthropic API).
3. **Multimodal messages:**
   Images, video, and audio are preprocessed by `lumen-media`; the template inserts the corresponding number of placeholder tokens into the token stream.
4. **Incremental detokenization and special block parsing:**
   - The detokenizer tracks prefix and read offsets, holding incomplete multi-byte UTF-8 sequences until the next token arrives.
   - Reasoning blocks and tool calls are parsed by **special token IDs** rather than text matching: `gpt-oss` models use `openai-harmony` (channels `analysis`, `commentary`, `final`, and `to=functions.*` calls), while Qwen models use the `<think>` and `<tool_call>` special tokens.

`lumen-chat` emits a unified stream of `ChatEvent` items:
- `ReasoningDelta(String)`
- `TextDelta(String)`
- `ToolCallStart`, `ToolArgsDelta`, `ToolCallEnd`
- `Done { reason, usage }`

---

## 2. Command-Line Interface (`lumen-cli`)

| Command | Purpose |
| :--- | :--- |
| **`lumen run`** | Single-prompt generation (`lumen run -c cfg.json -p "…"`). Runs the `Engine::step()` loop synchronously on the main thread, flushing `stdout` after each token. |
| **`lumen chat`** | Interactive multi-turn terminal chat preserving the KV prefix cache across turns. |
| **`lumen serve`** | Starts the HTTP server using parameters from the config's `server` section. |
| **`lumen inspect`** | Prints the resolved per-layer `PlacementPlan` with the JSON rule source for every value, memory usage by device and class (weights, KV per token, state per sequence), per-token resource byte counts, and both hardware step ceilings ([performance.md](performance.md)). `--model-config <config.json>` takes the architecture from a HuggingFace `config.json` instead of the `.ltq` file, so a placement can be inspected before the model is quantized. |
| **`lumen probe`** | Measures hardware bandwidths and latencies and saves the profile to `hw-profile.json`. |
| **`lumen bench`** | Measures prompt processing (`pp`) and token generation (`tg`) throughput in tokens/s along with hardware efficiency $\eta$ per step, resource, and segment; `--trace` writes a Chrome Trace timeline. |
| **`lumen pack`** | Repacks `.ltq` model weights into target backend layouts (`CpuBand16` / `GpuWarp`) on disk ([Section 6 of architecture.md](architecture.md#6-weight-representations-and-model-loading)). |
| **`lumen quantize`** | Quantizes a BF16 (`safetensors`) checkpoint into a `.ltq` container. |

Speculative decoding flags: `--draft cfg.json`, `--spec-k 4`, `--ngram`, `--mtp`.

---

## 3. HTTP Server (`lumen-server`)

The `server` configuration section defines `host`, `port`, and `max-concurrent-requests` (the maximum number of sequences processed concurrently by `Engine`; additional requests wait in a queue).

The `Engine` thread receives `Command::{Submit, Cancel, Shutdown}` messages over a `std::sync::mpsc` channel and sends generated events to a per-request `tokio::sync::mpsc::UnboundedSender`. The `axum` handler wraps the receiver into a Server-Sent Events (SSE) stream. If a client disconnects, the receiver is dropped, `send` returns an error, and `Engine` cancels the sequence and frees its KV and state allocations.

| API | Endpoints | SSE Stream Format | Implementation Details |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `POST /v1/chat/completions`<br/>`GET /v1/models` | `chat.completion.chunk` chunks, terminated by `data: [DONE]`; `usage` included when `stream_options.include_usage` is set | Reasoning mapped to `reasoning_content`; tool calls streamed in `tool_calls` with index and streaming `arguments`; `developer` role mapped to `system`; images passed via `image_url` (data URL), audio via `input_audio`. |
| **Anthropic** | `POST /v1/messages`<br/>`POST /v1/messages/count_tokens` | `message_start` → `content_block_start/delta/stop` → `message_delta` → `message_stop`, plus `ping` | `system` passed as a top-level field, `max_tokens` required; reasoning emitted as `thinking_delta`, tool arguments as `input_json_delta`; stop reasons: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`; `budget_tokens` closes reasoning via a sampler hook; images passed as `base64` `image` blocks. |
