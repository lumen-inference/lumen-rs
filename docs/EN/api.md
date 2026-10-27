# lumen-rs — Chat Layer, CLI, and Server

Oct 27, 2026 · @Vemloir

The engine's external interface on top of `Engine` ([section 6 of architecture.md](architecture.md#6-runtime-generation-core-and-speculative-decoding)): the chat layer (`lumen-chat`), the `lumen` binary and the HTTP server (`lumen-server`).

**Chat layer** (`lumen-chat`)

- Tokenizer — `tokenizers`; it and the template come from the `.ltq` header (`tokenizer`, `chat_template`). Template — via minijinja with `minijinja-contrib` (pycompat for Python string methods in HF templates), plus `raise_exception` and `strftime_now`.
- Template kwargs: `chat-template.kwargs` from the config, overlaid by request parameters (`reasoning_effort` from OpenAI, `thinking` from Anthropic).
- Multimodal message parts (images, audio, video) go through `lumen-media`; the template inserts as many placeholder tokens as the preprocessor returned.
- One output parser per family: Harmony for gpt-oss via `openai-harmony` (analysis, commentary, final channels; `to=functions.*` calls), `<think>` and `<tool_call>` for Qwen. It works on special-token ids, not text.
- Incremental detokenization with prefix and read offsets: incomplete UTF-8 never leaks out.

**CLI** (`lumen`)

- `run -c cfg.json -p "…"` and `chat`: the `Engine::step` loop on the main thread, `stdout().lock()` and a flush after every token.
- `serve`: the server from the `server` section.
- `inspect`: the resolved plan per layer with the source of every value; memory per device and class (weights, KV per token, state per sequence); bytes per resource per token and both step ceilings ([performance.md](performance.md)).
- `probe`: measures the hardware ceilings → `hw-profile.json`.
- `bench`: pp and tg in tokens per second and η — per step, per resource, and per segment; `--trace` writes a segment timeline in Chrome trace format.
- `pack`: writes the model in its final layouts ([section 4 of architecture.md](architecture.md#4-weights-descriptors-and-loading)).
- Speculative decoding: `--draft cfg.json`, `--spec-k 4`, `--ngram`, `--mtp`.

**Server** (`lumen-server`)

The `server` config section sets `host`, `port`, and `max-concurrent-requests` — how many sequences the engine runs at once; the rest wait in a queue. The engine thread receives `Command::{Submit, Cancel, Shutdown}` via `std::sync::mpsc` and sends events to the request's `tokio::sync::mpsc::UnboundedSender` — sending from a plain thread is safe. The axum handler turns the receiver into SSE. Client disconnects → receiver dropped → `send` returns an error → the engine cancels the sequence and frees its KV.

Both APIs are adapters over one internal model: `ChatRequest` in, a stream of `ChatEvent` out (`ReasoningDelta`, `TextDelta`, `ToolCallStart`, `ToolArgsDelta`, `ToolCallEnd`, `Done`).

| API | Endpoints | Stream | Notes |
| --- | --- | --- | --- |
| OpenAI | `POST /v1/chat/completions`, `GET /v1/models` | SSE `chat.completion.chunk`, final `data: [DONE]`; usage with `stream_options.include_usage` | reasoning → `reasoning_content`; `tool_calls` with index and streamed `arguments`; `developer` role = system; images as `image_url` parts (data URL), audio as `input_audio` |
| Anthropic | `POST /v1/messages`, `POST /v1/messages/count_tokens` | `message_start` → `content_block_start/delta/stop` → `message_delta` → `message_stop`, plus `ping` | separate `system` field, `max_tokens` required; `thinking_delta`, `input_json_delta`; stop\_reason: end\_turn, max\_tokens, stop\_sequence, tool\_use; `budget_tokens` force-closes reasoning via a sampler hook; images as base64 `image` blocks |
