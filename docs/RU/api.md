# lumen-rs — чат-слой, CLI и сервер

Oct 27, 2026 · @Vemloir

Внешний интерфейс движка поверх `Engine` ([раздел 6 architecture.md](architecture.md#6-рантайм-ядро-генерации-и-спекулятивка)): чат-слой (`lumen-chat`), бинарь `lumen` и HTTP-сервер (`lumen-server`).

**Чат-слой** (`lumen-chat`)

- Токенизатор — `tokenizers`; он и шаблон берутся из заголовка `.ltq` (`tokenizer`, `chat_template`). Шаблон — через minijinja с `minijinja-contrib` (pycompat для Python-методов строк в HF-шаблонах), плюс `raise_exception` и `strftime_now`.
- kwargs шаблона: `chat-template.kwargs` из конфига, поверх — параметры запроса (`reasoning_effort` из OpenAI, `thinking` из Anthropic).
- Мультимодальные части сообщений (картинки, аудио, видео) проходят через `lumen-media`; шаблон вставляет столько токенов-заполнителей, сколько вернул препроцессор.
- Парсер вывода на семейство: Harmony у gpt-oss через `openai-harmony` (каналы analysis, commentary, final; вызовы `to=functions.*`), `<think>` и `<tool_call>` у Qwen. Работает по id спецтокенов, а не по тексту.
- Инкрементальная детокенизация со смещениями prefix и read: незаконченный UTF-8 не уходит наружу.

**CLI** (`lumen`)

- `run -c cfg.json -p "…"` и `chat`: цикл `Engine::step` в главном потоке, `stdout().lock()` и flush после каждого токена.
- `serve`: сервер из секции `server`.
- `inspect`: разрешённый план по слоям с источником каждого значения; память по устройствам и классам (веса, KV на токен, состояние на последовательность); байты по ресурсам на токен и оба потолка шага ([performance.md](performance.md)).
- `probe`: замер потолков железа → `hw-profile.json`.
- `bench`: pp и tg в токенах в секунду и η — по шагу, по ресурсам и по сегментам; `--trace` пишет временную шкалу сегментов в формате Chrome trace.
- `pack`: запись модели в итоговых укладках ([раздел 4 architecture.md](architecture.md#4-веса-дескрипторы-и-загрузка)).
- Спекулятивка: `--draft cfg.json`, `--spec-k 4`, `--ngram`, `--mtp`.

**Сервер** (`lumen-server`)

Секция `server` конфига задаёт `host`, `port` и `max-concurrent-requests` — сколько последовательностей движок ведёт одновременно; остальные ждут в очереди. Поток движка принимает `Command::{Submit, Cancel, Shutdown}` через `std::sync::mpsc`, а события отдаёт в `tokio::sync::mpsc::UnboundedSender` запроса — отправка из обычного потока безопасна. Хендлер axum превращает приёмник в SSE. Клиент отключился → приёмник уничтожен → `send` возвращает ошибку → движок отменяет последовательность и освобождает KV.

Оба API — адаптеры над одной внутренней моделью: на входе `ChatRequest`, на выходе поток `ChatEvent` (`ReasoningDelta`, `TextDelta`, `ToolCallStart`, `ToolArgsDelta`, `ToolCallEnd`, `Done`).

| API | Эндпоинты | Поток | Особенности |
| --- | --- | --- | --- |
| OpenAI | `POST /v1/chat/completions`, `GET /v1/models` | SSE `chat.completion.chunk`, финал `data: [DONE]`; usage при `stream_options.include_usage` | reasoning → `reasoning_content`; `tool_calls` с index и потоковыми `arguments`; роль `developer` = system; картинки — части `image_url` (data URL), аудио — `input_audio` |
| Anthropic | `POST /v1/messages`, `POST /v1/messages/count_tokens` | `message_start` → `content_block_start/delta/stop` → `message_delta` → `message_stop`, плюс `ping` | `system` отдельным полем, `max_tokens` обязателен; `thinking_delta`, `input_json_delta`; stop\_reason: end\_turn, max\_tokens, stop\_sequence, tool\_use; `budget_tokens` закрывает reasoning принудительно через хук сэмплера; картинки — блоки `image` с base64 |
