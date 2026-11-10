# Velri external interfaces

[Русский](../RU/api.md)

This is the proposed **Velri engine API**. Computation resources and execution live in Velri-lib behind it; token IDs, Model and Session are engine concepts. Program wraps a library execution plan and Context uses a library device context. Velri-lib does not depend on these engine objects. See the [project boundary](velri-lib.md).

Terms used in this document are explained in the [glossary](glossary.md).

**Status: proposed C API and applications.** Public C headers do not exist yet. The current Rust CLI implements `inspect`; generation, serving and quantization commands are not ready. This document describes how the future interfaces should work. Mentioning a function here does not mean it can already be called.

## How an application uses the library

An application will be able to use the compute engine directly, without a CLI, HTTP server, chat template or tokenizer. The minimal path accepts token IDs, executes a computation step and returns logits or selected tokens, depending on the mode.

Long-lived objects use opaque types. Applications create and access them through the API; the library controls their internal representation. Each object has its own role and lifetime requirements:

| Object | Responsibility and lifetime |
| :--- | :--- |
| Context | Holds devices, the worker pool and platform resources. Must outlive all dependent objects. |
| Model | Holds configuration, weights and prepared representations. Several sessions may share one model. |
| Program | Holds a checked execution plan for a particular model, context and set of limits. |
| Session | Holds mutable sequence state and reserved workspace. |

Every creation operation has a matching release operation. Attempting to destroy an object with live dependents must return a useful error. For example, a context cannot be freed while a session still needs it, leaving the session with pointers to released memory.

Exact function names and public descriptor sizes will be fixed with the first working implementation. Internal library structures do not carry a stable ABI promise.

Applications enumerate and select devices through the common context API. Backend registrations and function tables belong to the internal [backend interface](backends.md). Adding a module to the build does not change the model-step API. Public headers expose neither dependency types, Vulkan types nor other device API types; compatibility with external binary plugins is not promised yet.

## Buffers and concurrent calls

The API specifies ownership for every buffer passed across it. This tells the application when it may modify or release memory.

The first `step` will be synchronous. Borrowed input remains valid until the call returns with a ready result. Borrowed session output remains valid until the next operation that changes it or until the session is destroyed. An application makes a copy to retain the output longer.

Backends may execute asynchronously inside the synchronous call, using completion objects. Before returning to the application, `step` must wait for accepted work and any required output transfers.

A future asynchronous API would need its own completion object and buffer-retention rules. The synchronous `step` rules cannot simply be applied to it.

Only one step may run in a session at a time. Different sessions have independent state, but their shared context scheduler must coordinate concurrent execution. Headers will identify which functions may be called concurrently. Creating several sessions does not itself promise higher throughput from running them together.

## What happens on an error

Functions return execution statuses. Causes include invalid arguments, unsupported operations, memory or capacity exhaustion, I/O errors and device errors. Diagnostics preserve the cause and context so the application can explain the failure. The library does not terminate the process or use `longjmp` through caller code.

Sizes and offsets are checked before access, including multiplication overflow. Invalid input arguments must be detected before a step starts.

A failure during an already-started step needs different handling. If state has changed, the session becomes unusable until an explicit `reset`. Automatic `rollback` is not promised until implemented. Partial output cannot be reported as a successful result: its producers must complete and the numerical requirements must be checked first.

## How cancellation works

A separate cancellation request may be available during a step. Requesting cancellation does not mean all computation has stopped. Memory cannot be freed while an outstanding CPU or GPU job still accesses it.

Completion or reset must wait for every user of the relevant resources. Applications must distinguish three outcomes: normal `stop`, requested cancellation and failure.

Cancellation accepted before the backend accepts the step's first job leaves the previous session state intact. Once work is accepted, the initial API requires `reset` after cancellation completes. Continuing from the last confirmed token without resetting is not promised; that would need separate state-rollback rules. The runtime commits one terminal outcome, so cancellation after successful completion does not revoke a ready result.

Device loss invokes the backend's shutdown procedure. Affected resources cannot be reused until safe release is established.

## Text and chat above computation

Tokenizers and chat templates will be adapters above the compute core. Verify compatibility using exact token IDs, special tokens, Unicode handling, byte fallback and streaming decoding. Check conversation formatting separately from tokenization: correct token splitting does not prove that messages are formatted as the model expects.

An unsupported chat template must not silently be replaced with an approximation. Such a replacement requires an explicit user choice.

Model metadata determines message handling and output special-token processing. Applications receive text, reasoning and tool-call events. Concrete HTTP protocols remain outside mathematical blocks. Candidate libraries and unresolved choices are listed in [dependencies.md](dependencies.md).

## CLI and HTTP server

The command-line applications are planned to provide:

| Command | Purpose |
| :--- | :--- |
| `inspect` | Show requested and resolved placement, registered backends, actual devices, applied settings, capabilities and available memory calculations. |
| `run`, `chat` | Run generation or interactive chat through the library. |
| `probe`, `bench` | Measure hardware, kernels and complete steps; retain parameters and timelines. |
| `pack` | Perform verified weight repacking with an explicit destination-layout version. |
| `quantize` | Run a separate calibration and LTQ export tool. |
| `serve` | Run an optional HTTP/SSE application. |

Distinguish the working Rust `inspect` from the planned C applications. This table describes the target commands and their responsibilities.

The HTTP layer translates requests into runtime commands and runtime events into client responses. Compatibility with selected external API versions is planned. Adapter tests must define the supported fields: “compatible” does not mean every feature of the external service is implemented.

Request and output-event queues must be bounded. When a client reads slowly, the server applies backpressure or cancels with an explanation. It must not accumulate ready tokens without a memory limit.

A disconnected client requests cancellation. The runtime then waits for outstanding work and releases state. The server library does not control GPU-buffer lifetimes. CivetWeb remains a candidate until this behavior is verified.
