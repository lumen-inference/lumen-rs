# C engine and extensible backend design

The user requested documentation for a C engine with CPU and Vulkan execution and an easy way to add further backends. The canonical design is [RU/backends.md](../../RU/backends.md), mirrored in [EN/backends.md](../../EN/backends.md). It is a proposed source-level interface, not implemented API.

The design keeps model composition, operation semantics, placement and resource lifetimes common. A backend provides device discovery, memory and host transfers, support queries, prepared execution and completion. Static build registration is sufficient initially; no stable binary plugin ABI is promised. Optional fusion and device-to-device transfers do not replace the baseline operation and RAM-transfer paths.

The accompanying documentation adopts registered device names for storage and compute, and backend-owned settings under `backends.<name>`. This updates an unimplemented C configuration proposal; it does not change the current Rust parser. C++ dependency policy and the conditional VMA choice remain open.

A builder review identified cancellation after partial execution as requiring an explicit policy. The initial design preserves the session if cancellation precedes acceptance of the first job of the step. Once any work has been accepted, an accepted cancellation drains work and requires reset. A late cancellation does not override a successful terminal result. Transactional rollback is not promised.

The first implementation should prove the interface with a minimal backend that has limited operation support and controlled delayed completion, then CPU and Vulkan. Adding that test module must not change model definitions or introduce backend-name branches into scheduling. This task changes documentation only; production implementation and performance claims require separate validation.
