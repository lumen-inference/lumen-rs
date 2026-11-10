# CPU/Vulkan preparation and execution

[Русский](../RU/execution-design.md)

**Project ownership:** the computation contracts and implementations on this page belong to **Velri-lib**. Velri uses them through the library API; it owns model and request policy. See the [project boundary](velri-lib.md).

Terms used in this document are explained in the [glossary](glossary.md).

**Status: execution design.** This document refines the [architecture](architecture.md). It does not describe an implemented Vulkan runtime. Before accepting an optimization, verify computation results, resource lifetimes and complete execution latency.

## The common executor and connected backends

The runtime works with prepared jobs and their dependencies. It does not choose execution by the name CPU, Vulkan or a future backend. The device function table handles submission and completion; its contract is described in [backends.md](backends.md).

The common plan determines placement, data lifetimes and device transfers. A backend prepares its kernels and implements required dependencies within its device. The baseline cross-backend route uses bounded RAM buffers. Direct transfers and shared mappings are optional accelerations with separately checked support.

Data transfers, layout changes and state movement must be planned before execution. An operation's failure during a step does not authorize a silent switch to CPU. When settings permit a mixed plan, preparation reports all its devices and transfers.

## What the engine prepares before the first step

Before final allocation, preparation queries permitted devices for support of concrete operations. Within its preparation budget, it chooses implementations, placement, layouts, explicit conversions and transfers. A backend may support some operations, but the whole-model plan must cover all of them.

Some computation properties do not change between steps. For those graph regions, preparation retains selected kernel functions, constant tensor dimensions, weight layouts, worker ranges, offsets and scratch requirements. Execution does not need to make those decisions again.

Immutable weights are packed once. An unnecessary source representation may be released after preparation, but release must be explicit. If both source and prepared representations remain necessary, both count toward the memory budget.

Prepared weights are tied to the source owner and data version, device, implementation and packing format. A representation can be shared only when these identities are compatible. Mutable parameters, commands, completion records and workspace are reserved for each session or concurrently active execution slot.

Sequence lengths, batch sizes and MoE routes may change. The plan supplies templates with fixed limits; only necessary parameters are filled before a step. Looking up JSON fields, revalidating constant descriptors and rediscovering all geometry in every worker are not kernel responsibilities.

The CPU pool receives prepared ranges. Measure publication, worker startup, arithmetic and completion separately. Several operations may share a job if dependencies are preserved. A block boundary does not itself require another barrier.

## The first shared GPU/CPU path

The general route is device A → agreed RAM representation → device B. Each backend implements its own host transfers, so a separate adapter for every backend pair is unnecessary. Byte copying does not change a layout: preparation must declare supported packing or unpacking, rejecting placement if no such path exists. Pinned staging, overlapping transfers and actual allocation overhead count toward the memory budget. Direct transfers and external-memory import require their own compatibility checks.

The first hybrid path uses CPU-routed experts. Some computation remains on the GPU, so the point at which data becomes ready for its next consumer must be explicit:

1. The GPU computes routes and required inputs, then copies them into host-visible staging memory.
2. The CPU waits for confirmed completion of GPU writes and establishes their visibility. Only then do CPU workers compute the selected experts.
3. The CPU completes its writes and performs `flush` if necessary. Results transfer to the GPU; the dependent `dispatch` must wait for that transfer.
4. Reuse a staging slot only after its last consumer completes.

Slot management must identify the particular outstanding execution using it. A layer number alone does not identify that work.

The initial implementation can use fences for host waits, binary semaphores for cross-queue dependencies and required memory barriers. Timeline semaphores are an optional route, not a first-backend requirement. Use them only when the Vulkan 1.2 or `VK_KHR_timeline_semaphore` capability is checked and enabled. See the [timeline semaphore specification](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_timeline_semaphore.html).

Memory barriers must describe actual reads and writes of the dependent operations. Account for ownership when a resource crosses queue families.

A separate transfer queue does not guarantee that copies overlap computation. Verify this on the device. Submitting GPU work and waiting for the entire queue after every operation cannot replace the dependency plan.

Device-local dependencies use the backend's synchronization facilities; common completion objects track the required groups of work. A separate CPU fence wait after every operation is unnecessary. A cross-backend consumer waits for the producer and required transfers. Direct synchronization may remove a CPU wait only after support is checked.

## CPU and GPU access to mapped memory

`HOST_VISIBLE` permits mapping, but does not guarantee the fastest CPU or GPU access. That property alone is insufficient to choose a transfer path.

Noncoherent memory requires correct `flush` and `invalidate` ranges respecting `nonCoherentAtomSize`. Coherent memory removes the corresponding cache-management operations. Completion waits and correctly defined dependencies are still necessary. Neither a C atomic nor `volatile` replaces the Vulkan protocol. The [Vulkan memory specification](https://docs.vulkan.org/spec/latest/chapters/memory.html) defines the details.

UMA can sometimes avoid a copy. Choose that path using memory properties and measurements. Shared physical memory alone does not prove the chosen access method is faster.

A discrete GPU may also read host-visible memory directly instead of copying into device-local memory. This is a separate candidate requiring verification. Sparse computation does not itself make that candidate preferable.

## Which commands can be reused

Prepare compute pipelines and constant descriptors in advance. Reuse command buffers only with compatible resource bindings and parameters. Mutable data and slots still have their own lifetimes: a ready command does not permit overwriting memory another job is using.

Vulkan command reuse is not CUDA Graphs and does not promise identical command-update capabilities.

## Step completion, cancellation and errors

The first public step will be synchronous, though internal execution may be asynchronous. Both output and session state must be ready when the result is returned. The device must no longer need borrowed memory used by the call.

Acceptance and successful completion are different events. If an error is discovered after a partial launch, the backend still returns a completion object for accepted work. The runtime retains its resources while they are used. Rejection before acceptance means nothing was launched and no borrowed data is retained.

A completed object reports final status and the end of those borrows. It does not itself make GPU data accessible to CPU code: required transfers and visibility operations remain explicit. `poll` checks status without waiting; `wait` targets the relevant work instead of stopping all devices. Objects and slots remain live until their last user finishes.

Cancellation stops future submissions and waits for already-submitted work. Only then can its memory be released. Cancellation does not authorize immediately freeing a buffer that CPU or GPU work still accesses.

Cancellation before the first job of a step is accepted preserves the previous session state. In the initial implementation, cancellation after acceptance requires an explicit `reset` before continuing; a partial step is not rolled back automatically. The runtime commits one outcome: late cancellation after success does not revoke a ready result.

Device loss and CPU failure put a session into a defined error state. Partially written state does not become a successfully produced token. Further session handling follows the [API contract](api.md).

Device loss invokes the backend's shutdown and release procedure. Resources are not reused until safe release is established.

## Disk and sequence state

Disk tables use a bounded staging region. Loaded data has an explicit first consumer so disk reads are connected to subsequent computation.

Basic reads and file mappings are portable across operating systems through `platform/`. Specialized asynchronous I/O is added based on measurements. Account for the file cache when measuring disk access: a cache hit is not a cold NVMe read.

Sliding windows, speculative-token rollback and shared prefixes need explicit page-ownership and state-commit rules. A correct sequential session is sufficient until those rules are implemented. Before enabling each capability, budget its required history reserves, snapshots and temporary data.

## Checking further optimizations

Fusion, overlap, partition changes and new waiting protocols need a reproducible baseline. Compare complete p50 and p99 latency, memory use and cancellation correctness.

A high share of CPU samples in waiting does not imply that the same share of total latency can be removed. Some waits follow necessary dependencies. Similarly, machine-code observations do not replace measurement of an actual speedup.
