# C/Vulkan engine configuration

[Русский](../RU/config.md)

Terms used in this document are explained in the [glossary](glossary.md).

**Status: proposed new schema.** The working Rust parser and existing JSON examples are described in [rust-config.md](rust-config.md). The current CLI does not accept the example on this page yet. Moving configuration to the new schema is a separate task.

The new schema retains JSON and three separate fields: `store`, `compute` and `prefill-compute`. Configuration is read once. After validation, the engine receives a placement plan for data and computations that it can use during model execution.

Compute devices are resolved through the [backend registry](backends.md). The parser reads `<backend>:<index>` and checks the name against that registry. Adding a module does not add its name to a separate `switch` in the parser or planner. Each module's settings live under `backends.<name>` and are checked by its handler when creating the context.

Recognizing a key does not mean a backend can execute the requested operation. Parsing JSON and checking device capabilities are therefore separate preparation tasks.

## Where to store data and run computations

`store` selects persistent storage. `compute` selects decode execution. `prefill-compute` may separately select prefill execution; when omitted, it uses `compute`.

The placement fields provide these values:

| Value | Meaning in the new schema |
| :--- | :--- |
| `cpu:0` | CPU execution context. |
| `vulkan:0` | In `compute`, execution on the selected Vulkan device; in `store`, storage managed by that device. |
| `ram:0` | Host-memory storage. It does not promise placement on a particular NUMA node. |
| `disk` | File storage. Loading the required data must be explicitly prepared. |

A device index is resolved to an actual device at startup. `inspect` must show its name and identifier so the user can check where work was placed. The schema does not assume an NVIDIA GPU.

The index belongs to the named backend's devices and does not promise stable ordering across runs. `ram:0` and `disk` identify stores and cannot be selected as compute backends. A device must separately confirm support for its selected storage or execution role. On UMA, memory managed by the GPU backend may occupy shared physical RAM.

The earlier proposed name `device:0` is no longer used: with several backends, it would be unclear which device it names. The new example explicitly places storage on `vulkan:0`. This changes the proposed C configuration; the existing Rust parser remains documented separately.

Old `cuda:N` values must not silently become `vulkan:N`. Diagnostics should explain configuration migration, after which the selected device's capabilities must be checked again.

Separate storage and execution locations require a supported access or copying strategy and a compatible data representation. For example, settings requesting RAM storage and GPU execution do not implement the transfer between them. Preparation must verify that the path exists. The common RAM route connects backends; changing a layout requires an explicit supported conversion.

## How settings inherit through the model tree

Configuration retains the model's component tree. This lets users set a general placement at the top and override it for particular components. Only three placement fields inherit downward: `store`, `compute` and `prefill-compute`.

`backends.<name>` settings do not inherit through the model tree. The module validates them once when creating the context and converts them into internal parameters; kernels do not receive model JSON.

For a component of a particular layer, the deepest applicable node supplies the value. A layer-selector override takes precedence over a rule in the general layer subtree. Overlapping selectors are rejected, so JSON key order cannot affect the result.

The particular model description supplies valid component paths. Not every attention block must have `q`, `k`, `v` and two caches: different models can organize attention differently.

A rule targeting a known component absent from this model is reported as inapplicable. A misspelled key is an error. Out-of-range layer selectors must follow an explicit strictness policy rather than accidental parser behavior.

## Example for a model with experts

The following JSON illustrates the target schema:

```json
{
  "model": {
    "path": "model.vtq",
    "store": "vulkan:0",
    "compute": "vulkan:0",
    "hidden-layers": {
      "moe": {
        "routed-experts": {
          "store": "ram:0",
          "compute": "cpu:0",
          "prefill-compute": "vulkan:0"
        }
      }
    }
  },
  "backends": {
    "cpu": {
      "threads": 0,
      "reserve-cores": 1,
      "affinity": "auto"
    }
  }
}
```

At model level, storage and execution are requested on `vulkan:0`. Routed experts override those settings: their data resides in `ram:0`, decode runs on `cpu:0` and prefill on `vulkan:0`. Data access and required copies still need validation. `backends.cpu` configures the CPU module; other backends have their own sections under the same rule.

`threads: 0` asks the engine to choose a worker count automatically from OS resources available to the process, accounting for the reserve. A positive value requests that number of workers. Available logical CPUs must not be treated as physical cores.

If required affinity cannot be established, preparation must fail. `affinity: "auto"` permits explained fallback to unbound execution; `"none"` disables binding.

## Limits and plan validation

Preparation parameters include context, step-token, sequence and memory limits. Their exact JSON names and defaults will be fixed with the C API. Old experimental settings are not all carried over automatically.

After resolving placement, preparation checks concrete operations, shapes, dtypes, strides, layouts, memory types and available capacities. An unknown operation or missing GPU capability must produce a useful error. A backend may support only some operations, but a prepared whole-model plan must have implementations for every operation it needs.

Diagnostics distinguish an unknown backend name, a module disabled at build time and the absence of a suitable device. Misspelled module-specific settings are not ignored either. Understanding a settings schema still does not establish support for every operation.

Explicitly pinning an unsupported operation to a device fails preparation. A permitted placement mode may choose another device, including CPU, or a declared equivalent decomposition. The report shows the actual plan, reason, transfers and their costs. An unsupported operation cannot switch to CPU midway through a step. Exact names and defaults for automatic-placement modes remain to be fixed in the new schema.

`inspect` must separately show user requests, resolved placement rules and verified device capabilities. Every value shows its origin: directly set on a component, inherited, or obtained while resolving configuration.

Memory accounting follows storage owners and includes padding, repacking and scratch. If a calculation is not implemented, its field is marked unavailable. A conditional estimate must not appear to be a verified value.
