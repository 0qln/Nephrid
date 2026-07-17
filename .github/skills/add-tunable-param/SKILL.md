---
name: add-tunable-param
description: >-
    Skill for adding a new tunable parameter to the nephrid engine. Covers the
    full pipeline: trait method, TunableParams struct field, config option,
    ConfigBuilder seeder, accessor, set() match arm, print_uci(), const impls
    for all param types, and optionally the tuning/config.json entry.
user-invocable: true
---

# Adding a new tunable parameter to nephrid

This skill documents the exact, complete set of changes required to add a new
tunable engine parameter. Every change site is listed; skipping any one of them
will cause a compile error or leave the parameter untunable.

## Overview of the pipeline

```
trait (chrono.rs / quiesce/mod.rs / ...)
  → TunableParams struct field  (params.rs)
  → TunableParams ChronoParams/QSearchParams/… impl  (params.rs)
  → from_config() read  (params.rs)
  → from_config() struct-init field  (params.rs)
  → IConfigBuilder::build_config  (already calls builder.chrono/qsearch/…)
  → Configuration struct field  (config.rs)
  → Configuration::builder() default  (config.rs)
  → ConfigBuilder::<group>() seeder  (config.rs)
  → Configuration accessor fn  (config.rs)
  → Configuration::set() match arm  (config.rs)
  → Configuration::print_uci()  (config.rs)
  → all const C_*Params impls of the trait  (params.rs)
  → tuning/src/params/config.json entry  (optional, for SPSA tuning)
```

## File map

| File | Role |
|---|---|
| `engine/src/core/chrono.rs` | `ChronoParams` trait |
| `engine/src/core/search/quiesce/mod.rs` | `QSearchParams` trait |
| `engine/src/core/search/id/mod.rs` | `IdParams`, `ScorerParams` traits |
| `engine/src/core/params.rs` | `TunableParams` struct + all impls |
| `engine/src/core/config.rs` | UCI option definitions, builder, accessors |
| `tuning/src/params/config.json` | SPSA tuning bounds |

## Step-by-step

### 1. Add the method to the correct trait

Open the trait file (e.g. `QSearchParams` in `quiesce/mod.rs`).

**If the parameter has a sensible default that non-tunable impls can use**, keep
the default. If it must always be supplied (like the existing tunable params),
remove the default so the compiler enforces all impls.

```rust
pub const trait QSearchParams {
    // existing methods …
    fn my_new_param(&self) -> AnyScore;          // no default → must impl everywhere
    // OR
    fn my_new_param(&self) -> AnyScore { AnyScore::new(42) }  // default → only TunableParams needs override
}
```

### 2. Add the field to `TunableParams` (params.rs)

Follow the naming convention `<group>_<param>`, e.g.:
- `hce_q_*` for q-search params
- `timeman_*` for chrono params
- `id_*` for iterative-deepening params

```rust
pub struct TunableParams<Base> {
    // … existing fields …
    hce_q_my_new_param: AnyScore,
    // …
}
```

### 3. Implement the trait method on `TunableParams` (params.rs)

Find the `impl<B, X: Deref<Target = TunableParams<B>>> QSearchParams for X`
block and add:

```rust
fn my_new_param(&self) -> AnyScore { self.hce_q_my_new_param }
```

### 4. Read from config in `from_config()` (params.rs)

Inside `TunableParams::from_config()`, add a local binding using the
corresponding `Configuration` accessor (which you will add in step 7):

```rust
let hce_q_my_new_param = config.eval_my_new_param();
```

Then include it in the `Self { … }` struct literal:

```rust
Self {
    // … existing fields …
    hce_q_my_new_param,
    // …
}
```

### 5. Add a `ConfigOption` field to `Configuration` (config.rs)

Place it near its siblings (all chrono options together, all q-search options
together, etc.):

```rust
/// Brief description of what my_new_param controls.
eval_my_new_param: ConfigOption<Spin<UciInteger>>,   // or UciPercent for f32 ratios
```

Use `UciInteger` when the value is an `i32` / `AnyScore` / `Depth` / count.
Use `UciPercent` when the value is an `f32` ratio (stored as a percentage
integer, retrieved with `.get::<ratio>()`).

### 6. Add the default in `Configuration::builder()` (config.rs)

```rust
eval_my_new_param: ConfigOption::new(
    "eval-my-new-param",
    Spin::new(42, -200, 200),      // (default, min, max) – use Spin::<UciPercent> for ratio params
),
```

The name string becomes the UCI option name and must use kebab-case.

### 7. Seed the value in the `ConfigBuilder` group method (config.rs)

Find the `ConfigBuilder::qsearch()` (or `chrono()`, `id()`, etc.) method and
add:

```rust
cfg.eval_my_new_param.seed(params.my_new_param().v());
// For UciPercent fields: cfg.timeman_my_param.seed(Ratio::new::<ratio>(params.my_new_param()));
```

### 8. Add the public accessor (config.rs)

```rust
pub fn eval_my_new_param(&self) -> AnyScore { AnyScore::new(self.eval_my_new_param.value) }
// For UciPercent: pub fn timeman_my_param(&self) -> f32 { self.timeman_my_param.value.get::<ratio>() }
```

### 9. Add the `set()` match arm (config.rs)

```rust
#[cfg(feature = "tunable")] "eval-my-new-param" => self.eval_my_new_param.set(value),
```

The string must exactly match the UCI option name from step 6.

### 10. Add to `print_uci()` (config.rs)

Inside the `if cfg!(feature = "tunable")` block:

```rust
println!("{}", self.eval_my_new_param);
```

### 11. Implement the trait on every `C_*Params` type (params.rs)

Every const params type that implements the trait must get the new method.
Find all `impl const QSearchParams for C_*` blocks and add your method:

```rust
fn my_new_param(&self) -> AnyScore { AnyScore::new(42) }
```

The types currently implementing each trait:

| Trait | Implementors |
|---|---|
| `ChronoParams` | `C_MctsHceParams`, `C_MctsNnParams`, `C_MctsPureParams`, `C_IdHceParams`, `C_IdNnueParams` |
| `QSearchParams` | `C_MctsHceParams`, `C_IdHceParams`, `C_IdNnueParams` |
| `IdParams` | `C_IdHceParams`, `C_IdNnueParams` |
| `ScorerParams` | `C_IdHceParams`, `C_IdNnueParams` |

If a type uses `todo!()` for a value it can't yet provide a real default for,
that is acceptable while the feature is under development, but it will panic at
runtime in that configuration.

### 12. Add to `tuning/src/params/config.json` (optional)

Only needed if you want the parameter included in SPSA runs:

```json
"eval-my-new-param": {
    "value": 42,
    "min_value": -100,
    "max_value": 100,
    "step": 5
}
```

The `value` should match your default from step 6. `step` should be roughly
5–15% of the total range for efficient SPSA convergence.

## Naming conventions

| Kind | Field prefix | UCI name prefix | Example |
|---|---|---|---|
| Q-search (integer score) | `hce_q_` | `eval-` | `hce_q_futility_margin` / `eval-futility-margin` |
| Q-search (tapervalue) | `hce_q_` | `eval-` | `hce_q_delta_pruning_threshold` |
| Time management (f32 ratio) | `timeman_` | `timeman-` | `timeman_stability_slope` |
| ID search | `id_` | `id-` | `id_nmp_reduction` |
| MCTS | `mcts_` | `mcts-` | `mcts_killer_exploitation` |
| Selection | `select_` | `select-` | `select_cpuct` |

## Validation

Run `bin/check` after making all changes. It runs clippy across every feature
combination. A missing impl on any `C_*Params` type will surface as a compile
error in one of the feature-gated crates.
