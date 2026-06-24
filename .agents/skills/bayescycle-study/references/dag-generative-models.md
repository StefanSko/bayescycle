# DAGs as generative-model scaffolding

DAG reasoning is a required lens for causal bayescycle studies. It is mainly
used in the `generative_model` phase and then consumed by `estimator_plan`,
`simulation`, and `critique`.

A DAG is not the full generative model. It gives the causal and information-flow
structure. The full generative model still needs distributions, priors,
measurement error, missingness/censoring, and observation status.

Use the rule:

```text
DAG first, probability model second.
```

## Why draw the picture?

When the question is whether `X` causes `Y`, a regression coefficient for `X`
usually mixes the causal effect with non-causal associations that flow through
other variables. Before choosing controls, write down what is believed to cause
what.

A directed acyclic graph has:

- nodes for variables
- arrows from cause to effect
- no directed loops

Useful intuition: arrows are the direction of listening. If `Z -> X`, then `X`
listens to `Z` and carries information about it. Associations arise when
variables carry information about shared causes or selected consequences.
Causal inference is bookkeeping about which information paths should remain open
for the estimand and which should be closed.

## Four building blocks

Any larger DAG is assembled from four local patterns.

### Fork / common cause

```text
X <- Z -> Y
```

`Z` causes both `X` and `Y`. `X` and `Y` become associated even if neither causes
the other. This is the classic confound.

For a total causal effect of `X` on `Y`, conditioning on `Z` closes this
non-causal path.

### Pipe / mediator

```text
X -> Z -> Y
```

`X` affects `Y` through `Z`. This is a causal path. Conditioning on `Z` blocks
that path.

For a total effect, do not condition on the mediator. For a controlled or direct
effect, conditioning or intervention on the mediator may be part of the estimand,
but that must be stated explicitly.

### Collider

```text
X -> Z <- Y
```

`X` and `Y` both cause `Z`. Left alone, this path is closed. Conditioning on `Z`
opens the path and can create a spurious association.

Intuition: suppose a movie is funded if it is high quality or has a famous star.
Among funded movies, learning there is no famous star suggests high quality.
Quality and fame look related only because you conditioned on funding.

### Descendant / proxy

```text
Z -> A
```

`A` is a descendant or proxy of `Z`. Conditioning on `A` partially conditions on
`Z`. This can help when `Z` is a confounder and `A` is a good proxy, but it can
also hurt when `A` is a descendant of a collider or mediator.

Descendants inherit the dangers of the node they proxy, usually more weakly and
with measurement uncertainty.

## Backdoor workflow

For a causal estimand involving exposure/treatment `X` and outcome `Y`:

1. List every path between `X` and `Y`, ignoring arrow direction while listing.
2. Separate paths that start with an arrow out of `X` from paths that enter `X`.
   - Paths starting out of `X` are causal paths relevant to the effect.
   - Paths entering `X` are backdoor paths and can create non-causal association.
3. Find an adjustment set that closes every open backdoor path while preserving
   the causal paths required by the estimand.
4. Avoid conditioning on colliders or descendants of colliders.
5. Avoid conditioning on mediators unless the estimand is a direct/controlled
   effect rather than a total effect.

If such a set exists, the causal effect is identified under the DAG assumptions.
This does not guarantee good finite-sample estimation, good priors, good
measurement, or successful computation.

If no such set exists, record that the effect is not identified from the proposed
data and assumptions. That is a useful study result.

## Warnings

### Do not control for everything

Controls are interventions on information flow. Some close biasing paths; others
open biasing paths or block the effect of interest. The DAG is how the agent
separates good controls from bad controls.

### Table 2 fallacy

A model built to estimate the effect of `X` generally does not make the
coefficients for control variables valid causal effects. Controls deconfound the
chosen estimand; they are not automatically deconfounded themselves.

When reporting a causal model, clearly distinguish:

- the target estimand
- variables used only for adjustment
- variables whose coefficients should not be causally interpreted

## Required generative-model artifact sections

For causal questions, the `generative_model` phase should produce or propose an
artifact with these sections:

```markdown
## DAG purpose

## Exposure/treatment X and outcome Y

## Estimand type
- total effect
- direct/controlled effect
- mediated effect
- descriptive/predictive quantity instead of causal effect

## Variables
- observed
- latent/unobserved
- proxies/descendants
- missing/censored/future

## DAG sketch

## Paths from X to Y

## Backdoor paths

## Forks / pipes / colliders / descendants

## Candidate adjustment sets

## Bad controls to avoid

## Unmeasured confounding risks

## What the DAG does not specify yet
```

## Required estimator-plan sections

The `estimator_plan` phase should consume the DAG artifact and state:

```markdown
## Chosen adjustment set

## Backdoor paths closed

## Causal paths intentionally left open

## Mediators/colliders/proxies intentionally excluded

## Table 2 warning for this model

## Why this estimator targets the approved estimand under the DAG
```

## Simulation use

Use the DAG as the skeleton for simulation:

```text
simulate background/common causes
simulate exposure/treatment
simulate mediators and outcome
simulate measurement, censoring, missingness, and selection
compute the known estimand
run the estimator
compare estimated and known estimand
```

Recovery checks should fail loudly if the proposed estimator does not recover the
causal estimand under the assumed DAG.

## Critique use

During critique or revision, revisit the DAG:

- Could an unmeasured confounder explain the result?
- Did a proxy act like a bad control?
- Did selection/censoring induce collider bias?
- Did posterior predictive failures suggest missing structure?
- Should the question be reframed as descriptive rather than causal?
