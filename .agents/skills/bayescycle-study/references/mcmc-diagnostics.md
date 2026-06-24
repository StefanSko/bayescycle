# MCMC diagnostics intuition

Use this as a light checklist during `simulation`, `fit`, and `critique` phases.
Diagnostics answer one question:

```text
Do these draws actually represent the posterior quantity we want to use?
```

Do diagnostics before interpreting posterior results. If diagnostics fail, report
the failure or revise; do not quietly compute the estimand anyway.

## Mental model

Think of each chain as a hiker exploring the posterior landscape. We want
multiple hikers, started in different places, to cover the same high-probability
terrain in the right proportions. Diagnostics ask whether they mixed, agreed,
and produced enough independent information for the estimand.

## Warmup is not posterior sample

Warmup/adaptation draws are tuning experiments. Do not summarize them as
posterior draws. Approved posterior summaries must use post-warmup draws only.

## Visual checks

Use visual checks when possible:

- trace plots: healthy chains look stationary and mixed, not drifting or stuck
- rank plots: healthy chains take turns being high and low in rank
- autocorrelation plots: reveal redundant draws
- energy plots when available: reveal HMC/NUTS exploration problems

Trace plots are useful but not sufficient. Rank plots often expose chain
separation that overlaid traces hide.

## Numeric checks

- R-hat asks whether chains agree. It should be very close to 1 for quantities
  used in inference.
- ESS asks how much independent information remains after autocorrelation.
- Bulk-ESS is about the center of the posterior.
- Tail-ESS is about extreme quantiles and may matter for interval/tail claims.

Avoid magic thresholds. A borderline diagnostic may be fine for a rough mean but
not for a tail probability or policy decision. Treat project thresholds as
repair prompts, not as substitutes for judgment.

## Check the estimand too

Diagnostics on raw parameters are not enough. If the approved estimand is a
function of parameters or predictions, check diagnostics for that derived
quantity when the toolchain makes this practical.

Example: if the estimand is

```text
P(adoption <= 30 days | black) - P(adoption <= 30 days | non-black)
```

then diagnostic review should address that contrast, not only the coefficients
used to compute it.

## NUTS-specific red flags

When reported, these usually block approval unless explicitly waived:

- chain failures
- nonzero divergences in final inference
- max-treedepth saturation that changes interpretation
- bad R-hat or ESS for important quantities
- energy problems / low E-BFMI when available

## Gate rule

A fit artifact can be proposed before diagnostics are perfect, but it should not
be approved until failures are fixed, justified, or waived by a recorded human
decision.
