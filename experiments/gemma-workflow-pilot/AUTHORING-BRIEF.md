# One small task: author the specified model

Read your local API.md, then WRITE model.py using the write tool. Do not run an
analysis. You have four minutes and only need this one Python file.

Do not read raw data, other arm directories, previous reports, other solutions or
any other files. Everything needed is here and in API.md. No shell, installation,
plots, diagnostics, saved results, study reports or approvals are required.
Do not merely paste code in chat: create the file, then briefly state it was written.
The evaluator will separately check and execute the model; do not claim a fit ran.

Data schema (actual arrays are supplied by the evaluator later):
- x: float64 vector of length N
- clinic_design: float64 matrix, shape (N, 8), one-hot clinic indicators
- y: observed float64 vector of length N

Exact model, with a non-centered clinic intercept:
- alpha ~ Normal(0, 2)
- beta ~ Normal(0, 1)
- tau ~ HalfNormal(1)
- z: vector of 8 independent Normal(0, 1) variables
- sigma ~ HalfNormal(1)
- mu = alpha + tau * (clinic_design @ z) + beta * x
- y ~ Normal(mu, sigma)

Normal's second argument is a standard deviation. Preserve the parameter names
alpha, beta, tau, z, sigma and the data names x, clinic_design, y.
The estimand beta is an associational within-clinic contrast, not a causal claim.

Use your assigned API, not a different modeling library. Do not add extra priors,
latent variables, constraints, inference code, main blocks or data-loading code.
