# File-only handoff

## Scientific state

The question is the expected within-clinic outcome difference for a one-unit increase in x (beta): an associational conditional contrast, not a causal effect. The saved input has 160 finite observations across eight one-hot clinics, 20 per clinic; x/design are fixed and units arbitrary.

Current proposal: `model_revised.py`, with beta ~ Normal(0, **0.25 SD**), alpha ~ Normal(0,2), tau and sigma ~ HalfNormal(1), and eight independent z ~ Normal(0,1). The non-centered likelihood is y ~ Normal(alpha + tau*(clinic_design @ z) + beta*x, sigma). Assumptions include a shared linear slope, exchangeable Gaussian clinic intercepts, independent Gaussian residuals and common residual scale.

`STUDY.md` explicitly records a scripted revision awaiting scientific review. **Nothing is human-approved.** Completed execution, agent figure inspection and passing computational screens are not approval.

Current evidence is `results/revised/`, `results/revised-recovery/` and `results/revised-prior/`. `model.py` (beta SD 1), `STUDY-initial.md`, and `results/initial/`, `results/recovery/`, `results/prior/` are preserved but superseded for current-model inference. The historical material appended to `STUDY.md` is not current status. `results/failed-prior-with-y/` is a failed preparation, not a successful simulation.

## Independent posterior reconstruction

Loaded `results/revised/posterior.npz` directly, preserving its four-chain by 1,000-draw beta array. Using NumPy sample SD (ddof=1), equal-tailed quantiles and ArviZ 0.22.0:

| Metric | Recomputed |
|---|---:|
| Mean | 0.5611618697217569 |
| SD | 0.056388976291427445 |
| 2.5% quantile | 0.4498333725531566 |
| 97.5% quantile | 0.6732247128279578 |
| Rank Rhat | 1.0019683482042574 |
| Bulk ESS | 2562.207597703112 |
| Tail ESS | 2601.640881989224 |
| MCSE(mean) | 0.001114578369926984 |

All eight match `results/revised/result.json` within rtol=atol=1e-12 (indeed printed values agree exactly). These are newly calculated, not copied summary values. Native `posterior.ndjson` uses different classic diagnostic definitions; it is not the source of these rank diagnostics. Saved run settings are Bayesjax/BlackJAX NUTS, seed 4201, float64, 4 chains, 500 warmup, 1,000 retained draws, target acceptance 0.9, depth limit 10.

## Provenance verification and limits

Exact-byte SHA-256 checks passed for all 42 entries in `provenance.json`, all 49 in `revision-provenance.json`, and all 60 in `preservation-before-revision.json` (overlapping inventories, not 151 unique files). Inventory checks included hashing recorded logs without reading earlier agent logs.

After reading its code, ran `verify_revision.py` without `--record`. All checks passed: source text changes only the status description and beta SD; both saved IR representations change only beta scale from 1 to 0.25; revised canonical inputs and sampler settings match their historical counterparts; native source/input hash records match exact files; posterior NPZ values equal native NDJSON draws for every parameter; and posterior model/data fingerprints agree with saved summaries and header/trailer. No model execution/refit occurred.

Additionally reconstructed canonical arrays from each revised `data.json` and compared them exactly with `input/data.json`, `input/recovery.json`, or `design.json`, respectively. All matched. Original byte hashes and canonical serialization hashes are distinct and were not conflated.

Key verified SHA-256 values:

- `model_revised.py`: `77cc13070f6bcbdcface3af3f14e8bc3b284c98311c8cc63a8345636b0f82c6c`
- `input/data.json`: `8a753a950786ff60f47747dd0176ff01619d2f958402e0d042299fcb4523018d`
- `results/revised/model.ir.json`: `64572ad19ff1bcc9607cdbce6adfab8ac15a2d444be27b2e414f48fdc3cd7911`
- `results/revised/data.json`: `3231a98c65616c2edaff34e572ed9b59887b88a0ce1b2a16a0b7b725964abd34`
- `results/revised/posterior.npz`: `0ccfcb712da1103ed6f533608c732b4652463bd9a33aafc85a807bb96d6b6641`
- Native combined model/data fingerprint: `sha256:49a328509aa3c2dcc53494c6ccb25ceeb48cbbd7ac67acc073621370af90f108`.

These checks establish local saved-artifact consistency, not independent attestation of execution history or data collection. A Python source hash does not prove arbitrary Python execution deterministic, capture imported-code/environment/external-state effects, or prove source-to-IR generation; no fresh compilation or sampling was attempted. Recorded versions/settings are evidence, not a verified replay environment. Absolute source paths in `run.json` also limit relocation/replay. The supplied data's real-world validity and scientific assumptions cannot be verified from these artifacts.

## Caveats and next human decision

Saved parameter summaries flag no coordinate at rank Rhat >1.01 or bulk/tail ESS <400. Independently counted zero divergences and no depth-limit hits in current supplied/recovery sample stats (maximum depths 6 and 7). Alpha Rhat is about 1.0094 and tau bulk ESS about 816: screens are finite-run evidence, not proof of convergence or correctness.

Recovery beta truth 0.4 is outside the revised interval [0.1552749496, 0.3655355886], as it was outside the historical interval. One fixture cannot establish calibration or justify selecting another seed. The supplied-data beta mean decreased by about 0.02808 under the tighter prior; sensitivity is not validation of that prior.

`STUDY.md` reports agent-inspected trace and predictive figures; this handoff did not independently inspect images and implies no human review. Its marginal mean/SD PPCs, using 200 conditional predictions at the observed clinic design, do not assess clinic-specific residuals, tails, structural adequacy or new-clinic generalization. Arbitrary units limit prior-plausibility assessment. Dedicated Bayesjax diagnose/PPC support was limited; supplementary `analysis_revised.py` calculations must not be mistaken for native CLI diagnostic features.

The next action is a **human scientific decision** about the substantive basis for beta SD 0.25, persistent recovery noncoverage, the shared-slope/exchangeability/Gaussian assumptions, richer checks or revision, and whether any scientific use is appropriate. No study advancement is authorized by this handoff.

## Reproducible read-only calculations

Run from this directory in the existing environment; no regeneration or installation is required:

```sh
uv run --project .. python - <<'PY'
from pathlib import Path
import hashlib, json
import numpy as np
import arviz as az
p = Path('results/revised')
b = np.load(p/'posterior.npz')['beta']
d = az.from_dict(posterior={'beta': b})
m = dict(mean=float(b.mean()), sd=float(b.std(ddof=1)),
         q025=float(np.quantile(b, .025)), q975=float(np.quantile(b, .975)),
         rhat=float(az.rhat(d, method='rank').beta),
         ess_bulk=float(az.ess(d, method='bulk').beta),
         ess_tail=float(az.ess(d, method='tail').beta),
         mcse_mean=float(az.mcse(d, method='mean').beta))
s = json.loads((p/'result.json').read_text())['beta']
assert all(np.isclose(v, s[k], rtol=1e-12, atol=1e-12) for k,v in m.items())
print(az.__version__, b.shape, json.dumps(m, indent=2))
for name in ['provenance.json', 'revision-provenance.json',
             'preservation-before-revision.json']:
    obj = json.loads(Path(name).read_text())
    hashes = obj.get('sha256', obj)
    for f,h in hashes.items():
        assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, f
    print(name, len(hashes), 'hashes match')
for stage, source in [('revised','input/data.json'),
                      ('revised-recovery','input/recovery.json'),
                      ('revised-prior','design.json')]:
    run = Path('results')/stage
    original = json.loads(Path(source).read_text())
    canonical = json.loads((run/'data.json').read_text())['variables']
    assert set(original) == set(canonical)
    for k,v in canonical.items():
        a = np.asarray(v['values'], dtype=v['dtype']).reshape(v['shape'])
        np.testing.assert_array_equal(a, original[k])
    if stage != 'revised-prior':
        stats = np.load(run/'sample_stats.npz')
        print(stage, 'divergences', stats['diverging'].sum(),
              'max depth', stats['tree_depth'].max(),
              'depth-limit hits', (stats['tree_depth'] >= 10).sum())
PY
uv run --project .. python verify_revision.py
```

Only `HANDOFF.md` and `handoff-result.json` were written. Existing study artifacts were not changed; no fitting or simulations were run.
