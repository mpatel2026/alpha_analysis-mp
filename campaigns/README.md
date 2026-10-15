# Campaigns: batch ASCOT5 input creation + runs

One manifest per dataset -> `campaign.py` writes per-case params JSONs and two
chained Slurm job arrays (prep -> run). Run task *i* starts as soon as prep task
*i* succeeds (`--dependency=aftercorr`); if prep fails, the run task is cancelled.

```bash
python3 campaign.py plan   campaigns/<name>.json            # expand + check inputs exist
python3 campaign.py submit campaigns/<name>.json --dry-run  # sbatch --test-only
python3 campaign.py submit campaigns/<name>.json            # prep + run, all cases
python3 campaign.py submit campaigns/<name>.json --cases 2,4 --stages run
python3 campaign.py status campaigns/<name>.json            # h5 size + Slurm state per case
```

With the prep stage, every selected case gets a fresh input: an existing h5 is
deleted and recreated (`submit` lists which ones). Use `--stages run` to reuse
existing inputs.

## Recipes

| recipe | prep steps | jobs per case | resources |
|---|---|---|---|
| `beam` | create-input.py, set_mrk_gc.py | prep job, then run job (`aftercorr`) | prep: shared QOS, 1 GPU, 30 min; run: 2 nodes x 4 GPU, regular, 48 h |
| `fastparticle` | create-input.py, set_field_and_profiles.py | one `prep_run` job (prep then run) | shared QOS, 1 GPU, 1.5 h |

The shared QOS bills only for the 1/4 node you use. `--stages prep` or
`--stages run` still submit the separate `prep.sbatch` / `run.sbatch` for either recipe.

## If prep fails

Every step runs under `set -e`, so the first failing command ends the job
(state FAILED, log in `campaigns/<name>/logs/`). For `fastparticle` the
simulation never starts; for `beam` the waiting run task is cancelled
automatically (`--kill-on-invalid-dep`). Other cases are unaffected. The h5
may be left half-written (create-input deletes the old one first); fix the
cause and resubmit just that case with `--cases i`. `status` shows FAILED /
CANCELLED per case. A script that hits a problem but still exits 0 is not
caught; ASCOT then usually fails within seconds when it reads the input.

## Manifest

```json
{
  "recipe": "beam",
  "defaults": {                       // copied into every case's params JSON
    "inp_dir": "/pscratch/.../equil/Eos10-beams/",
    "out_dir": "/pscratch/.../ascot_h5s/Eos10-beams/",
    "equil_file": "equil_{tag}.h5",   // {tag} and any other key can be used in strings
    "sim_name": "ascot_{tag}",
    "marker_file": "beam-markers/markers_{tag}.json",   // relative to repo root
    "nmrk": 10000
  },
  "cases": ["TagA", "TagB", {"tag": "TagC", "nmrk": 20000}],
  "tag_glob": "EosOP3LowDensity*_P12MWbroader",   // optional: add a case for every equil_file match
  "slurm": {"run": {"time": "24:00:00", "max_concurrent": 4}},  // optional overrides
  "prep_steps": ["python -u create-input.py --params=\"$PARAMS\"", "..."],  // optional
  "site": {"conda_env": "...", "ascot_bin": "..."}                          // optional
}
```

- `fastparticle` cases also need `EC_file` and `SC_file` (looked up in `inp_dir/coils/`).
- `use_stell_sym` (default `true` for `beam`, `false` for `fastparticle`) is passed
  to create-input.py and also sets its toroidal resolution: `nPhi` 42 / 200 and
  AFSI `afsi_nsymm` 4 / 1 for true / false. Set `nPhi` or `afsi_nsymm` in the
  params to override. All other create-input settings are the same for both.
- Any `slurm.prep` / `slurm.run` key becomes an `#SBATCH --key=value` line.
  `max_concurrent` throttles the array (`%N`).
- Prep steps can use `$SIM`, `$H5`, `$PARAMS`, `$MARKER`. To add a step such as
  rescaling profiles, append it to `prep_steps`.

Generated per campaign in `campaigns/<name>/`: `params/<sim_name>.json`,
`cases.tsv` (row *i* = array task *i*), `prep.sbatch`, `run.sbatch`, `logs/`,
`jobs.jsonl` (submission history used by `status`). Edit the manifest, not these.
