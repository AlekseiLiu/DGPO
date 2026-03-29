---
name: validate
description: >
  Validate the DGPO companion repository before running experiments or publishing.
  Use when the user asks to check, verify, or validate the repo setup, or when
  troubleshooting import errors, broken configs, or missing data. Also use after
  making code changes to confirm nothing is broken.
---

# Validate DGPO Repository

Run a 6-step validation to confirm the repo is correctly set up.

## Procedure

### Step 1: Import all source modules

```bash
python -c "
from src.model import GraphTransformer
from src.diffusion import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.train_pretrain import main as pretrain_main
from src.train_rlft import main as rlft_main
from src.rewards import RewardComputer
from src.evaluate import sample_architectures, run_single_evaluation, run_aggregate
from src.baselines import random_search_baseline, pretrained_only_baseline
from src.dataset import load_nb101, load_nb201
from src.ops import NB101_OP_PRIMITIVES, NB201_OP_PRIMITIVES, upper_triangular
from src.utils import PlaceHolder, load_config
print('All imports OK')
"
```

If any import fails, diagnose and fix before continuing.

### Step 2: Parse all YAML configs

```bash
python -c "
import yaml, glob, sys
errors = []
for path in sorted(glob.glob('configs/*.yaml')):
    try:
        with open(path) as f: yaml.safe_load(f)
    except Exception as e: errors.append(f'{path}: {e}')
if errors:
    for e in errors: print(f'  FAIL: {e}')
    sys.exit(1)
print(f'All {len(glob.glob(\"configs/*.yaml\"))} configs valid')
"
```

### Step 3: Verify Makefile targets

Run `make help` and confirm all targets are listed: setup, build-nb201-cache, pretrain-nb101, pretrain-nb201, rlft-nb101, rlft-nb201, evaluate, clean.

### Step 4: Smoke test (requires data + GPU)

```bash
./scripts/run_experiment_1_nb201.sh --quick
```

Runs a minimal pipeline: pretrain (5 epochs), RL-FT (3 epochs), evaluate (50 samples), aggregate. Skip if data or GPU unavailable.

### Step 5: Verify output format

After smoke test, check the output JSON:

```bash
python -c "
import json, sys
with open('outputs/experiment_1_nb201_summary.json') as f:
    data = json.load(f)
assert 'tasks' in data and isinstance(data['tasks'], dict), 'Bad format'
print('Output format OK')
"
```

### Step 6: Code quality check

```bash
grep -rn 'TODO\|FIXME\|HACK' src/ scripts/ configs/ --include='*.py' --include='*.yaml' --include='*.sh'
```

No matches expected in source code (vendored code may have upstream TODOs -- those are fine).

## Report

Summarize results:

```
VALIDATION: Step 1 (imports) / Step 2 (configs) / Step 3 (Makefile) /
            Step 4 (smoke test) / Step 5 (output) / Step 6 (quality)
Result:     PASS / FAIL / SKIPPED for each
```
