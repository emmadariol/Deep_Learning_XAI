# Pipeline workflow

The project can be run from the command line through one pipeline wrapper:

~~~bash
python scripts/run_pipeline.py --profile outputs
~~~

The wrapper separates dataset creation, baseline training and output
generation. Normal output runs assume that the dataset manifest and baseline
checkpoint already exist.

## Main commands

Check which commands would be executed:

~~~bash
python scripts/run_pipeline.py --profile outputs --dry-run
~~~

Create or validate the portable dataset subset:

~~~bash
python scripts/run_pipeline.py --profile data --source-root /path/to/AwA2
~~~

Train the baseline checkpoint:

~~~bash
python scripts/run_pipeline.py --profile train
~~~

Generate the main figures and reports from existing data and checkpoint:

~~~bash
python scripts/run_pipeline.py --profile outputs
~~~

Run the full workflow, including TCAV and the Concept Bottleneck Model:

~~~bash
python scripts/run_pipeline.py --profile full
~~~

Recompute outputs instead of reusing them:

~~~bash
python scripts/run_pipeline.py --profile outputs --force
~~~

## Profiles

| Profile | Purpose | Stages |
| --- | --- | --- |
| `data` | Dataset subset preparation. | prepare, validate |
| `train` | Baseline model training. | validate, train |
| `outputs` | Main project outputs. | attribution examples, stress metrics, concept analysis, audits |
| `full` | Extended outputs. | `outputs` plus TCAV, TCAV stress, and CBM |

## Optional controls

Run only selected stages:

~~~bash
python scripts/run_pipeline.py --stages train xai stress
~~~

The portable subset is not created by `outputs` or `full`. Run `data` only when
the subset manifest is missing or must be regenerated. The baseline checkpoint
is not trained by `outputs` or `full`; run `train` only when the checkpoint is
missing or must be replaced.

Use the individual notebooks when the goal is to inspect intermediate outputs
or follow the analysis step by step.

For the complete list of options, run:

~~~bash
python scripts/run_pipeline.py --help
~~~
