# Pre-call Bank Marketing Response Prediction

Predict whether a customer will subscribe to a term deposit before the call takes place. The project compares linear and ensemble classifiers, calibrates the selected model, and chooses an operating threshold on a separate holdout.

## Setup

Use Python 3.11 or later. Commands below are run inside this project folder.
Create a separate environment for each project.

~~~bash
python -m venv .venv
~~~

Activate it with .venv\Scripts\activate on Windows, or source .venv/bin/activate
on Linux/macOS. Then:

~~~bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
~~~

The dependency versions are the tested CPU environment. No GPU, account, or
paid API is required. Package installation requires internet unless those
packages are already installed. Use requirements-notebook.txt if you also want
to run notebooks/Project_Walkthrough.ipynb in JupyterLab.

## Run

For a first local demonstration:

~~~bash
python main.py demo
~~~


~~~bash
python main.py download
python main.py train --mode quick --output outputs/quick
python main.py train --mode full --output outputs/full
python main.py predict --model outputs/full/models/model.joblib --input examples/customer.csv
~~~

The first regular run downloads the official UCI dataset. If the download fails, use the manual file location described in the error. A separate offline demonstration uses simulated data:

~~~bash
python main.py train --mode quick --data data/demo/bank_demo.csv --output outputs/demo
python main.py predict --model outputs/demo/models/model.joblib --input examples/customer.csv
~~~

The simulated demonstration checks the software; its scores are not benchmark results. Full mode uses the official dataset when --data is omitted. There is no silent fallback to simulated data.

## Outputs and code

Every run writes models, reports, and audit information inside the requested
output folder. Open reports/index.html in a browser to inspect the plots and
metric tables. reports/metrics.csv holds the measured comparison.
audit/run.json records settings, versions, seeds, and data provenance.
The inference command loads the saved training artifact; it does not retrain.
Load only checkpoints produced by this repository or another trusted source,
because joblib uses Python object serialization.

project/ contains the implementation. main.py is the command-line entry point.
tests/ exercises the data protocol and essential mathematical behavior.
notebooks/Project_Walkthrough.ipynb runs the same implementation step by step.
METHODOLOGY.md explains the experiment, DATA_CARD.md documents the source,
and STUDY_GUIDE.md lists the code and concepts for the project exam.
VALIDATION.md states what was actually verified in the delivery environment.

Quick mode is for checking the workflow. Use full mode for the course's main
experiment and report the run's actual results. Seeds support reproducibility,
but numerical differences can occur across dependency versions or platforms.

## GitHub

This folder is an independent repository root. It includes a dependency file,
package metadata, a license, ignore rules, and a GitHub Actions test workflow.
Upload the folder's contents, including .github and .gitignore, as one repository.
Do not upload .venv, data/raw, or outputs. Original code uses MIT; public
datasets retain the licenses and attribution described in DATA_CARD.md.

demo_run/ includes a completed, explicitly simulated software demonstration and its saved model. Inspect it without retraining:

~~~bash
python main.py predict --model demo_run/models/model.joblib --input examples/customer.csv
~~~

**Development date:** March 2024
