# Validation

The delivery environment used Python 3.12.14 with the versions
in requirements.txt. All 5 tests in this repository passed.
The complete delivery contains 26 passing tests across the five projects.
Every project's command-line prediction or replay path also completed.

Verified properties include the project's mathematical and data-protocol checks,
model serialization, and the example inference path. External ZIP member selection,
nested archives, download caching, and downloaded-file hashes were tested with
mocked responses in projects that download a dataset.

Notebook files were checked as JSON notebooks, and every code cell was checked
for Python syntax. The underlying experiments were executed from the command line.
The Jupyter interface was not executed in the delivery environment.

The real public dataset could not be downloaded in this environment
because outbound access to its host was blocked. The complete training,
selection, evaluation, report generation, and inference workflow ran on an
explicitly simulated schema fixture. Both quick and full configurations were
exercised on that fixture. No real-dataset benchmark result is claimed.

The supplied demo_run is the quick simulated demonstration. data/demo contains
its inputs. Full-mode course experiments use the official dataset after download.
There is no automatic substitution of simulated data when downloading fails.
