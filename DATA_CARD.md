# Data card

Canonical dataset: UCI Bank Marketing, using the older bank-full.csv variant
inside bank.zip. Do not substitute bank-additional-full.csv; its schema differs.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

Attribution: Moro, S., Rita, P., and Cortez, P. (2014). Bank Marketing.
UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5K306

UCI lists the dataset under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/
The original dataset is not included in this repository. The downloader reads
only the named CSV member and records its actual SHA-256 after downloading.
Raw data stays in data/raw and is excluded from Git.

Features include customer attributes, contact calendar and channel, and previous
campaign information. The target is y, with yes and no labels. Duration is
excluded. Unknown categories remain explicit. Exact duplicate feature-and-label
rows are removed before partitioning.

data/demo/bank_demo.csv is an original, simulated software demonstration fixture
with the expected schema. Its target pattern is constructed, and its scores
cannot be interpreted as bank-marketing benchmark performance. Code and original
demo fixtures use the repository's MIT license. examples/customer.csv contains
constructed examples.
