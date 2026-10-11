# Study guide

Read project/data.py, project/models.py, project/workflow.py, and project/cli.py
in that order. Run the complete experiment and open the HTML report.

Be prepared to explain:

1. The exact point in time at which each input would be available.
2. Why duration is excluded and why keeping it would change the task.
3. Why the five partitions have distinct roles.
4. How placing preprocessing outside the fitted pipeline could leak information.
5. How class weighting changes a logistic classifier and its probabilities.
6. Why average precision is more informative than accuracy alone here.
7. The difference between calibration and threshold selection.
8. Which quantities sigmoid calibration estimates while the classifier is frozen.
9. How a different false-positive cost would change the threshold objective.
10. What a negative permutation importance means.
11. Why day and month do not support a customer-independent temporal guarantee.
12. How the saved artifact reproduces preprocessing at inference.

For the practical exam, load a constructed customer with an unseen job category,
trace its transformed features, and obtain a probability and decision. Then
change the threshold-selection objective to a stated precision requirement.
Make the change on the threshold partition, document infeasible cases, and
explain the expected recall tradeoff without choosing from test labels.
