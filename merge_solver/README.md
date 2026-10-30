# Purpose
This code can be used to reproduce Figures 9(b) and 10 ("Experiment 5") in the Quilt paper (Section 7.5.2).

# Installation

## Requirements

### Gurobi
You need your own Gurobi license. You can obtain one for free at https://www.gurobi.com/academia/academic-program-and-licenses/.

Download Gurobi Optimizer for your platform. First log in and then download at: https://www.gurobi.com/downloads/gurobi-software/.

After installation, run this in your terminal:
```bash
grbgetkey <your-license-key>
```
This creates a license file in your home directory (~/gurobi.lic).
Note: Keep `gurobi.lic` in your home directory and leave it there. Gurobi expects it up there. Gurobi also expects
you to be inside a university network when you activate your license.

### Python libraries

Quilt's partition algorithm is written in Python with Gurobi bindings.
We tested this with `Python 3.13.5`, but it should work with most Python 3 versions.
We need Python3 libraries: `networkx`, `gurobipy`, `matplotlib`.
You can install them with `pip`:

```bash
python3 -m venv quilt_venv
. ./quilt_venv/bin/activate
pip3 install networkx gurobipy matplotlib 
```

# Quilt's decision algorithm

## Experimental memory-model hook

`memory_model_template.py` provides two unfinished extension points:
`estimate_memory(graph, root, members)` for a concrete group's numeric memory
estimate, and `build_memory_expression(model, graph, root, roots, y, z)` for the
symbolic ILP formulation. Both raise `NotImplementedError` until you implement
them. The template documents graph inputs, traversal stages, and concurrency
assumptions that your implementation must specify.

After implementing the symbolic function, call the solver directly from the
repository root:

```python
from merge_solver.ilp import solve_subgraph_construction
from merge_solver.memory_model_template import build_memory_expression
from merge_solver.memory_inputs import prepare_memory_inputs
from merge_solver.rdag import preprocess_graph

# graph must use the solver's node m/c and edge weight/alpha/type fields.
prepare_memory_inputs(graph, workflow_invocations=10)
# Custom estimators can read graph.edges[u, v]["calls_per_caller"].
root, nodes, predecessors, reachable = preprocess_graph(graph)
status, cost, groups = solve_subgraph_construction(
    graph=graph, R_set={root}, M=150, C=100,
    all_nodes=nodes, predecessors=predecessors,
    full_reachable_from=reachable,
    memory_expression_builder=build_memory_expression,
)
```

Direct callers explicitly prepare the original graph once before solving or
searching roots. The Gurobi-independent helper stores node `profile_invocations`
and edge `calls_per_caller`, preserving `alpha`. Counts are the workflow count
for the original entry and the sum of incoming weights for other nodes. Ratios
are unrounded averages over all caller invocations, including shared callers;
they do not imply peak concurrency or context-specific behavior. Zero calls from
an invoked caller give zero; an uninvoked caller gives `None`, and positive calls
from an uninvoked caller are rejected. The testbench Quilt adapter prepares its
private solver graph automatically, without changing saved scenario JSON.

The callback is called for each candidate root before optimization. It may add
auxiliary variables/constraints and must return a linear expression in the same
memory units as `M`; the solver adds the capacity constraint. Membership is still
symbolic, so a traversal of an assumed final group cannot substitute for this
formulation. Keep common traversal logic in pure helpers, and compare the
symbolic formulation with your concrete estimator using fixed memberships.

Omitting the callback preserves Quilt's original formula. CPU accounting, root
selection, and their existing callers are unchanged. This hook is currently for
direct ILP calls: experiment configuration and root-search callback propagation
are not wired up. `print_solution_details()` and the testbench evaluator still
report the original Quilt memory formula; they must be adapted before validating
or reporting results from your custom model.

## Content
The code is split into 10 files.

Algorithms:
- `rdag.py` includes code to generate a random rDAG as well as utility functions for the rDAG such as finding the root and connectivity.
- `ilp.py` includes the solver logic (Gurobi calls and ILP constraints)
- `weighted_degree.py` selects roots based on weighted degree
- `downstream_impact.py` selects roots based on downstream impact heuristic
- `root_selection.py` first uses either optimal, weighted_degree, or downstream impact to find roots, then calls ILP to solve.

Tests and experiments:
- `tests.py` has unit tests.
- `experiment.py` has to code to run the experiment and save the results in a file called `merge_decision_result.json`.

Plotting and results parsing:
- `plot_merge_decision_time.py` has code to process the `json` file and output the graph in `Figure 9(b)`.
- `plot_merge_decision_quality.py` has code to process the `json` file and output the graphs in `Figure 10`.
- `summarize_results.py` will print summary of results which contain some of the values written in the main text of Section 7.5.2.


## Running the algorithm

First, confirm everything is working by running the `tests`

```bash
python3 tests.py
```

This should output something like:

```
<some debugging printouts>

Ran 22 tests in 2.422s

OK
```

If the above is OK, you are now ready to run the decision algorithm experiments to
reproduce figures 9(b) and 10. Before you run it, see the **WARNING** message below.

```bash
python3 experiment.py
```

This will produce a `JSON` file called `merge_decision_results.json` with the results.

To plot the figures, simply run:

```
python3 plot_merge_decision_time.py
python3 plot_merge_decision_quality.py
python3 summarize_results.py
```

You should now see 3 PDFs with figures corresponding to 9(b), 10(a) and 10(b).


**WARNING**: `experiment.py` takes a very long time to run (yikes!). 
This is because it tests 3 different approaches, across 10 different graph
scales, 100 times each (3000 runs total). Meanwhile, the optimal solution (the
red line in Figure 9b) takes 100s of seconds per run once the graph is large
enough (beyond 20) due to its exponential nature.

If you want to get a solution faster (**strongly recommended**), go to line 191 in 
`experiment.py` and reduce the `NUM_TRIALS` variable to run fewer times.
You can set it to something like 5 (the variance is not high from our experience).
With this change, we estimate the script will complete in around 30 minutes.
