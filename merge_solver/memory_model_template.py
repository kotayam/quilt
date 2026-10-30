"""Extension points for an experimental memory model; no algorithm is supplied.

Use the same memory units as graph.nodes[node]['m'] and the solver's M limit.
This module deliberately does not import Gurobi so the concrete estimator can
eventually be used independently of the optimizer.
"""


def estimate_memory(graph, root, members):
    """Return numeric memory usage for a concrete group containing ``root``.

    First call ``prepare_memory_inputs(graph, workflow_invocations)`` from
    ``merge_solver.memory_inputs`` once on the original graph before root search.
    Inputs use the prepared solver graph format:
      * node ``m``: memory in the caller's chosen units;
      * edge ``weight``: invocation count in the original profiling window;
      * edge ``alpha``: existing multiplicity, normalized to the original workflow;
      * edge ``type``: 'sync' or 'async'.
      * node ``profile_invocations``: observed invocation count;
      * ``graph.edges[u, v]["calls_per_caller"]``: weight divided by the caller's
        profile_invocations, without rounding; None for an uninvoked caller.

    ``members`` is the selected group's collection of node IDs, not ILP variables.
    Do not mutate the graph or overwrite alpha. Timing information is not
    supplied; define what execution assumptions your algorithm requires.
    Ratios average across all observed caller invocations, including shared
    callers, and do not describe context-specific behavior. Counts do not determine
    concurrency, particularly for shared callees and conditional paths.

    Implementation outline:
      1. Traverse the concrete group downward from root to derive root-relative
         invocation counts and concurrency separately, preserving calling context
         where necessary. Do not multiply workflow-normalized alpha blindly.
      2. Traverse upward from the leaves to aggregate memory under your explicit
         overlap/lifetime assumptions. Account for distinct invocations of shared
         callees instead of assuming each graph node is executed once.
      3. Return memory usage in the same units as the node m values.
    """
    raise NotImplementedError("Implement concrete-group memory estimation first")


def build_memory_expression(model, graph, root, roots, y, z):
    """Return a Gurobi linear expression for one candidate root's memory usage.

    Called during ILP construction, before group membership has been selected:
      * ``model`` is the active Gurobi model;
      * ``roots`` contains all candidate roots in this ILP;
      * ``y[node, root]`` is the binary node-membership variable;
      * ``z[source, target, root]`` is binary internal-edge membership.

    First call ``prepare_memory_inputs(graph, workflow_invocations)`` from
    ``merge_solver.memory_inputs`` once before root search or a direct solve.
    Read caller-relative ratios through
    ``graph.edges[u, v]["calls_per_caller"]`` and counts through node
    ``profile_invocations``. Ratios are unrounded averages across all observed
    caller invocations (including shared callers), or None for uninvoked callers;
    they do not establish peak concurrency or context-specific behavior.

    The caller adds ``expression <= M``. You may add auxiliary variables and
    constraints here, using names unique to this root. Variables have no solution
    values yet: do not read .X or branch on symbolic membership. In particular,
    estimate_memory cannot be called on an assumed final group to encode the ILP.

    Implementation outline:
      1. Precompute any membership-independent traversal data for this root.
      2. Express membership-dependent concurrency and bottom-up aggregation using
         y/z and auxiliary linear constraints as needed. Avoid variable products
         or Python max() on symbolic expressions without a valid formulation.
      3. Return the resulting linear memory expression. Check it against
         estimate_memory for fixed memberships in tests.

    Leave the original graph alpha unchanged so CPU accounting is unaffected.
    """
    raise NotImplementedError("Implement the symbolic memory formulation before enabling it")
