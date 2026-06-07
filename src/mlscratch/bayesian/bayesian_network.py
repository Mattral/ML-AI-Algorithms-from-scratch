"""
Bayesian Network
=================
A Directed Acyclic Graph (DAG) where each node represents a discrete random
variable and each node stores a Conditional Probability Table (CPT).

Supports:
- Manual CPT specification
- Exact inference via Variable Elimination
- Ancestral sampling

Notation
--------
Each variable is a string name.  Observations are dicts {name: value}.
CPTs are given as numpy arrays indexed in the order (var, *parents).

Only numpy and Python stdlib are used.
"""

import numpy as np
from itertools import product


class BayesianNetwork:
    """
    Discrete Bayesian Network.

    Usage
    -----
    >>> bn = BayesianNetwork()
    >>> bn.add_variable('Rain',     2)
    >>> bn.add_variable('Sprinkler',2, parents=['Rain'])
    >>> bn.add_variable('Wet',      2, parents=['Rain','Sprinkler'])
    >>> bn.set_cpt('Rain',      np.array([0.8, 0.2]))
    >>> bn.set_cpt('Sprinkler', np.array([[0.6,0.4],[0.99,0.01]]))
    >>> bn.set_cpt('Wet',       np.array([[[0.99,0.01],[0.1,0.9]],
    ...                                    [[0.1,0.9],[0.01,0.99]]]))
    >>> bn.query('Wet', evidence={'Rain':1})
    """

    def __init__(self):
        self._parents: dict[str, list] = {}   # name → parent names
        self._cpt: dict[str, np.ndarray] = {}  # name → CPT array
        self._domain: dict[str, int] = {}      # name → domain size
        self._order: list[str] = []            # topological insertion order

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_variable(
        self,
        name: str,
        domain_size: int,
        parents: list | None = None,
    ) -> None:
        """
        Register a variable.

        Parameters
        ----------
        name : str
        domain_size : int
            Number of possible values (0 … domain_size-1).
        parents : list of str or None
            Names of parent variables (must already be added).
        """
        self._domain[name] = domain_size
        self._parents[name] = parents or []
        self._order.append(name)

    def set_cpt(self, name: str, cpt: np.ndarray) -> None:
        """
        Set the CPT for a variable.

        The array shape must be:
        (domain_size,)                  for root nodes (no parents)
        (*parent_domain_sizes, domain_size)  for nodes with parents
        """
        self._cpt[name] = np.array(cpt, dtype=float)

    # ------------------------------------------------------------------
    # Topological order (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_sort(self) -> list:
        in_degree = {n: len(self._parents[n]) for n in self._order}
        queue = [n for n in self._order if in_degree[n] == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in self._order:
                if node in self._parents[child]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        return result

    # ------------------------------------------------------------------
    # Ancestral sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples: int = 1, random_state=None) -> list:
        """
        Generate samples by ancestral sampling.

        Returns
        -------
        samples : list of dicts {variable_name: value}
        """
        rng = np.random.default_rng(random_state)
        order = self._topological_sort()
        results = []

        for _ in range(n_samples):
            assignment = {}
            for var in order:
                parents = self._parents[var]
                cpt = self._cpt[var]
                if not parents:
                    probs = cpt
                else:
                    idx = tuple(assignment[p] for p in parents)
                    probs = cpt[idx]
                assignment[var] = int(rng.choice(len(probs), p=probs))
            results.append(assignment)

        return results if n_samples > 1 else results[0]

    # ------------------------------------------------------------------
    # Variable Elimination (exact inference)
    # ------------------------------------------------------------------

    def query(
        self,
        query_var: str,
        evidence: dict | None = None,
    ) -> np.ndarray:
        """
        Compute P(query_var | evidence) via variable elimination.

        Parameters
        ----------
        query_var : str
        evidence : dict {var_name: observed_value} or None

        Returns
        -------
        proba : ndarray of shape (domain_size_of_query_var,)
        """
        evidence = evidence or {}

        # Build initial factors from CPTs
        # A factor maps a tuple of variable names to an ndarray
        factors: list[tuple[tuple, np.ndarray]] = []

        for var in self._order:
            cpt = self._cpt[var].copy()
            scope = tuple(self._parents[var] + [var])
            # Reduce observed variables
            reduced_scope = []
            reduced_cpt = cpt
            for i, v in enumerate(scope):
                if v in evidence:
                    # Index into that axis
                    sl = [slice(None)] * len(scope)
                    sl[i] = evidence[v]
                    reduced_cpt = reduced_cpt[tuple(sl)]
                else:
                    reduced_scope.append(v)
            factors.append((tuple(reduced_scope), reduced_cpt))

        # Determine elimination order: all non-query, non-evidence variables
        to_eliminate = [
            v for v in self._order
            if v != query_var and v not in evidence
        ]

        for var in to_eliminate:
            # Collect factors that involve `var`
            relevant = [(s, f) for s, f in factors if var in s]
            remaining = [(s, f) for s, f in factors if var not in s]

            # Multiply relevant factors
            product_scope, product_factor = self._factor_product(relevant)

            # Sum out `var`
            var_idx = list(product_scope).index(var)
            summed = np.sum(product_factor, axis=var_idx)
            new_scope = tuple(s for s in product_scope if s != var)

            remaining.append((new_scope, summed))
            factors = remaining

        # Multiply remaining factors
        if not factors:
            return np.ones(self._domain[query_var]) / self._domain[query_var]

        final_scope, final_factor = self._factor_product(factors)

        # Sum out everything except query_var
        while len(final_scope) > 1:
            for i, v in enumerate(final_scope):
                if v != query_var:
                    final_factor = np.sum(final_factor, axis=i)
                    final_scope = tuple(s for j, s in enumerate(final_scope) if j != i)
                    break

        result = final_factor.ravel()
        total = result.sum()
        return result / total if total > 0 else result

    def _factor_product(
        self, factors: list[tuple[tuple, np.ndarray]]
    ) -> tuple[tuple, np.ndarray]:
        """Multiply a list of (scope, array) factors together."""
        if not factors:
            return ((), np.array(1.0))

        # Compute union scope (maintaining order)
        union_scope = []
        for scope, _ in factors:
            for v in scope:
                if v not in union_scope:
                    union_scope.append(v)
        union_scope = tuple(union_scope)

        # Build shape
        shape = tuple(self._domain[v] for v in union_scope)
        result = np.ones(shape)

        for scope, factor in factors:
            # Expand factor axes to match union_scope
            expand_axes = [union_scope.index(v) for v in scope]
            expanded = np.ones(shape)
            # Use np.einsum-style axis alignment
            source_shape = [self._domain[v] for v in scope]
            reshaped = factor.reshape(source_shape)

            # Map each axis of scope into union_scope
            new_shape = [1] * len(union_scope)
            for i, ax in enumerate(expand_axes):
                new_shape[ax] = source_shape[i]
            expanded = reshaped.reshape(new_shape)
            result = result * expanded

        return union_scope, result
