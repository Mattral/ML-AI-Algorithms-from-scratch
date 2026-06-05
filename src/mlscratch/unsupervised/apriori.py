"""
Apriori Algorithm for Association Rule Mining
==============================================
Discovers frequent itemsets in a transaction database and generates
association rules with user-specified support and confidence thresholds.

Core concepts
-------------
- Support(A)    = (# transactions containing A) / (# transactions total)
- Confidence(A→B) = Support(A ∪ B) / Support(A)
- Lift(A→B)       = Confidence(A→B) / Support(B)

The Apriori property: every subset of a frequent itemset is frequent.
This prunes the candidate search space dramatically.

Only Python stdlib and numpy are used.
"""

from itertools import combinations
from collections import defaultdict


class Apriori:
    """
    Apriori frequent-itemset mining and association-rule generation.

    Parameters
    ----------
    min_support : float
        Minimum support threshold in [0, 1].
    min_confidence : float
        Minimum confidence threshold for rule generation, in [0, 1].
    min_lift : float
        Minimum lift for rule generation (default 1.0, i.e. no filter).
    """

    def __init__(
        self,
        min_support: float = 0.5,
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

        self.frequent_itemsets_ = {}   # frozenset → support
        self.rules_ = []               # list of dicts with antecedent/consequent/metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_support(self, itemset: frozenset, transactions: list) -> float:
        """Compute support of an itemset over the transaction list."""
        count = sum(1 for t in transactions if itemset.issubset(t))
        return count / len(transactions)

    def _generate_candidates(self, prev_frequent: list, k: int) -> list:
        """
        Generate candidate k-itemsets by joining (k-1)-itemsets that share
        their first k-2 items (standard Apriori join step).
        """
        candidates = set()
        prev_list = sorted([sorted(fs) for fs in prev_frequent])
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                # Join if first k-2 elements match
                if prev_list[i][:k - 2] == prev_list[j][:k - 2]:
                    union = frozenset(prev_list[i]) | frozenset(prev_list[j])
                    if len(union) == k:
                        candidates.add(union)
        return list(candidates)

    def _prune_candidates(
        self, candidates: list, prev_frequent_set: set, k: int
    ) -> list:
        """
        Remove candidates that have an infrequent subset (Apriori pruning).
        """
        pruned = []
        for candidate in candidates:
            subsets = [
                frozenset(s) for s in combinations(candidate, k - 1)
            ]
            if all(s in prev_frequent_set for s in subsets):
                pruned.append(candidate)
        return pruned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, transactions: list) -> "Apriori":
        """
        Mine frequent itemsets from a list of transactions.

        Parameters
        ----------
        transactions : list of list/set
            Each element is a collection of items in one transaction.
            e.g. [['milk', 'bread'], ['milk', 'eggs', 'butter'], ...]

        Returns
        -------
        self
        """
        # Encode transactions as frozensets
        trans = [frozenset(t) for t in transactions]
        all_items = sorted({item for t in trans for item in t})

        # --- Pass 1: find frequent 1-itemsets ---
        frequent_k = {}
        for item in all_items:
            fs = frozenset([item])
            sup = self._get_support(fs, trans)
            if sup >= self.min_support:
                frequent_k[fs] = sup

        self.frequent_itemsets_.update(frequent_k)

        k = 2
        while frequent_k:
            prev_frequent_set = set(frequent_k.keys())

            # Generate and prune candidates
            candidates = self._generate_candidates(list(frequent_k.keys()), k)
            candidates = self._prune_candidates(candidates, prev_frequent_set, k)

            # Count support
            new_frequent = {}
            for candidate in candidates:
                sup = self._get_support(candidate, trans)
                if sup >= self.min_support:
                    new_frequent[candidate] = sup

            self.frequent_itemsets_.update(new_frequent)
            frequent_k = new_frequent
            k += 1

        # --- Generate association rules ---
        self.rules_ = []
        for itemset, itemset_sup in self.frequent_itemsets_.items():
            if len(itemset) < 2:
                continue
            # Try all non-empty proper subsets as antecedents
            for r in range(1, len(itemset)):
                for antecedent in map(frozenset, combinations(itemset, r)):
                    consequent = itemset - antecedent
                    ant_sup = self.frequent_itemsets_.get(antecedent, 0)
                    con_sup = self.frequent_itemsets_.get(consequent, 0)

                    if ant_sup == 0:
                        continue

                    confidence = itemset_sup / ant_sup
                    lift = confidence / con_sup if con_sup > 0 else 0.0

                    if confidence >= self.min_confidence and lift >= self.min_lift:
                        self.rules_.append({
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "support": itemset_sup,
                            "confidence": confidence,
                            "lift": lift,
                        })

        return self

    def get_frequent_itemsets(self) -> list:
        """Return list of (itemset, support) tuples sorted by support desc."""
        return sorted(
            self.frequent_itemsets_.items(), key=lambda x: -x[1]
        )

    def get_rules(self) -> list:
        """Return association rules sorted by confidence descending."""
        return sorted(self.rules_, key=lambda r: -r["confidence"])
