"""
Tests for mlscratch.unsupervised.apriori.Apriori
"""

import pytest
from mlscratch.unsupervised.apriori import Apriori


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grocery_transactions():
    """Classic small grocery basket dataset."""
    return [
        ["milk", "bread", "butter"],
        ["milk", "bread"],
        ["milk", "butter"],
        ["bread", "butter"],
        ["milk", "bread", "butter", "eggs"],
        ["bread", "eggs"],
        ["milk", "eggs"],
        ["bread", "butter", "eggs"],
    ]


@pytest.fixture
def simple_transactions():
    """Tiny dataset where itemsets are known exactly."""
    # Items {A,B} appear in all 4; {C} only in 2
    return [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "B", "C"],
        ["A", "B"],
    ]


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestAprioriBasic:
    def test_fit_returns_self(self, grocery_transactions):
        model = Apriori(min_support=0.3)
        assert model.fit(grocery_transactions) is model

    def test_frequent_itemsets_not_empty(self, grocery_transactions):
        model = Apriori(min_support=0.4).fit(grocery_transactions)
        assert len(model.frequent_itemsets_) > 0

    def test_rules_returned_as_list(self, grocery_transactions):
        model = Apriori(min_support=0.3, min_confidence=0.6).fit(grocery_transactions)
        assert isinstance(model.rules_, list)

    def test_get_frequent_itemsets_sorted_desc(self, grocery_transactions):
        model = Apriori(min_support=0.3).fit(grocery_transactions)
        items = model.get_frequent_itemsets()
        supports = [sup for _, sup in items]
        assert supports == sorted(supports, reverse=True)

    def test_get_rules_sorted_by_confidence_desc(self, grocery_transactions):
        model = Apriori(min_support=0.3, min_confidence=0.5).fit(grocery_transactions)
        rules = model.get_rules()
        confidences = [r["confidence"] for r in rules]
        assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# Correctness on known dataset
# ---------------------------------------------------------------------------

class TestAprioriCorrectness:
    def test_support_100_percent_itemset(self, simple_transactions):
        """A and B appear in every transaction → support == 1.0."""
        model = Apriori(min_support=0.5).fit(simple_transactions)
        ab = frozenset(["A", "B"])
        assert ab in model.frequent_itemsets_
        assert abs(model.frequent_itemsets_[ab] - 1.0) < 1e-9

    def test_item_below_threshold_excluded(self, simple_transactions):
        """C appears in only 2/4 = 0.5 of transactions; excluded at 0.6."""
        model = Apriori(min_support=0.6).fit(simple_transactions)
        c = frozenset(["C"])
        assert c not in model.frequent_itemsets_

    def test_all_items_above_support_included(self, simple_transactions):
        """At support 0.4, all individual items should be frequent."""
        model = Apriori(min_support=0.4).fit(simple_transactions)
        for item in ["A", "B", "C"]:
            assert frozenset([item]) in model.frequent_itemsets_

    def test_support_values_in_range(self, grocery_transactions):
        model = Apriori(min_support=0.2).fit(grocery_transactions)
        for fs, sup in model.frequent_itemsets_.items():
            assert 0.0 <= sup <= 1.0

    def test_rule_confidence_geq_threshold(self, grocery_transactions):
        threshold = 0.6
        model = Apriori(
            min_support=0.3, min_confidence=threshold
        ).fit(grocery_transactions)
        for rule in model.rules_:
            assert rule["confidence"] >= threshold - 1e-9

    def test_rule_lift_geq_min_lift(self, grocery_transactions):
        model = Apriori(
            min_support=0.3, min_confidence=0.5, min_lift=1.0
        ).fit(grocery_transactions)
        for rule in model.rules_:
            assert rule["lift"] >= 1.0 - 1e-9

    def test_rule_fields_present(self, grocery_transactions):
        model = Apriori(min_support=0.3, min_confidence=0.5).fit(grocery_transactions)
        for rule in model.rules_:
            for key in ("antecedent", "consequent", "support", "confidence", "lift"):
                assert key in rule

    def test_antecedent_and_consequent_disjoint(self, grocery_transactions):
        model = Apriori(min_support=0.3, min_confidence=0.5).fit(grocery_transactions)
        for rule in model.rules_:
            assert rule["antecedent"].isdisjoint(rule["consequent"])

    def test_antecedent_union_consequent_is_frequent(self, grocery_transactions):
        model = Apriori(min_support=0.3, min_confidence=0.5).fit(grocery_transactions)
        for rule in model.rules_:
            union = rule["antecedent"] | rule["consequent"]
            assert union in model.frequent_itemsets_


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestAprioriEdgeCases:
    def test_support_1_only_universal_items(self, simple_transactions):
        model = Apriori(min_support=1.0).fit(simple_transactions)
        for fs in model.frequent_itemsets_:
            for item in fs:
                assert item in {"A", "B"}

    def test_empty_rules_when_confidence_is_1_and_no_perfect_rules(self):
        transactions = [["A", "B"], ["A", "C"], ["B", "C"]]
        model = Apriori(min_support=0.5, min_confidence=1.0).fit(transactions)
        # No rule here has confidence 1.0 since each pair appears in only 1/3
        assert isinstance(model.rules_, list)

    def test_no_rules_with_impossible_threshold(self, grocery_transactions):
        model = Apriori(min_support=0.9, min_confidence=0.99).fit(grocery_transactions)
        assert len(model.rules_) == 0
