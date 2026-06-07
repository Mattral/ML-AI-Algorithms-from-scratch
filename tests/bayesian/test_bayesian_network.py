"""
Tests for mlscratch.bayesian.bayesian_network.BayesianNetwork

Uses the classic Sprinkler / Wet Grass (Judea Pearl) network as ground truth.

        Rain  (R)
       /     \\
  Sprinkler   \\
    (S)       |
       \\     /
        WetGrass (W)

P(R=T)              = 0.2
P(S=T | R=F)        = 0.4
P(S=T | R=T)        = 0.01
P(W=T | R=F, S=F)   = 0.0
P(W=T | R=F, S=T)   = 0.9
P(W=T | R=T, S=F)   = 0.8
P(W=T | R=T, S=T)   = 0.99
"""

import numpy as np
import pytest
from mlscratch.bayesian.bayesian_network import BayesianNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sprinkler_net():
    """Classic sprinkler / wet-grass Bayesian network."""
    bn = BayesianNetwork()
    bn.add_variable("Rain",      2)
    bn.add_variable("Sprinkler", 2, parents=["Rain"])
    bn.add_variable("WetGrass",  2, parents=["Rain", "Sprinkler"])

    bn.set_cpt("Rain", np.array([0.8, 0.2]))                         # F, T
    bn.set_cpt("Sprinkler", np.array([[0.6, 0.4],                    # R=F: S=F,T
                                       [0.99, 0.01]]))               # R=T: S=F,T
    # shape: (Rain, Sprinkler, WetGrass)
    bn.set_cpt("WetGrass", np.array([
        [[1.0, 0.0], [0.1, 0.9]],   # R=F: S=F, S=T
        [[0.2, 0.8], [0.01, 0.99]], # R=T: S=F, S=T
    ]))
    return bn


@pytest.fixture
def chain_net():
    """Simple 3-node chain: A → B → C."""
    bn = BayesianNetwork()
    bn.add_variable("A", 2)
    bn.add_variable("B", 2, parents=["A"])
    bn.add_variable("C", 2, parents=["B"])
    bn.set_cpt("A", np.array([0.4, 0.6]))
    bn.set_cpt("B", np.array([[0.7, 0.3], [0.2, 0.8]]))
    bn.set_cpt("C", np.array([[0.9, 0.1], [0.1, 0.9]]))
    return bn


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class TestBNConstruction:
    def test_variables_registered(self, sprinkler_net):
        assert "Rain" in sprinkler_net._domain
        assert "Sprinkler" in sprinkler_net._domain
        assert "WetGrass" in sprinkler_net._domain

    def test_domain_sizes(self, sprinkler_net):
        assert sprinkler_net._domain["Rain"] == 2
        assert sprinkler_net._domain["WetGrass"] == 2

    def test_parents_stored(self, sprinkler_net):
        assert sprinkler_net._parents["Rain"] == []
        assert sprinkler_net._parents["Sprinkler"] == ["Rain"]
        assert sprinkler_net._parents["WetGrass"] == ["Rain", "Sprinkler"]

    def test_cpt_shapes(self, sprinkler_net):
        assert sprinkler_net._cpt["Rain"].shape == (2,)
        assert sprinkler_net._cpt["Sprinkler"].shape == (2, 2)
        assert sprinkler_net._cpt["WetGrass"].shape == (2, 2, 2)


# ---------------------------------------------------------------------------
# Inference — query output contracts
# ---------------------------------------------------------------------------

class TestBNQueryContracts:
    def test_query_sums_to_one(self, sprinkler_net):
        proba = sprinkler_net.query("Rain")
        np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-6)

    def test_query_with_evidence_sums_to_one(self, sprinkler_net):
        proba = sprinkler_net.query("Rain", evidence={"WetGrass": 1})
        np.testing.assert_allclose(proba.sum(), 1.0, atol=1e-6)

    def test_query_proba_non_negative(self, sprinkler_net):
        proba = sprinkler_net.query("Sprinkler")
        assert np.all(proba >= 0)

    def test_query_shape(self, sprinkler_net):
        proba = sprinkler_net.query("Rain")
        assert proba.shape == (2,)

    def test_query_observed_variable_returns_degenerate(self, sprinkler_net):
        """Querying the observed variable itself should give a point mass."""
        proba = sprinkler_net.query("Rain", evidence={"Rain": 1})
        np.testing.assert_allclose(proba[1], 1.0, atol=1e-6)
        np.testing.assert_allclose(proba[0], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Inference — numerical correctness
# ---------------------------------------------------------------------------

class TestBNQueryCorrectness:
    def test_prior_rain(self, sprinkler_net):
        """P(Rain=T) = 0.2 by definition."""
        proba = sprinkler_net.query("Rain")
        np.testing.assert_allclose(proba[1], 0.2, atol=1e-5)

    def test_prior_sprinkler_marginal(self, sprinkler_net):
        """
        P(S=T) = P(S=T|R=F)P(R=F) + P(S=T|R=T)P(R=T)
               = 0.4*0.8 + 0.01*0.2 = 0.322
        """
        proba = sprinkler_net.query("Sprinkler")
        np.testing.assert_allclose(proba[1], 0.322, atol=1e-4)

    def test_conditional_sprinkler_given_rain(self, sprinkler_net):
        """P(S=T | R=T) = 0.01 (directly from CPT)."""
        proba = sprinkler_net.query("Sprinkler", evidence={"Rain": 1})
        np.testing.assert_allclose(proba[1], 0.01, atol=1e-5)

    def test_explaining_away(self, sprinkler_net):
        """
        Explaining away: knowing rain makes the sprinkler less likely.
        P(S=T|W=T, R=T) < P(S=T|W=T)
        """
        p_s_given_wet = sprinkler_net.query("Sprinkler", evidence={"WetGrass": 1})
        p_s_given_wet_and_rain = sprinkler_net.query(
            "Sprinkler", evidence={"WetGrass": 1, "Rain": 1}
        )
        assert p_s_given_wet_and_rain[1] < p_s_given_wet[1]

    def test_wet_grass_more_likely_given_rain(self, sprinkler_net):
        p_wet_no_rain = sprinkler_net.query("WetGrass", evidence={"Rain": 0})
        p_wet_rain    = sprinkler_net.query("WetGrass", evidence={"Rain": 1})
        assert p_wet_rain[1] > p_wet_no_rain[1]

    def test_chain_d_separation(self, chain_net):
        """
        In chain A→B→C, C⊥A | B (d-separation).
        P(C | B=b) should not change when we also condition on A.
        """
        p_c_given_b0 = chain_net.query("C", evidence={"B": 0})
        p_c_given_b0_a0 = chain_net.query("C", evidence={"B": 0, "A": 0})
        p_c_given_b0_a1 = chain_net.query("C", evidence={"B": 0, "A": 1})
        np.testing.assert_allclose(p_c_given_b0, p_c_given_b0_a0, atol=1e-6)
        np.testing.assert_allclose(p_c_given_b0, p_c_given_b0_a1, atol=1e-6)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestBNSampling:
    def test_single_sample_has_all_variables(self, sprinkler_net):
        sample = sprinkler_net.sample(random_state=0)
        assert set(sample.keys()) == {"Rain", "Sprinkler", "WetGrass"}

    def test_sample_values_in_domain(self, sprinkler_net):
        samples = sprinkler_net.sample(n_samples=50, random_state=0)
        for s in samples:
            for var, val in s.items():
                assert 0 <= val < sprinkler_net._domain[var]

    def test_sample_reproducible(self, sprinkler_net):
        s1 = sprinkler_net.sample(n_samples=10, random_state=7)
        s2 = sprinkler_net.sample(n_samples=10, random_state=7)
        assert s1 == s2

    def test_marginal_from_samples_approx_query(self, sprinkler_net):
        """Empirical Rain=T frequency should approximate P(Rain=T)=0.2."""
        samples = sprinkler_net.sample(n_samples=5000, random_state=0)
        rain_true = sum(1 for s in samples if s["Rain"] == 1) / len(samples)
        assert abs(rain_true - 0.2) < 0.05

    def test_sample_sprinkler_given_no_rain(self, chain_net):
        """Sample from chain; empirical P(B=1|A=0) ≈ 0.3."""
        samples = chain_net.sample(n_samples=3000, random_state=0)
        b1_given_a0 = [s["B"] for s in samples if s["A"] == 0]
        if b1_given_a0:
            emp = sum(b1_given_a0) / len(b1_given_a0)
            assert abs(emp - 0.3) < 0.1
