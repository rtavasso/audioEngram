"""
Tests for reproducibility across runs.

Verifies that same seed produces identical results.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase0.utils.seed import set_seed, get_rng
from phase0.clustering.vq import fit_kmeans, assign_clusters
from phase0.clustering.pca import fit_pca, project_pca
from phase0.clustering.baselines import permute_cluster_ids, create_random_clusters


class TestSeedReproducibility:
    """Tests for seed-based reproducibility."""

    def test_numpy_reproducibility(self):
        """Test that numpy random is reproducible with same seed."""
        set_seed(42)
        a1 = np.random.randn(100, 10)

        set_seed(42)
        a2 = np.random.randn(100, 10)

        assert np.allclose(a1, a2), "NumPy random not reproducible"

    def test_rng_reproducibility(self):
        """Test that get_rng produces reproducible results."""
        rng1 = get_rng(123)
        a1 = rng1.random(100)

        rng2 = get_rng(123)
        a2 = rng2.random(100)

        assert np.allclose(a1, a2), "get_rng not reproducible"

    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        a1 = np.random.randn(100, 10)

        set_seed(43)
        a2 = np.random.randn(100, 10)

        assert not np.allclose(a1, a2), "Different seeds should produce different results"


class TestKMeansReproducibility:
    """Tests for k-means reproducibility."""

    def test_kmeans_same_seed(self):
        """Test that k-means produces identical results with same seed."""
        np.random.seed(0)
        data = np.random.randn(500, 20).astype(np.float32)

        model1 = fit_kmeans(data, k=10, seed=42)
        model2 = fit_kmeans(data, k=10, seed=42)

        assert np.allclose(model1.centroids, model2.centroids), \
            "K-means centroids not reproducible"

    def test_kmeans_different_seed(self):
        """Test that k-means with different seeds may differ."""
        np.random.seed(0)
        data = np.random.randn(500, 20).astype(np.float32)

        model1 = fit_kmeans(data, k=10, seed=42)
        model2 = fit_kmeans(data, k=10, seed=43)

        # Centroids should likely be different (not guaranteed but very likely)
        assert not np.allclose(model1.centroids, model2.centroids), \
            "K-means with different seeds should typically differ"

    def test_cluster_assignment_reproducible(self):
        """Test that cluster assignments are reproducible."""
        np.random.seed(0)
        train_data = np.random.randn(500, 20).astype(np.float32)
        test_data = np.random.randn(100, 20).astype(np.float32)

        model = fit_kmeans(train_data, k=10, seed=42)

        ids1, dist1 = assign_clusters(test_data, model)
        ids2, dist2 = assign_clusters(test_data, model)

        assert np.array_equal(ids1, ids2), "Cluster assignments not reproducible"
        assert np.allclose(dist1, dist2), "Cluster distances not reproducible"


class TestPCAReproducibility:
    """Tests for PCA reproducibility."""

    def test_pca_same_seed(self):
        """Test that PCA produces identical results with same seed."""
        np.random.seed(0)
        data = np.random.randn(500, 50).astype(np.float32)

        model1 = fit_pca(data, n_components=10, seed=42)
        model2 = fit_pca(data, n_components=10, seed=42)

        # Components may have sign flips, so compare absolute values
        assert np.allclose(np.abs(model1.components), np.abs(model2.components)), \
            "PCA components not reproducible"

    def test_pca_projection_reproducible(self):
        """Test that PCA projection is reproducible."""
        np.random.seed(0)
        train_data = np.random.randn(500, 50).astype(np.float32)
        test_data = np.random.randn(100, 50).astype(np.float32)

        model = fit_pca(train_data, n_components=10, seed=42)

        proj1 = project_pca(test_data, model)
        proj2 = project_pca(test_data, model)

        assert np.allclose(proj1, proj2), "PCA projection not reproducible"


class TestBaselineReproducibility:
    """Tests for baseline reproducibility."""

    def test_random_clusters_reproducible(self):
        """Test that random cluster creation is reproducible."""
        ids1 = create_random_clusters(n_samples=1000, k=10, seed=42)
        ids2 = create_random_clusters(n_samples=1000, k=10, seed=42)

        assert np.array_equal(ids1, ids2), "Random clusters not reproducible"

    def test_permutation_reproducible(self):
        """Test that cluster permutation is reproducible."""
        np.random.seed(0)
        original_ids = np.random.randint(0, 10, 1000).astype(np.int32)

        permuted1 = permute_cluster_ids(original_ids, seed=42)
        permuted2 = permute_cluster_ids(original_ids, seed=42)

        assert np.array_equal(permuted1, permuted2), "Permutation not reproducible"

    def test_permutation_preserves_histogram(self):
        """Test that permutation preserves cluster size distribution."""
        np.random.seed(0)
        original_ids = np.random.randint(0, 10, 1000).astype(np.int32)

        permuted = permute_cluster_ids(original_ids, seed=42)

        original_hist = np.bincount(original_ids)
        permuted_hist = np.bincount(permuted)

        # Histograms should be identical (same counts per cluster)
        assert np.array_equal(
            np.sort(original_hist),
            np.sort(permuted_hist)
        ), "Permutation changed histogram"


class TestEndToEndReproducibility:
    """End-to-end reproducibility tests."""

    def test_full_pipeline_reproducible(self):
        """Test that the full feature-to-metric pipeline is reproducible."""
        from phase0.metrics.variance_ratio import compute_variance_ratio

        # Create synthetic data
        np.random.seed(0)
        data = np.random.randn(500, 20).astype(np.float32)

        # Run pipeline twice with same seed
        results = []
        for _ in range(2):
            set_seed(42)

            # Fit k-means
            model = fit_kmeans(data, k=10, seed=42)
            cluster_ids, _ = assign_clusters(data, model)

            # Compute metrics
            vr = compute_variance_ratio(data, cluster_ids)
            results.append(vr)

        assert results[0]["variance_ratio"] == results[1]["variance_ratio"], \
            "Variance ratio not reproducible"
        assert results[0]["sse_within"] == results[1]["sse_within"], \
            "SSE not reproducible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
