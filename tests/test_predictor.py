"""Tests for HairpinPredictor."""

import pytest
from ivt_hairpinstat import HairpinPredictor
from ivt_hairpinstat.core.predictor import PredictionResult, SaltConditions


class TestHairpinPredictor:
    @pytest.fixture
    def predictor(self):
        return HairpinPredictor()

    @pytest.fixture
    def tetraloop_seq(self):
        return "GCGCAAAAGCGC", "((((....))))"

    @pytest.fixture
    def triloop_seq(self):
        return "GCGCAAAGCGC", "((((...))))"

    def test_predict_returns_result(self, predictor, tetraloop_seq):
        seq, struct = tetraloop_seq
        result = predictor.predict(seq, struct)
        assert isinstance(result, PredictionResult)

    def test_predict_tetraloop_tm_range(self, predictor, tetraloop_seq):
        seq, struct = tetraloop_seq
        result = predictor.predict(seq, struct)
        assert 50 < result.Tm < 90

    def test_predict_triloop_tm_range(self, predictor, triloop_seq):
        seq, struct = triloop_seq
        result = predictor.predict(seq, struct)
        assert 50 < result.Tm < 95

    def test_predict_direct_tm_method(self, predictor, tetraloop_seq):
        seq, struct = tetraloop_seq
        result = predictor.predict(seq, struct)
        assert "direct_tm" in result.prediction_method

    def test_predict_structure_type_tetraloop(self, predictor, tetraloop_seq):
        seq, struct = tetraloop_seq
        result = predictor.predict(seq, struct)
        assert result.structure_type == "tetraloop"

    def test_predict_structure_type_triloop(self, predictor, triloop_seq):
        seq, struct = triloop_seq
        result = predictor.predict(seq, struct)
        assert result.structure_type == "triloop"

    def test_predict_gc_content(self, predictor, tetraloop_seq):
        seq, struct = tetraloop_seq
        result = predictor.predict(seq, struct)
        expected_gc = (seq.count("G") + seq.count("C")) / len(seq) * 100
        assert abs(result.gc_content - expected_gc) < 0.1

    def test_predict_salt_adjustment(self, predictor, tetraloop_seq):
        seq, struct = tetraloop_seq
        salt = SaltConditions(Na=0.05)
        result = predictor.predict(seq, struct, salt_conditions=salt)
        assert result.Tm_adjusted is not None
        assert result.Tm_adjusted != result.Tm

    def test_predict_batch(self, predictor):
        sequences = ["GCGCAAAAGCGC", "GCGCAAAGCGC", "GCGCAAAAAGCGC"]
        structures = ["((((....))))", "((((...))))", "((((.....))))"]
        results = predictor.predict_batch(sequences, structures)
        assert len(results) == 3
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_predict_to_dict(self, predictor, tetraloop_seq):
        seq, struct = tetraloop_seq
        result = predictor.predict(seq, struct)
        d = result.to_dict()
        assert "Tm" in d
        assert "dH" in d
        assert "dG_37" in d
        assert "structure_type" in d

    def test_get_feature_contributions(self, predictor, tetraloop_seq):
        seq, struct = tetraloop_seq
        report = predictor.get_feature_contributions(seq, struct)
        assert "features" in report
        assert "total_dH" in report
        assert "total_dG" in report
        assert len(report["features"]) > 0

    def test_get_model_info(self, predictor):
        info = predictor.get_model_info()
        assert "n_dH_coefficients" in info
        assert "n_dG_coefficients" in info
        assert info["n_dH_coefficients"] > 1000


class TestAutoModelSelection:
    @pytest.fixture
    def auto_predictor(self):
        try:
            return HairpinPredictor(auto_select_model=True)
        except ImportError:
            pytest.skip("GNN dependencies not available")

    def test_triloop_uses_linear(self, auto_predictor):
        result = auto_predictor.predict("GCGCAAAGCGC", "((((...))))")
        assert "direct_tm" in result.prediction_method

    def test_tetraloop_uses_linear(self, auto_predictor):
        result = auto_predictor.predict("GCGCAAAAGCGC", "((((....))))")
        assert "direct_tm" in result.prediction_method

    def test_pentaloop_uses_gnn(self, auto_predictor):
        result = auto_predictor.predict("GCGCAAAAAGCGC", "((((.....))))")
        assert result.prediction_method == "gnn"

    def test_hexaloop_uses_gnn(self, auto_predictor):
        result = auto_predictor.predict("GCGCAAAAAAGCGC", "((((......))))")
        assert result.prediction_method == "gnn"


class TestGNNPredictor:
    @pytest.fixture
    def gnn_predictor(self):
        try:
            return HairpinPredictor(use_gnn=True)
        except ImportError:
            pytest.skip("GNN dependencies not available")

    def test_gnn_predict(self, gnn_predictor):
        result = gnn_predictor.predict("GCGCAAAAGCGC", "((((....))))")
        assert result.prediction_method == "gnn"
        assert 50 < result.Tm < 100

    def test_gnn_batch_predict(self, gnn_predictor):
        sequences = ["GCGCAAAAGCGC", "GCGCAAAGCGC", "GCGCAAAAAGCGC"]
        structures = ["((((....))))", "((((...))))", "((((.....))))"]
        results = gnn_predictor.predict_batch(sequences, structures)
        assert len(results) == 3
        assert all(r.prediction_method == "gnn" for r in results)

    def test_gnn_thermodynamics(self, gnn_predictor):
        result = gnn_predictor.predict("GCGCAAAAGCGC", "((((....))))")
        assert result.dH < 0
        assert result.dG_37 < 0
        assert result.dS < 0


class TestThermodynamicCalculations:
    @pytest.fixture
    def thermo_predictor(self):
        return HairpinPredictor(use_direct_tm=False)

    def test_thermodynamic_tm(self, thermo_predictor):
        result = thermo_predictor.predict("GCGCAAAAGCGC", "((((....))))")
        assert result.prediction_method == "thermodynamic"
        assert result.dH < 0
        assert result.dG_37 < 0


class TestEnsemblePrediction:
    @pytest.fixture
    def ensemble_predictor(self):
        try:
            return HairpinPredictor(auto_select_model=True)
        except ImportError:
            pytest.skip("GNN dependencies not available")

    def test_ensemble_tetraloop(self, ensemble_predictor):
        result = ensemble_predictor.predict_ensemble("GCGCAAAAGCGC", "((((....))))")
        assert "ensemble" in result.prediction_method
        assert "0.7L" in result.prediction_method
        assert 50 < result.Tm < 90

    def test_ensemble_pentaloop(self, ensemble_predictor):
        result = ensemble_predictor.predict_ensemble("GCGCAAAAAGCGC", "((((.....))))")
        assert "ensemble" in result.prediction_method
        assert "0.7G" in result.prediction_method
        assert 50 < result.Tm < 100

    def test_ensemble_custom_weights(self, ensemble_predictor):
        result = ensemble_predictor.predict_ensemble(
            "GCGCAAAAGCGC", "((((....))))", weights=(0.5, 0.5)
        )
        assert "0.5L_0.5G" in result.prediction_method

    def test_ensemble_thermodynamics(self, ensemble_predictor):
        result = ensemble_predictor.predict_ensemble("GCGCAAAAGCGC", "((((....))))")
        assert result.dH < 0
        assert result.dG_37 < 0
        assert result.dS < 0
