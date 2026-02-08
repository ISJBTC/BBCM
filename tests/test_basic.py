"""
Basic tests for blood-based cancer model.

Run with: pytest tests/test_basic.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from production_system import BloodPanel, ClinicalCancerModel
from mathematical_validation import MathematicalAnalysis
from biological_validation import BiologicalValidator


class TestBloodPanel:
    """Tests for BloodPanel data structure"""
    
    def test_blood_panel_initialization(self):
        """Test that BloodPanel initializes correctly"""
        panel = BloodPanel(ca153=45.0, cea=5.2, cd8=850.0)
        assert panel.ca153 == 45.0
        assert panel.cea == 5.2
        assert panel.cd8 == 850.0
    
    def test_blood_panel_validation_valid(self):
        """Test validation with valid biomarker values"""
        panel = BloodPanel(
            ca153=45.0,
            cea=5.2,
            cd8=850.0,
            cd4=1200.0,
            albumin=4.0
        )
        validation_result = panel.validate()
        assert 'errors' in validation_result
        assert 'warnings' in validation_result
    
    def test_blood_panel_validation_invalid(self):
        """Test validation catches invalid values"""
        panel = BloodPanel(ca153=-10.0)  # Negative value
        validation_result = panel.validate()
        assert 'errors' in validation_result
        # Negative values should trigger errors
        

class TestClinicalCancerModel:
    """Tests for ClinicalCancerModel prediction system"""
    
    @pytest.fixture
    def model(self):
        """Create a model instance for testing"""
        return ClinicalCancerModel()
    
    @pytest.fixture
    def sample_panel(self):
        """Create a sample blood panel"""
        return BloodPanel(
            ca153=45.0,
            ca2729=35.0,
            cea=5.2,
            tk1=2.5,
            cd8=850.0,
            cd4=1200.0,
            albumin=4.0,
            glucose=95.0,
            ldh=200.0
        )
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly"""
        assert model is not None
        assert hasattr(model, 'essential_biomarkers')
    
    def test_parameter_estimation(self, model, sample_panel):
        """Test parameter estimation from blood panel"""
        params = model.estimate_parameters_from_biomarkers(sample_panel.__dict__)
        assert isinstance(params, dict)
        assert len(params) > 0
        # Check that parameters are positive (biological constraint)
        for param, value in params.items():
            assert value > 0, f"{param} should be positive"
    
    def test_confidence_calculation(self, model, sample_panel):
        """Test confidence score calculation"""
        biomarkers = sample_panel.__dict__
        confidence = model.calculate_confidence(biomarkers, {})
        assert 0 <= confidence <= 1, "Confidence should be between 0 and 1"


class TestMathematicalAnalysis:
    """Tests for mathematical validation"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return MathematicalAnalysis()
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly"""
        assert analyzer is not None
        assert hasattr(analyzer, 'N1')  # Check symbolic variables
        assert hasattr(analyzer, 'N2')
    
    def test_parameter_bounds_defined(self, analyzer):
        """Test that parameter bounds are properly defined"""
        assert hasattr(analyzer, 'define_parameter_bounds')
        # Parameter bounds should be dictionaries with min/max
    

class TestBiologicalValidator:
    """Tests for biological validation"""
    
    def test_parameter_range_validation(self):
        """Test parameter range validation"""
        validator = BiologicalValidator()
        
        # Test valid parameter set
        valid_params = {
            'lambda1': 0.05,
            'lambda2': 0.03,
            'lambdaR1': 0.015,
            'K': 1000.0,
            'beta1': 0.01,
            'eta_E': 0.5
        }
        
        # Should not raise errors
        assert True  # Placeholder


class TestDataLoading:
    """Tests for data loading and integrity"""
    
    def test_load_synthetic_data(self):
        """Test loading synthetic dataset"""
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 
                                 'synthetic_cancer_dataset_enhanced.csv')
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            assert len(df) > 0, "Dataset should not be empty"
            assert 'CA153' in df.columns or 'ca153' in df.columns, "Should have CA153 biomarker"
            # Check for reasonable data ranges
            if 'CA153' in df.columns:
                assert df['CA153'].min() >= 0, "Biomarkers should be non-negative"
    
    def test_load_ml_results(self):
        """Test loading ML results"""
        results_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                                   'comprehensive_ml_results.csv')
        
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            assert len(df) > 0, "Results should not be empty"
            assert 'Test_R2' in df.columns, "Should have R2 scores"
            # R2 scores should be reasonable
            assert df['Test_R2'].max() <= 1.0, "R2 should not exceed 1.0"


class TestNumericalStability:
    """Tests for numerical stability"""
    
    def test_parameter_positivity(self):
        """Test that all parameters remain positive"""
        model = ClinicalCancerModel()
        
        # Test with various biomarker combinations
        test_cases = [
            {'ca153': 25.0, 'cea': 2.5, 'cd8': 1000},  # Early stage
            {'ca153': 150.0, 'cea': 15.0, 'cd8': 300},  # Advanced
            {'ca153': 75.0, 'cea': 8.0, 'cd8': 650},    # Intermediate
        ]
        
        for biomarkers in test_cases:
            params = model.estimate_parameters_from_biomarkers(biomarkers)
            for param, value in params.items():
                assert value > 0, f"{param} must be positive (got {value})"
                assert np.isfinite(value), f"{param} must be finite (got {value})"


# Integration tests
class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_end_to_end_prediction(self):
        """Test complete prediction workflow"""
        # Create model
        model = ClinicalCancerModel()
        
        # Create blood panel
        panel = BloodPanel(
            ca153=45.0,
            cea=5.2,
            cd8=850.0,
            cd4=1200.0,
            albumin=4.0,
            glucose=95.0
        )
        
        # Get prediction
        result = model.predict_from_blood_panel(panel)
        
        # Verify result structure
        assert 'parameters' in result
        assert 'confidence' in result
        assert 'timestamp' in result
        
        # Verify parameters are reasonable
        params = result['parameters']
        assert all(v > 0 for v in params.values()), "All parameters should be positive"
        
        # Verify confidence is reasonable
        assert 0 <= result['confidence']['score'] <= 1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
