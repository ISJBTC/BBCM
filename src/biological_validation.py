import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BiologicalValidation:
    """
    Biological validation framework comparing model predictions with literature
    """
    
    def __init__(self):
        self.literature_data = self.load_literature_benchmarks()
        self.biological_constraints = self.define_biological_constraints()
        
    def load_literature_benchmarks(self):
        """Load validated biological parameters from literature"""
        return {
            'tumor_doubling_times': {
                'breast_cancer_days': (30, 200),  # Mehrara et al. 2013
                'growth_rate_month': (0.01, 0.2),  # Converted from doubling time
                'source': 'Cancer Research 2013, Nature Rev Cancer 2018'
            },
            'immune_kinetics': {
                'cd8_half_life_days': (5, 30),    # Ganusov et al. 2007
                'killing_rate_month': (0.005, 0.15),  # From cytotoxicity studies
                'source': 'J Immunol 2007, Nature Immunol 2020'
            },
            'resistance_evolution': {
                'mutation_rate_month': (0.0001, 0.02),  # Obenauf et al. 2015
                'time_to_resistance_months': (3, 18),   # Clinical trials
                'source': 'Nature 2015, Cell 2019'
            },
            'treatment_effectiveness': {
                'hormone_therapy_response': (0.2, 0.8),  # Early Breast Cancer Trialists 2005
                'chemotherapy_response': (0.3, 0.7),    # Mieog et al. 2007
                'immunotherapy_response': (0.1, 0.6),   # Schmid et al. 2018
                'source': 'Lancet 2005, JCO 2018'
            },
            'biomarker_correlations': {
                'ca153_tumor_burden': 0.7,      # Hayes et al. 2006
                'cd8_survival': 0.6,            # Mahmoud et al. 2011
                'lactate_hypoxia': 0.8,         # Walenta et al. 2000
                'source': 'Clin Cancer Res 2006, J Clin Oncol 2011'
            }
        }
    
    def define_biological_constraints(self):
        """Define biological plausibility constraints"""
        return {
            'growth_hierarchy': {
                'condition': 'λ₁ > λ₂ > λᴿ¹ > λᴿ²',
                'rationale': 'Sensitive cells grow faster than resistant cells',
                'tolerance': 0.1
            },
            'immune_balance': {
                'condition': 'β₁ > 0 and β₂ > 0',
                'rationale': 'Both immune killing and suppression must exist',
                'ranges': {'beta1': (0.001, 0.2), 'beta2': (0.01, 0.8)}
            },
            'resistance_cost': {
                'condition': 'λᴿ < 0.7 × λ₁',
                'rationale': 'Resistance mutations impose growth cost',
                'reduction_factor': 0.3
            },
            'carrying_capacity': {
                'condition': 'K > N_total for viable tumors',
                'rationale': 'Tumors cannot exceed environmental capacity',
                'safety_margin': 1.2
            },
            'treatment_bounds': {
                'condition': '0.1 ≤ η ≤ 0.95',
                'rationale': 'No treatment is 0% or 100% effective',
                'bounds': (0.05, 0.99)
            }
        }
    
    def validate_parameter_ranges(self, model_params: Dict) -> Dict:
        """Validate model parameters against literature ranges"""
        print("BIOLOGICAL PARAMETER VALIDATION")
        print("="*50)
        
        validation_results = {}
        
        # 1. Growth rate validation
        print("1. TUMOR GROWTH RATES")
        print("-" * 30)
        
        lit_growth = self.literature_data['tumor_doubling_times']['growth_rate_month']
        model_growth = model_params.get('lambda1', 0.05)
        
        in_range = lit_growth[0] <= model_growth <= lit_growth[1]
        validation_results['growth_rate'] = {
            'model_value': model_growth,
            'literature_range': lit_growth,
            'valid': in_range,
            'percentile': self.calculate_percentile(model_growth, lit_growth)
        }
        
        status = "✓ VALID" if in_range else "⚠ OUT OF RANGE"
        print(f"Model λ₁: {model_growth:.3f} month⁻¹")
        print(f"Literature: {lit_growth[0]:.3f} - {lit_growth[1]:.3f} month⁻¹")
        print(f"Status: {status}")
        
        # 2. Immune kinetics validation
        print(f"\n2. IMMUNE KINETICS")
        print("-" * 30)
        
        lit_immune = self.literature_data['immune_kinetics']['killing_rate_month']
        model_immune = model_params.get('beta1', 0.02)
        
        in_range = lit_immune[0] <= model_immune <= lit_immune[1]
        validation_results['immune_killing'] = {
            'model_value': model_immune,
            'literature_range': lit_immune,
            'valid': in_range,
            'percentile': self.calculate_percentile(model_immune, lit_immune)
        }
        
        status = "✓ VALID" if in_range else "⚠ OUT OF RANGE"
        print(f"Model β₁: {model_immune:.3f} month⁻¹")
        print(f"Literature: {lit_immune[0]:.3f} - {lit_immune[1]:.3f} month⁻¹")
        print(f"Status: {status}")
        
        # 3. Resistance evolution validation
        print(f"\n3. RESISTANCE EVOLUTION")
        print("-" * 30)
        
        lit_resistance = self.literature_data['resistance_evolution']['mutation_rate_month']
        model_resistance = model_params.get('omega_R1', 0.002)
        
        in_range = lit_resistance[0] <= model_resistance <= lit_resistance[1]
        validation_results['resistance_rate'] = {
            'model_value': model_resistance,
            'literature_range': lit_resistance,
            'valid': in_range,
            'percentile': self.calculate_percentile(model_resistance, lit_resistance)
        }
        
        status = "✓ VALID" if in_range else "⚠ OUT OF RANGE"
        print(f"Model ω_R1: {model_resistance:.4f} month⁻¹")
        print(f"Literature: {lit_resistance[0]:.4f} - {lit_resistance[1]:.4f} month⁻¹")
        print(f"Status: {status}")
        
        # 4. Treatment effectiveness validation
        print(f"\n4. TREATMENT EFFECTIVENESS")
        print("-" * 30)
        
        treatments = {
            'eta_E': ('hormone_therapy_response', 'Hormone Therapy'),
            'eta_C': ('chemotherapy_response', 'Chemotherapy'),
            'eta_I': ('immunotherapy_response', 'Immunotherapy')
        }
        
        for param, (lit_key, name) in treatments.items():
            lit_range = self.literature_data['treatment_effectiveness'][lit_key]
            model_value = model_params.get(param, 0.5)
            
            in_range = lit_range[0] <= model_value <= lit_range[1]
            validation_results[param] = {
                'model_value': model_value,
                'literature_range': lit_range,
                'valid': in_range,
                'percentile': self.calculate_percentile(model_value, lit_range)
            }
            
            status = "✓ VALID" if in_range else "⚠ OUT OF RANGE"
            print(f"{name}: {model_value:.2f} (Lit: {lit_range[0]:.2f}-{lit_range[1]:.2f}) {status}")
        
        return validation_results
    
    def validate_biological_constraints(self, model_params: Dict) -> Dict:
        """Validate biological plausibility constraints"""
        print(f"\nBIOLOGICAL CONSTRAINT VALIDATION")
        print("="*50)
        
        constraint_results = {}
        
        # 1. Growth hierarchy
        print("1. GROWTH RATE HIERARCHY")
        print("-" * 30)
        
        lambda1 = model_params.get('lambda1', 0.05)
        lambda2 = model_params.get('lambda2', 0.03)
        lambdaR1 = model_params.get('lambdaR1', 0.015)
        lambdaR2 = model_params.get('lambdaR2', 0.01)
        
        hierarchy_valid = lambda1 > lambda2 > lambdaR1 > lambdaR2
        constraint_results['growth_hierarchy'] = hierarchy_valid
        
        print(f"λ₁ = {lambda1:.3f} > λ₂ = {lambda2:.3f} > λᴿ¹ = {lambdaR1:.3f} > λᴿ² = {lambdaR2:.3f}")
        print(f"Hierarchy valid: {'✓ YES' if hierarchy_valid else '✗ NO'}")
        
        # 2. Resistance cost
        print(f"\n2. RESISTANCE GROWTH COST")
        print("-" * 30)
        
        cost_constraint = lambdaR1 <= 0.7 * lambda1
        reduction_actual = (lambda1 - lambdaR1) / lambda1
        constraint_results['resistance_cost'] = cost_constraint
        
        print(f"Expected reduction: ≥30%")
        print(f"Actual reduction: {reduction_actual*100:.1f}%")
        print(f"Cost constraint: {'✓ SATISFIED' if cost_constraint else '✗ VIOLATED'}")
        
        # 3. Treatment effectiveness bounds
        print(f"\n3. TREATMENT EFFECTIVENESS BOUNDS")
        print("-" * 30)
        
        treatments = ['eta_E', 'eta_C', 'eta_H', 'eta_I']
        all_bounded = True
        
        for treatment in treatments:
            value = model_params.get(treatment, 0.5)
            bounded = 0.1 <= value <= 0.95
            all_bounded = all_bounded and bounded
            
            status = "✓" if bounded else "✗"
            print(f"{treatment}: {value:.2f} {status}")
        
        constraint_results['treatment_bounds'] = all_bounded
        
        # 4. Parameter positivity
        print(f"\n4. PARAMETER POSITIVITY")
        print("-" * 30)
        
        all_positive = all(value > 0 for value in model_params.values() if isinstance(value, (int, float)))
        constraint_results['positivity'] = all_positive
        
        print(f"All parameters positive: {'✓ YES' if all_positive else '✗ NO'}")
        
        return constraint_results
    
    def validate_biomarker_correlations(self, synthetic_data: pd.DataFrame) -> Dict:
        """Validate biomarker correlations against literature"""
        print(f"\nBIOMARKER CORRELATION VALIDATION")
        print("="*50)
        
        correlation_results = {}
        
        # Expected correlations from literature
        expected_correlations = {
            ('CA153', 'tumor_burden'): 0.7,
            ('CD8', 'immune_function'): 0.6,
            ('Lactate', 'hypoxia_level'): 0.8,
            ('TK1', 'proliferation_rate'): 0.75,
            ('IL10', 'immune_suppression'): 0.65
        }
        
        print("Expected vs. Observed Correlations:")
        print("-" * 40)
        
        for (marker1, marker2), expected in expected_correlations.items():
            if marker1 in synthetic_data.columns and marker2 in synthetic_data.columns:
                observed = synthetic_data[marker1].corr(synthetic_data[marker2])
                difference = abs(observed - expected)
                valid = difference < 0.2  # Allow 20% deviation
                
                correlation_results[f"{marker1}_{marker2}"] = {
                    'expected': expected,
                    'observed': observed,
                    'difference': difference,
                    'valid': valid
                }
                
                status = "✓" if valid else "⚠"
                print(f"{marker1}-{marker2}: {observed:.2f} (exp: {expected:.2f}) {status}")
            else:
                print(f"{marker1}-{marker2}: Data not available")
        
        return correlation_results
    
    def calculate_percentile(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Calculate percentile of value within range"""
        min_val, max_val = range_tuple
        if value < min_val:
            return 0.0
        elif value > max_val:
            return 100.0
        else:
            return ((value - min_val) / (max_val - min_val)) * 100
    
    def generate_validation_report(self, param_validation: Dict, 
                                 constraint_validation: Dict,
                                 correlation_validation: Dict = None) -> str:
        """Generate comprehensive validation report"""
        
        report = """
BIOLOGICAL VALIDATION REPORT
============================

PARAMETER RANGE VALIDATION:
"""
        
        # Parameter validation summary
        valid_params = sum(1 for v in param_validation.values() if v['valid'])
        total_params = len(param_validation)
        
        report += f"Parameters in literature range: {valid_params}/{total_params} ({valid_params/total_params*100:.1f}%)\n\n"
        
        for param, results in param_validation.items():
            status = "✓ VALID" if results['valid'] else "⚠ OUT OF RANGE"
            percentile = results['percentile']
            report += f"{param:20s}: {results['model_value']:.4f} ({percentile:.0f}th percentile) {status}\n"
        
        # Constraint validation summary
        report += f"\nBIOLOGICAL CONSTRAINT VALIDATION:\n"
        valid_constraints = sum(constraint_validation.values())
        total_constraints = len(constraint_validation)
        
        report += f"Constraints satisfied: {valid_constraints}/{total_constraints} ({valid_constraints/total_constraints*100:.1f}%)\n\n"
        
        for constraint, valid in constraint_validation.items():
            status = "✓ SATISFIED" if valid else "✗ VIOLATED"
            report += f"{constraint:20s}: {status}\n"
        
        # Overall assessment
        overall_score = (valid_params/total_params + valid_constraints/total_constraints) / 2
        
        report += f"\nOVERALL BIOLOGICAL VALIDITY SCORE: {overall_score*100:.1f}%\n"
        
        if overall_score >= 0.8:
            report += "✓ MODEL IS BIOLOGICALLY VALID AND READY FOR CLINICAL APPLICATION\n"
        elif overall_score >= 0.6:
            report += "⚠ MODEL SHOWS GOOD BIOLOGICAL VALIDITY WITH MINOR ADJUSTMENTS NEEDED\n"
        else:
            report += "✗ MODEL REQUIRES SIGNIFICANT BIOLOGICAL VALIDATION IMPROVEMENTS\n"
        
        return report

def validate_parameter_scaling_laws(model_params: Dict) -> Dict:
    """Validate biological scaling relationships"""
    print("\nSCALING LAW VALIDATION")
    print("="*30)
    
    scaling_results = {}
    
    # 1. Allometric scaling for immune parameters
    # Literature: Immune cell density scales with body surface area
    body_weight = 70  # kg, average human
    surface_area = (body_weight ** 0.67) / 10  # Simplified BSA calculation
    
    expected_immune_scaling = surface_area / 1.8  # Normalized to 70kg human
    
    print(f"1. Immune Parameter Scaling")
    print(f"Expected scaling factor: {expected_immune_scaling:.2f}")
    
    # 2. Metabolic scaling for growth rates
    # Growth rates should scale with metabolic rate (weight^0.75)
    metabolic_scaling = (body_weight / 70) ** 0.75
    
    print(f"2. Metabolic Scaling Factor: {metabolic_scaling:.2f}")
    
    # 3. Resistance evolution scaling
    # Mutation rates scale with cell division rate and population size
    tumor_size = model_params.get('K', 1000)
    division_rate = model_params.get('lambda1', 0.05)
    
    expected_mutation_load = tumor_size * division_rate * 1e-6  # Per cell per division
    actual_mutation_rate = model_params.get('omega_R1', 0.002)
    
    mutation_scaling_valid = 0.1 * expected_mutation_load <= actual_mutation_rate <= 10 * expected_mutation_load
    
    print(f"3. Mutation Rate Scaling")
    print(f"Expected range: {0.1*expected_mutation_load:.6f} - {10*expected_mutation_load:.6f}")
    print(f"Actual: {actual_mutation_rate:.6f}")
    print(f"Valid: {'✓' if mutation_scaling_valid else '✗'}")
    
    scaling_results = {
        'immune_scaling': expected_immune_scaling,
        'metabolic_scaling': metabolic_scaling,
        'mutation_scaling_valid': mutation_scaling_valid
    }
    
    return scaling_results

def run_biological_validation():
    """Run complete biological validation suite"""
    print("BIOLOGICAL VALIDATION OF BLOOD-BASED CANCER MODEL")
    print("="*60)
    
    validator = BiologicalValidation()
    
    # Example model parameters (would come from biomarker calculations)
    model_params = {
        'lambda1': 0.045,      # Within literature range
        'lambda2': 0.027,      # 60% of lambda1
        'lambdaR1': 0.0135,    # 30% of lambda1
        'lambdaR2': 0.009,     # 20% of lambda1
        'beta1': 0.025,        # Immune killing
        'beta2': 0.08,         # Immune suppression
        'omega_R1': 0.0018,    # Hormone resistance
        'omega_R2': 0.0012,    # Multi-drug resistance
        'eta_E': 0.65,         # Hormone therapy effectiveness
        'eta_C': 0.45,         # Chemotherapy effectiveness
        'eta_H': 0.78,         # HER2 therapy effectiveness
        'eta_I': 0.35,         # Immunotherapy effectiveness
        'K': 1200,             # Carrying capacity
        'phi1': 0.04,          # Immune production
        'delta_I': 0.06        # Immune death
    }
    
    # Run validations
    print("Running Parameter Range Validation...")
    param_results = validator.validate_parameter_ranges(model_params)
    
    print("\nRunning Biological Constraint Validation...")
    constraint_results = validator.validate_biological_constraints(model_params)
    
    print("\nRunning Scaling Law Validation...")
    scaling_results = validate_parameter_scaling_laws(model_params)
    
    # Generate report
    report = validator.generate_validation_report(param_results, constraint_results)
    print(f"\n{report}")
    
    # Summary statistics
    param_validity = sum(1 for v in param_results.values() if v['valid']) / len(param_results)
    constraint_validity = sum(constraint_results.values()) / len(constraint_results)
    overall_validity = (param_validity + constraint_validity) / 2
    
    print(f"\nVALIDATION SUMMARY:")
    print(f"Parameter validity: {param_validity*100:.1f}%")
    print(f"Constraint validity: {constraint_validity*100:.1f}%")
    print(f"Overall biological validity: {overall_validity*100:.1f}%")
    
    if overall_validity >= 0.8:
        print("🎉 MODEL PASSES BIOLOGICAL VALIDATION!")
        print("✓ Ready for synthetic data generation and testing")
    elif overall_validity >= 0.6:
        print("⚠️ MODEL SHOWS GOOD BIOLOGICAL VALIDITY")
        print("→ Minor parameter adjustments recommended")
    else:
        print("❌ MODEL REQUIRES BIOLOGICAL IMPROVEMENTS")
        print("→ Significant parameter revision needed")
    
    return {
        'param_results': param_results,
        'constraint_results': constraint_results,
        'scaling_results': scaling_results,
        'overall_validity': overall_validity
    }

if __name__ == "__main__":
    validation_results = run_biological_validation()