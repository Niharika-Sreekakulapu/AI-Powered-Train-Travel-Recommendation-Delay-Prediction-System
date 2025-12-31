import pytest

from backend.app import compute_risk


def test_low_risk_case():
    r = compute_risk(predicted_delay_min=0.0, conf_lower=0.0, conf_upper=5.0, imputation_flag=False, distance_km=10)
    assert isinstance(r, dict)
    assert 'risk_score' in r and 0 <= r['risk_score'] <= 100
    assert r['confidence'] == 'High'
    assert r['advice'].startswith('✅') or r['advice'].startswith('⚠️')
    # Expect very low score for trivial delay
    assert r['risk_score'] <= 10


def test_high_risk_case():
    r = compute_risk(predicted_delay_min=120.0, conf_lower=0.0, conf_upper=100.0, imputation_flag=True, distance_km=600)
    assert isinstance(r, dict)
    assert 'risk_score' in r
    assert r['risk_score'] >= 80
    assert r['confidence'] in ['Low', 'Medium', 'High']
    assert r['advice'].startswith('❌') or r['advice'].startswith('⚠️')


def test_mode_thresholds_change_advice():
    # Keep predicted_delay moderate so that mode changes advice
    r_casual = compute_risk(predicted_delay_min=30.0, conf_lower=5.0, conf_upper=30.0, imputation_flag=False, distance_km=50, mode='casual')
    r_exam = compute_risk(predicted_delay_min=30.0, conf_lower=5.0, conf_upper=30.0, imputation_flag=False, distance_km=50, mode='exam')
    # Advice should be at least as strict or stricter for 'exam'
    assert r_casual['risk_score'] == r_exam['risk_score']
    # exam mode threshold lower than casual -> advice may be stricter
    adv_strength = {'✅ Recommended': 0, '⚠️ Consider alternatives if you have a tight connection': 1, '❌ Not recommended if you have a tight connection': 2}
    assert adv_strength.get(r_exam['advice'], 1) >= adv_strength.get(r_casual['advice'], 1)