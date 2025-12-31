import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_generate_ap_strict_master_and_report():
    # Run the generator script to ensure report is fresh
    script = os.path.join(ROOT, 'scripts', 'generate_ap_strict_master.py')
    assert os.path.exists(script)
    subprocess.check_call(['python', script])

    report_file = os.path.join(ROOT, 'data', 'ap_strict_report.json')
    assert os.path.exists(report_file)
    with open(report_file, 'r', encoding='utf-8') as fh:
        report = json.load(fh)

    assert report.get('final_strict_count') == 463
    assert isinstance(report.get('strict_missing'), list)
    assert len(report.get('strict_missing')) == 0


def test_generate_frontend_station_list():
    gen_script = os.path.join(ROOT, 'scripts', 'generate_frontend_station_list.py')
    assert os.path.exists(gen_script)
    subprocess.check_call(['python', gen_script])

    out_file = os.path.join(ROOT, 'frontend', 'src', 'utils', 'stationMapping_ap_strict.json')
    assert os.path.exists(out_file)
    with open(out_file, 'r', encoding='utf-8') as fh:
        mapping = json.load(fh)
    assert len(mapping.get('stations', [])) == 463
