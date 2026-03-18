import json
import subprocess

passed = 0
failed = 0

for i in range(1, 101):
    input_path    = f'data/test_cases/inputs/test_{i:03d}.json'
    expected_path = f'data/test_cases/expected_outputs/test_{i:03d}.json'

    with open(input_path) as f:
        input_data = f.read()

    with open(expected_path) as f:
        expected = json.load(f)

    result = subprocess.run(
        ['python', 'solution/race_simulator.py'],
        input=input_data,
        capture_output=True,
        text=True
    )

    try:
        output = json.loads(result.stdout)
        if output['finishing_positions'] == expected['finishing_positions']:
            passed += 1
            print(f"PASS test_{i:03d}")
        else:
            failed += 1
            print(f"FAIL test_{i:03d}")
    except Exception as e:
        failed += 1
        print(f"ERROR test_{i:03d}: {e}")

print(f"\nTotal:  {passed + failed}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Score:  {passed}%")