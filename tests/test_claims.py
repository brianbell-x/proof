import json
import pytest
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.agent import ProverAgent

log_file = os.path.join(os.path.dirname(__file__), '..', 'temp', 'test_claims_log.txt')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

def log_to_file(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(formatted_message + '\n')


@pytest.fixture
def claims_data():
    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'claims.json')
    with open(fixture_path, 'r') as f:
        return json.load(f)


@pytest.fixture
def agent():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set in environment")
    return ProverAgent(api_key, model="x-ai/grok-4-fast")


@pytest.mark.parametrize("claim_data", [
    pytest.param(claim, id=claim["claim"][:30]) for claim in json.load(open(os.path.join(os.path.dirname(__file__), 'fixtures', 'claims.json')))
])
def test_claim_verdict(agent, claim_data):
    result = agent.prove_claim(claim_data["claim"], max_iterations=5)

    assert "verdict" in result, f"Missing verdict for claim: {claim_data['claim']}"
    valid_verdicts = ["PROVEN", "DISPROVEN", "UNSUPPORTED", "UNVERIFIABLE"]
    assert result["verdict"] in valid_verdicts, f"Invalid verdict '{result['verdict']}' for {claim_data['claim']}"

    assert "reasoning_summary" in result, "Missing reasoning summary"
    assert isinstance(result["reasoning_summary"], str), "Reasoning summary should be a string"

    assert "assumptions" in result, "Missing assumptions"
    assert isinstance(result["assumptions"], list), "Assumptions should be a list"
    assert "derivation" in result, "Missing derivation"
    assert isinstance(result["derivation"], list), "Derivation should be a list"
    assert "falsifiable_test" in result, "Missing falsifiable test"
    assert "iterations_used" in result, "Missing iterations_used"
    assert "tools_used" in result, "Missing tools_used"

    if claim_data["tools_needed"]:
        assert len(result["tools_used"]) > 0, f"Expected tools {claim_data['tools_needed']} but none used for {claim_data['claim']}"
        for expected_tool in claim_data["tools_needed"]:
            assert expected_tool in result["tools_used"], f"Expected tool '{expected_tool}' not used for {claim_data['claim']}"

    log_message = f"\n{'='*80}\n"
    log_message += f"Claim: {claim_data['claim']}\n"
    log_message += f"Verdict: {result['verdict']} (expected: {claim_data['expected_verdict']})\n"
    log_message += f"Tools used: {result['tools_used']}\n"
    log_message += f"Iterations: {result['iterations_used']}\n"
    log_message += f"Full result: {json.dumps(result, indent=2)}\n"
    log_message += f"{'='*80}"
    log_to_file(log_message)


def test_claim_structure_consistency(agent):
    result = agent.prove_claim("Water boils at 100Â°C at standard atmospheric pressure.")

    expected_keys = ["claim", "verdict", "assumptions", "derivation", "falsifiable_test", "iterations_used", "tools_used", "reasoning_summary"]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

    if result["derivation"]:
        for step in result["derivation"]:
            assert "step" in step, "Derivation step missing 'step' number"
            assert "principle" in step, "Derivation step missing 'principle'"


def test_error_handling():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY not found"):
        ProverAgent("", model="test-model")


if __name__ == '__main__':
    pytest.main([__file__])

