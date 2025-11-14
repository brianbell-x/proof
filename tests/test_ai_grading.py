import json
import pytest
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chat.agent import ProverAgent
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

log_file = os.path.join(os.path.dirname(__file__), '..', 'temp', 'ai_grading_log.txt')
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
def prover_agent():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set in environment")
    return ProverAgent(api_key, model="x-ai/grok-4-fast:online")


@pytest.fixture
def grading_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set in environment")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def load_grading_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'grading_prompt.md')
    with open(prompt_path, 'r') as f:
        return f.read()


def grade_analysis(grading_client, claim, prover_result):
    grading_prompt = load_grading_prompt()
    grading_input = f"""
CLAIM: {claim}

PROVER AGENT ANALYSIS:
{json.dumps(prover_result, indent=2)}

{grading_prompt}
"""

    try:
        response = grading_client.chat.completions.create(
            model="x-ai/grok-4-fast",
            messages=[{"role": "user", "content": grading_input}],
            temperature=0.1,
            max_tokens=1000
        )
        content = response.choices[0].message.content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            log_to_file(f"Failed to parse grading response: {e}")
            log_to_file(f"Raw response: {content}")
            return {
                "grade": "ERROR",
                "reasoning": f"Failed to parse grading response: {e}",
                "verdict_appropriateness": "UNKNOWN",
                "evidence_quality": "UNKNOWN"
            }
    except Exception as e:
        log_to_file(f"Grading API call failed: {e}")
        return {
            "grade": "ERROR",
            "reasoning": f"API call failed: {e}",
            "verdict_appropriateness": "UNKNOWN",
            "evidence_quality": "UNKNOWN"
        }


@pytest.mark.parametrize("claim_data", [
    pytest.param(claim, id=claim["claim"][:30]) for claim in json.load(open(os.path.join(os.path.dirname(__file__), 'fixtures', 'claims.json')))
])
def test_ai_graded_claim_analysis(prover_agent, grading_client, claim_data):
    result = prover_agent.prove_claim(claim_data["claim"], max_iterations=5)
    grade = grade_analysis(grading_client, claim_data["claim"], result)

    log_message = f"\n{'='*80}\n"
    log_message += f"Claim: {claim_data['claim']}\n"
    log_message += f"Prover Verdict: {result.get('verdict', 'NONE')}\n"
    log_message += f"Expected Verdict: {claim_data['expected_verdict']}\n"
    log_message += f"AI Grade: {grade.get('grade', 'UNKNOWN')}\n"
    log_message += f"Grading Reasoning: {grade.get('reasoning', 'NONE')}\n"
    log_message += f"Verdict Appropriateness: {grade.get('verdict_appropriateness', 'UNKNOWN')}\n"
    log_message += f"Evidence Quality: {grade.get('evidence_quality', 'UNKNOWN')}\n"
    log_message += f"Full Prover Result: {json.dumps(result, indent=2)}\n"
    log_message += f"{'='*80}"
    log_to_file(log_message)

    assert grade["grade"] in ["PASS", "FAIL"], f"Unexpected grade: {grade['grade']}"
    if grade["grade"] == "FAIL":
        pytest.fail(f"AI grading failed: {grade['reasoning']}")


def test_grading_prompt_structure(grading_client):
    mock_result = {
        "claim": "Test claim",
        "verdict": "PROVEN",
        "confidence": "HIGH",
        "reasoning_summary": "This is obviously true.",
        "assumptions": [],
        "evidence": [],
        "derivation": []
    }
    grade = grade_analysis(grading_client, "Test claim", mock_result)
    expected_fields = ["grade", "reasoning", "verdict_appropriateness", "evidence_quality"]
    for field in expected_fields:
        assert field in grade, f"Missing field in grade result: {field}"


if __name__ == '__main__':
    pytest.main([__file__])
