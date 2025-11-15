import unittest
import sys
import os

# Ensure tools are importable when running from tests directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tool.proof_tool import CodeExecutionTool


class TestCodeExecutionClaims(unittest.TestCase):
    def setUp(self):
        self.tool = CodeExecutionTool()

    def _exec(self, code: str):
        return self.tool.execute(code=code)

    def test_claim_float_sum_exact_equality(self):
        code = "result = (0.1 + 0.2) == 0.3\nprint(f'Exact equality: {result}')"
        result = self._exec(code)
        self.assertTrue(result["success"])
        self.assertIn("Exact equality: False", result["output"])

    def test_claim_2025_is_prime(self):
        code = (
            "import math\n"
            "def is_prime(n):\n"
            "    if n <= 1:\n"
            "        return False\n"
            "    if n % 2 == 0:\n"
            "        return n == 2\n"
            "    r = int(math.sqrt(n))\n"
            "    for i in range(3, r+1, 2):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
            "n = 2025\n"
            "claim_disproven = not is_prime(n)\n"
            "print(f'Claim disproven: {claim_disproven}')\n"
            "print(f'Example factor: 2025 % 3 == {2025 % 3}')\n"
        )
        result = self._exec(code)
        self.assertTrue(result["success"])
        self.assertIn("Claim disproven: True", result["output"])

    def test_claim_binary_search_is_linear_time(self):
        code = (
            "import math\n"
            "def binary_search(arr, target):\n"
            "    comps = 0\n"
            "    l, r = 0, len(arr)-1\n"
            "    while l <= r:\n"
            "        comps += 1\n"
            "        m = (l+r)//2\n"
            "        if arr[m] == target:\n"
            "            return m, comps\n"
            "        elif arr[m] < target:\n"
            "            l = m + 1\n"
            "        else:\n"
            "            r = m - 1\n"
            "    return -1, comps\n"
            "sizes = [100, 1000, 10000]\n"
            "ratios = []\n"
            "for n in sizes:\n"
            "    arr = list(range(n))\n"
            "    _, c = binary_search(arr, n-1)\n"
            "    ratios.append(c / math.log2(n))\n"
            "ok = all(r < 4.0 for r in ratios)\n"
            "print('Ratios:', [round(r,2) for r in ratios])\n"
            "print(f'Claim disproven: {ok}')\n"
        )
        result = self._exec(code)
        self.assertTrue(result["success"])
        self.assertIn("Claim disproven: True", result["output"])

    def test_claim_softmax_not_probability_distribution(self):
        code = (
            "import math\n"
            "def softmax(xs):\n"
            "    exps = [math.exp(x) for x in xs]\n"
            "    s = sum(exps)\n"
            "    return [e/s for e in exps]\n"
            "probs = softmax([2.0, 1.0, 0.1])\n"
            "sum_probs = sum(probs)\n"
            "all_pos = all(p > 0 for p in probs)\n"
            "all_lt1 = all(p < 1 for p in probs)\n"
            "ok = abs(sum_probs - 1.0) < 1e-6 and all_pos and all_lt1\n"
            "print('Sum:', round(sum_probs, 6))\n"
            "print(f'Claim disproven: {ok}')\n"
        )
        result = self._exec(code)
        self.assertTrue(result["success"])
        self.assertIn("Claim disproven: True", result["output"])

    def test_claim_sha256_collisions_are_trivial(self):
        code = (
            "import hashlib\n"
            "inputs = ['hello', 'world', 'test', 'crypto', 'hash', 'proof', 'engine', 'falsify']\n"
            "hashes = [hashlib.sha256(s.encode()).hexdigest() for s in inputs]\n"
            "unique = len(set(hashes)) == len(hashes)\n"
            "print(f'Inputs: {len(inputs)} Unique digests: {unique}')\n"
            "print(f'Claim disproven: {unique}')\n"
        )
        result = self._exec(code)
        self.assertTrue(result["success"])
        self.assertIn("Claim disproven: True", result["output"])

    def test_claim_tool_forgets_state_between_runs(self):
        r1 = self._exec("x = 42\nprint('set')")
        self.assertTrue(r1["success"])
        r2 = self._exec("print(x)\nprint('Claim disproven: ' + str(x == 42))")
        self.assertTrue(r2["success"])
        self.assertIn("42", r2["output"])
        self.assertIn("Claim disproven: True", r2["output"])

    def test_claim_division_by_zero_crashes_without_reporting(self):
        result = self._exec("1/0")
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("division by zero", result["error"].lower())

    def test_claim_tool_does_not_record_execution_metadata(self):
        result = self._exec("sum(range(1000))")
        self.assertTrue(result["success"])
        self.assertIn("timestamp", result)
        self.assertIsNotNone(result["timestamp"])
        self.assertIsNotNone(result["execution_time"])
        self.assertGreater(result["execution_time"], 0)

if __name__ == '__main__':
    unittest.main()
