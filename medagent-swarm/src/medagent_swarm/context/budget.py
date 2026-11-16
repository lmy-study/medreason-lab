"""Token-aware context budgeting, including tool schemas and output reserve."""
import json
import math

try:
    import tiktoken
except ImportError:
    tiktoken = None

class TokenBudget:
    def __init__(self, max_input_tokens=8192, reserve_tokens=1024):
        self.max_input_tokens = max_input_tokens
        self.reserve_tokens = reserve_tokens
        self._encoder = tiktoken.get_encoding("cl100k_base") if tiktoken else None

    def estimate(self, text):
        value = str(text)
        if self._encoder:
            return len(self._encoder.encode(value))
        return math.ceil(len(value) / 2)  # conservative fallback for mixed Chinese/English

    def remaining(self, system='', tools=None, question=''):
        fixed = self.estimate(system) + self.estimate(question)
        if tools:
            fixed += self.estimate(json.dumps(tools, ensure_ascii=False))
        return max(0, self.max_input_tokens - self.reserve_tokens - fixed)

    def fit(self, text, budget):
        if budget <= 0:
            return ''
        text = str(text)
        if self.estimate(text) <= budget:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.estimate(text[:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]
