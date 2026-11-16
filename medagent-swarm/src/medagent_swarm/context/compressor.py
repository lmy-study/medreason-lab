"""Conservative structured extraction of explicitly stated patient facts."""
import re

class StructuredMemoryCompressor:
    PATTERNS = {
        'allergies': r'[^。；\n]{0,40}(?:过敏|allerg(?:y|ic))[^。；\n]{0,50}',
        'medications': r'[^。；\n]{0,40}(?:服用|用药|medication)[^。；\n]{0,50}',
        'conditions': r'[^。；\n]{0,40}(?:病史|诊断|高血压|糖尿病)[^。；\n]{0,50}',
    }
    def compress(self, messages):
        facts = {key: [] for key in self.PATTERNS}
        for idx, message in enumerate(messages):
            if message.get('role') != 'user':
                continue
            value = str(message.get('content', ''))
            for category, pattern in self.PATTERNS.items():
                for fragment in re.findall(pattern, value, flags=re.IGNORECASE):
                    # A quotation is not a medical confirmation; preserve attribution.
                    facts[category].append({'text':fragment.strip(), 'source_message_index':idx, 'status':'user_reported'})
        return {key:values for key, values in facts.items() if values}
