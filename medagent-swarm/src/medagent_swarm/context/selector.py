"""Deduplicated, task-aware contextual evidence selection."""
from datetime import datetime, timezone
import hashlib

class ContextSelector:
    PROTECTED = ('过敏', '药物', '禁忌', 'allergy', 'medication', 'contraindication')
    FOCUS = {
        'diagnostic_agent': ('症状', '病史', '风险', '过敏', '药物'),
        'research_agent': ('指南', '文献', '证据', '研究', '来源'),
        'consultation_agent': ('建议', '生活', '治疗', '随访', '过敏'),
    }
    def select(self, items, question, agent_id=None):
        seen, ranked = set(), []
        focus = self.FOCUS.get(agent_id, ())
        query_terms = set(question.lower().split())
        for order, item in enumerate(items):
            text = str(item.get('content', '')).strip()
            if not text:
                continue
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            lower = text.lower()
            protected = any(term in lower for term in self.PROTECTED)
            task_relevance = sum(term in lower for term in focus)
            lexical = sum(t in lower for t in query_terms if len(t)>1)
            priority = (100 if protected else 0) + task_relevance * 5 + lexical * 2 + order / max(len(items),1)
            ranked.append((priority, order, dict(item, content=text, protected=protected)))
        return [x[2] for x in sorted(ranked, key=lambda x:(-x[0],-x[1]))]
