from medagent_swarm.context.budget import TokenBudget
from medagent_swarm.context.assembler import ContextAssembler
from medagent_swarm.context.compressor import StructuredMemoryCompressor
from medagent_swarm.context.selector import ContextSelector

def test_budget_limits_context():
    budget = TokenBudget(max_input_tokens=350, reserve_tokens=50)
    assert budget.estimate(budget.fit("历史"*1000, 20)) <= 20

def test_deduplicates_while_preserving_patient_information():
    items = [{"content":"用户报告青霉素过敏"}, {"content":"用户报告青霉素过敏"}, {"content":"普通随访"}]
    selected = ContextSelector().select(items, "健康评估", "diagnostic_agent")
    assert len(selected) == 2
    assert selected[0]["protected"]

def test_summary_marks_reported_not_confirmed():
    result = StructuredMemoryCompressor().compress([{"role":"user", "content":"我有高血压病史，服用降压药。"}])
    assert all(fact["status"] == "user_reported" for group in result.values() for fact in group)

def test_context_assembler_preserves_current_question():
    messages = ContextAssembler(max_input_tokens=700).build(
        "You are a medical assistant.", "有什么风险？",
        history=[{"role":"user", "content":"我有高血压病史"}], agent_id="diagnostic_agent")
    assert messages[-1] == {"role":"user", "content":"有什么风险？"}
