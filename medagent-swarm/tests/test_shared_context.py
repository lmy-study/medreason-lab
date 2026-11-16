from medagent_swarm.orchestration.shared_context import SubTask, TaskStatus

def test_subtask_status_transitions():
    item = SubTask(id="t1", type="risk", description="risk assessment", assigned_agent="diagnostic_agent")
    assert item.can_be_executed()
    item.start()
    assert item.status == TaskStatus.IN_PROGRESS
    item.fail("timeout")
    assert item.status == TaskStatus.FAILED
