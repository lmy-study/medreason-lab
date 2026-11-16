import asyncio
from medagent_swarm.core.skill_loader import discover_skills
from medagent_swarm.core.skill_registry import SkillRegistry, SkillParameter

def test_discovery_loads_all_skills():
    found = discover_skills()
    assert len(found) == 9
    assert len({entry["function_name"] for entry in found}) == 9

def test_registry_schema_and_execution():
    registry = SkillRegistry()
    async def echo(value: str):
        return {"value": value}
    registry.register("echo", echo, "Echo value", [SkillParameter("value","string","Value",True)])
    assert registry.to_openai_format()[0]["function"]["name"] == "echo"
    assert asyncio.run(registry.execute("echo",value="hello")) == {"value":"hello"}
