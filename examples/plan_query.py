from activegraph_memory.object_types import MemoryQuery
from activegraph_memory.planner import plan_query


def main() -> None:
    query = MemoryQuery(query="What is the latest ActiveGraph launch plan?")
    plan = plan_query(query, query_id="example_query")
    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
