from langgraph.checkpoint.memory import MemorySaver

# Central checkpointer instance shared across agent runs
memory_checkpointer = MemorySaver()


def get_checkpointer() -> MemorySaver:
    """Returns the shared conversation checkpointer instance."""
    return memory_checkpointer

