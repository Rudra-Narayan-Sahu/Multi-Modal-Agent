import unittest
import uuid
from unittest.mock import patch
from app.database import init_db
from app.memory.conversation import conversation_manager
from app.memory.manager import memory_manager
from app.memory.retriever import memory_retriever
from app.memory.extractor import memory_extractor
from app.memory.schemas import MemoryCategoryEnum


class TestMemorySystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        uid = uuid.uuid4().hex[:8]
        self.user_alice = f"user_alice_{uid}"
        self.user_bob = f"user_bob_{uid}"
        self.conv_id = f"conv_test_{uid}"

    # 1. Creating a conversation
    def test_01_create_conversation(self):
        conv = conversation_manager.create_conversation(
            user_id=self.user_alice,
            title="Research on Multi-Agent Systems"
        )
        self.assertIsNotNone(conv["id"])
        self.assertEqual(conv["user_id"], self.user_alice)
        self.assertEqual(conv["title"], "Research on Multi-Agent Systems")

    # 2. Saving user and assistant messages
    def test_02_saving_messages(self):
        conversation_manager.add_turn(
            conversation_id=self.conv_id,
            user_id=self.user_alice,
            query="What is LangGraph?",
            response="LangGraph is an orchestration framework for cyclic agent workflows.",
            category="GENERAL_QUERY",
            agents_used=["response_agent"]
        )
        msgs = conversation_manager.get_messages(self.conv_id, self.user_alice)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "What is LangGraph?")
        self.assertEqual(msgs[1]["role"], "assistant")

    # 3. Retrieving conversation history
    def test_03_retrieving_conversation_history(self):
        conversation_manager.add_turn(
            conversation_id=self.conv_id,
            user_id=self.user_alice,
            query="History query",
            response="History response"
        )
        conv = conversation_manager.get_conversation(self.conv_id, self.user_alice)
        self.assertIsNotNone(conv)
        self.assertEqual(conv["id"], self.conv_id)
        self.assertEqual(len(conv["messages"]), 2)

    # 4. Continuing a previous conversation
    def test_04_continuing_conversation(self):
        conversation_manager.add_turn(self.conv_id, self.user_alice, "Turn 1", "Resp 1")
        conversation_manager.add_turn(self.conv_id, self.user_alice, "Turn 2", "Resp 2")
        conv = conversation_manager.get_conversation(self.conv_id, self.user_alice)
        self.assertEqual(len(conv["messages"]), 4)

    # 5. Restoring LangGraph checkpoints & short-term context
    def test_05_short_term_context_window(self):
        conversation_manager.add_turn(self.conv_id, self.user_alice, "What is LangGraph?", "LangGraph is framework.")
        conversation_manager.add_turn(self.conv_id, self.user_alice, "Explain its state graph concept.", "StateGraph manages state.")
        context = conversation_manager.format_history_context(self.conv_id, self.user_alice, max_turns=2)
        self.assertIn("User: What is LangGraph?", context)
        self.assertIn("User: Explain its state graph concept.", context)

    # 6. Saving long-term memory
    def test_06_save_long_term_memory(self):
        mem = memory_manager.save_memory(
            user_id=self.user_alice,
            content="User prefers Python 3.13 with strict type annotations.",
            category="user_preferences",
            importance=4
        )
        self.assertIsNotNone(mem)
        self.assertEqual(mem["category"], "user_preferences")
        self.assertEqual(mem["importance"], 4)

    # 7. Retrieving relevant memories
    def test_07_retrieve_relevant_memories(self):
        memory_manager.save_memory(
            user_id=self.user_alice,
            content="User prefers Python 3.13 with strict type annotations.",
            category="user_preferences",
            importance=4
        )
        retrieved = memory_retriever.retrieve_relevant_memories(
            user_id=self.user_alice,
            query="Write Python code for text chunking"
        )
        self.assertTrue(len(retrieved) > 0)
        self.assertIn("Python 3.13", retrieved[0]["content"])

    # 8. Updating existing memory
    def test_08_update_existing_memory(self):
        mem = memory_manager.save_memory(
            user_id=self.user_alice,
            content="Old content to update",
            category="user_preferences",
            importance=3
        )
        updated = memory_manager.update_memory(
            memory_id=mem["id"],
            user_id=self.user_alice,
            content="User prefers Python 3.13 and FastAPI.",
            importance=5
        )
        self.assertEqual(updated["content"], "User prefers Python 3.13 and FastAPI.")
        self.assertEqual(updated["importance"], 5)

    # 9. Duplicate memory detection & merge
    def test_09_duplicate_memory_merge(self):
        mem1 = memory_manager.save_memory(
            user_id=self.user_alice,
            content="Project uses Qdrant Cloud as vector store.",
            category="project_context",
            importance=3
        )
        # Attempt to save duplicate with higher importance
        mem2 = memory_manager.save_memory(
            user_id=self.user_alice,
            content="Project uses Qdrant Cloud as vector store.",
            category="project_context",
            importance=5
        )
        self.assertEqual(mem1["id"], mem2["id"])
        self.assertEqual(mem2["importance"], 5)

    # 10. Deleting memory
    def test_10_delete_memory(self):
        mem = memory_manager.save_memory(
            user_id=self.user_alice,
            content="Temporary note to delete.",
            category="explicit_notes"
        )
        deleted = memory_manager.delete_memory(mem["id"], self.user_alice)
        self.assertTrue(deleted)

    # 11. Deleting a conversation
    def test_11_delete_conversation(self):
        temp_cid = "conv_to_delete"
        conversation_manager.create_conversation(self.user_alice, "To Delete", conversation_id=temp_cid)
        deleted = conversation_manager.delete_conversation(temp_cid, self.user_alice)
        self.assertTrue(deleted)
        self.assertIsNone(conversation_manager.get_conversation(temp_cid, self.user_alice))

    # 12. User-level memory isolation
    def test_12_user_level_memory_isolation(self):
        # Alice's memory should not be returned for Bob
        bob_mems = memory_retriever.retrieve_relevant_memories(self.user_bob, "Python 3.13")
        self.assertEqual(len(bob_mems), 0)

    # 13. Unauthorized access prevention
    def test_13_unauthorized_conversation_access(self):
        conversation_manager.create_conversation(self.user_alice, "Alice Secret", conversation_id=self.conv_id)
        # Bob tries to access Alice's conversation -> Must raise PermissionError
        with self.assertRaises(PermissionError):
            conversation_manager.get_conversation(self.conv_id, user_id=self.user_bob)

    # 14. Follow-up query understanding context resolution
    def test_14_follow_up_context_injection(self):
        conversation_manager.add_turn(self.conv_id, self.user_alice, "What is LangGraph?", "StateGraph framework")
        context_str = conversation_manager.format_history_context(self.conv_id, self.user_alice)
        self.assertIn("LangGraph", context_str)
        self.assertIn("StateGraph", context_str)

    # 15. Memory extraction from conversations
    def test_15_explicit_memory_extraction(self):
        extracted = memory_extractor.extract_and_save(
            user_id=self.user_alice,
            query="Please remember that my team uses PostgreSQL 16.",
            response="I will remember that your team uses PostgreSQL 16.",
            conversation_id=self.conv_id
        )
        self.assertTrue(len(extracted) > 0)
        self.assertIn("postgresql 16", extracted[0]["content"].lower())

    # 16. Secret / credential rejection in memory
    def test_16_secret_rejection(self):
        bad_mem = memory_manager.save_memory(
            user_id=self.user_alice,
            content="My secret api_key = 'sk-1234567890abcdef1234567890'",
            category="explicit_notes"
        )
        self.assertIsNone(bad_mem)

    # 17. Checkpoint / history clear
    def test_17_clear_session_history(self):
        c = conversation_manager.create_conversation(self.user_alice, "Clear Test", conversation_id="conv_clear_test")
        conversation_manager.clear("conv_clear_test", self.user_alice)
        self.assertIsNone(conversation_manager.get_conversation("conv_clear_test", self.user_alice))

    # 18. Empty memory retrieval returns empty list gracefully
    def test_18_empty_memory_retrieval(self):
        mems = memory_retriever.retrieve_relevant_memories("non_existent_user", "quantum physics")
        self.assertEqual(mems, [])


if __name__ == "__main__":
    unittest.main()
