import unittest
import uuid
from unittest.mock import patch
from fastapi.responses import FileResponse
from app.api.server import root, api_info, health_check
from app.api.routes.agents import get_agents_status, get_conversation_history, delete_conversation
from app.api.routes.chat import execute_chat, ChatRequest
from app.memory.conversation import conversation_manager


class TestMultiAgentAPI(unittest.TestCase):
    def setUp(self):
        uid = uuid.uuid4().hex[:8]
        self.conv_id = f"test_api_session_{uid}"
        self.user_id = f"test_user_{uid}"

    def test_root_endpoint(self):
        resp = root()
        if isinstance(resp, FileResponse):
            self.assertTrue(resp.path.endswith("index.html"))
        else:
            self.assertEqual(resp["version"], "2.1.0")

        info = api_info()
        self.assertEqual(info["version"], "2.1.0")
        self.assertEqual(info["status"], "online")
        self.assertIn("multi_agent_chat", info["endpoints"])

    def test_agents_status_endpoint(self):
        data = get_agents_status()
        self.assertEqual(data["status"], "operational")
        self.assertEqual(len(data["agents"]), 6)
        names = [a["name"] for a in data["agents"]]
        self.assertIn("supervisor", names)
        self.assertIn("rag_agent", names)
        self.assertIn("web_agent", names)
        self.assertIn("research_agent", names)
        self.assertIn("tool_agent", names)
        self.assertIn("response_agent", names)

    @patch("app.api.routes.chat.memory_extractor.extract_and_save")
    @patch("app.api.routes.chat.run_agent_workflow")
    def test_chat_and_conversation_lifecycle(self, mock_workflow, mock_extract):
        mock_extract.return_value = []
        mock_workflow.return_value = {
            "user_query": "What are vector embeddings?",
            "conversation_id": self.conv_id,
            "user_id": self.user_id,
            "query_category": "GENERAL_QUERY",
            "selected_agents": ["response_agent"],
            "execution_plan": ["response_agent"],
            "source_references": [],
            "execution_status": "completed",
            "errors": [],
            "final_response": "Vector embeddings represent text numerically in high-dimensional space."
        }

        # 1. Execute chat
        req = ChatRequest(
            query="What are vector embeddings?",
            conversation_id=self.conv_id,
            user_id=self.user_id
        )
        res = execute_chat(req)
        self.assertEqual(res.query, "What are vector embeddings?")
        self.assertEqual(res.execution_status, "completed")
        self.assertIn("Vector embeddings", res.final_response)

        # 2. Get conversation history
        hist = get_conversation_history(self.conv_id, user_id=self.user_id)
        self.assertEqual(len(hist["turns"]), 1)
        self.assertEqual(hist["turns"][0]["query"], "What are vector embeddings?")

        # 3. Delete conversation
        del_res = delete_conversation(self.conv_id, user_id=self.user_id)
        self.assertEqual(del_res["status"], "deleted")


if __name__ == "__main__":
    unittest.main()
