import sys
import unittest
from fastapi.testclient import TestClient

from app.router.engine import default_router
from app.router.models import QueryIntent, RouteDestination, RouterRequest
from app.router.context_store import conversation_store
from app.api.server import app


class TestIntelligentQueryRouter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        conversation_store.clear()

    def test_01_conversational_greeting(self):
        decision = default_router.route("Hello, how are you today?", conversation_id="test_conv_1")
        self.assertEqual(decision.intent, QueryIntent.CONVERSATIONAL)
        self.assertEqual(decision.route, RouteDestination.CONVERSATIONAL_HANDLER)
        self.assertGreaterEqual(decision.confidence, 0.90)
        self.assertFalse(decision.requires_retrieval)

    def test_02_document_rag(self):
        decision = default_router.route(
            "What does the research paper say about knowledge graph reasoning?",
            conversation_id="test_conv_2"
        )
        self.assertEqual(decision.intent, QueryIntent.DOCUMENT_RAG)
        self.assertEqual(decision.route, RouteDestination.DOCUMENT_RAG_PIPELINE)
        self.assertTrue(decision.requires_retrieval)
        self.assertGreaterEqual(decision.confidence, 0.80)

    def test_03_summarization(self):
        decision = default_router.route(
            "Summarize the uploaded research paper and provide an executive overview",
            conversation_id="test_conv_3"
        )
        self.assertEqual(decision.intent, QueryIntent.SUMMARIZATION)
        self.assertEqual(decision.route, RouteDestination.SUMMARIZATION_PIPELINE)
        self.assertTrue(decision.requires_retrieval)

    def test_04_technical_code(self):
        decision = default_router.route(
            "Write a Python function to implement binary search with type hints",
            conversation_id="test_conv_4"
        )
        self.assertEqual(decision.intent, QueryIntent.TECHNICAL_CODE)
        self.assertEqual(decision.route, RouteDestination.TECHNICAL_HANDLER)
        self.assertFalse(decision.requires_retrieval)

    def test_05_general_qa(self):
        decision = default_router.route(
            "What is the theory of general relativity and who formulated it?",
            conversation_id="test_conv_5"
        )
        self.assertEqual(decision.intent, QueryIntent.GENERAL_QA)
        self.assertEqual(decision.route, RouteDestination.GENERAL_LLM_PIPELINE)
        self.assertFalse(decision.requires_retrieval)

    def test_06_data_analysis(self):
        decision = default_router.route(
            "Compare the accuracy percentages and benchmark evaluation numbers between the models",
            conversation_id="test_conv_6"
        )
        self.assertIn(decision.intent, [QueryIntent.DATA_ANALYSIS, QueryIntent.INFORMATION_EXTRACTION])
        self.assertTrue(decision.requires_retrieval)

    def test_07_out_of_scope(self):
        decision = default_router.route(
            "asdfghjklzxvbnmqqwerty",
            conversation_id="test_conv_7"
        )
        self.assertEqual(decision.intent, QueryIntent.OUT_OF_SCOPE)
        self.assertEqual(decision.route, RouteDestination.FALLBACK_HANDLER)

    def test_08_context_aware_follow_up(self):
        conv_id = "test_follow_up_session"
        # Turn 1: Discuss a document
        res1 = default_router.process(RouterRequest(
            query="What are knowledge graphs according to the paper?",
            conversation_id=conv_id,
            execute_handler=False
        ))
        conversation_store.add_turn(conv_id, "user", "What are knowledge graphs according to the paper?", res1.intent)
        conversation_store.add_turn(conv_id, "assistant", "The paper defines knowledge graphs...", res1.intent)

        # Turn 2: Follow-up referencing "it"
        decision2 = default_router.route("Can you summarize it in 3 bullet points?", conversation_id=conv_id)
        self.assertEqual(decision2.intent, QueryIntent.SUMMARIZATION)
        self.assertTrue(decision2.context_used)

    def test_09_api_endpoint(self):
        response = self.client.post("/api/query-router", json={
            "query": "Summarize the research findings",
            "conversation_id": "api_test_session"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["intent"], "summarization")
        self.assertIn("document_summarization", data["route"])
        self.assertGreater(data["confidence"], 0.70)

    def test_10_api_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["api"], "healthy")
        self.assertEqual(data["groq"], "configured")


if __name__ == "__main__":
    unittest.main()

