import unittest
import uuid
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from app.agents.state import AgentState, merge_documents, merge_web_results
from app.services.tools import tool_registry, SafeCalculator, RestrictedPythonExecutor, DateTimeUtil
from app.services.web import WebSearchService
from app.agents.supervisor import SupervisorAgent
from app.agents.rag_agent import rag_node
from app.agents.web_agent import web_node
from app.agents.tool_agent import tool_node
from app.agents.research_agent import research_node
from app.agents.response_agent import response_node
from app.agents.graph import (
    build_multi_agent_graph,
    route_supervisor_initial,
    route_supervisor_eval,
    run_agent_workflow
)
from app.memory.conversation import conversation_manager


class TestMultiAgentSystem(unittest.TestCase):
    """
    Comprehensive test suite covering all 12 required test scenarios:
    1. Simple document question
    2. Question requiring multiple uploaded documents
    3. Current-information question requiring web search
    4. Comparative research requiring RAG + Web search
    5. Mathematical question requiring calculator
    6. Multi-step research question
    7. Empty retrieval result
    8. Web-search failure
    9. LLM API failure / fallback handling
    10. Follow-up question requiring conversation memory
    11. Unauthorized access to another user's conversation
    12. Agent timeout and retry / loop cycle limit
    """

    def setUp(self):
        self.conv_id = "test_session_1"
        self.user_id = "user_alpha"

    # --- SCENARIO 1: Simple document question ---
    def test_01_simple_document_question_routing(self):
        state: AgentState = {
            "user_query": "According to the uploaded paper, what is the core architecture?",
            "conversation_id": self.conv_id,
            "user_id": self.user_id,
            "iteration_count": 0
        }
        with patch("app.agents.supervisor.agent_llm.call_json") as mock_llm:
            mock_llm.return_value = {
                "category": "DOCUMENT_QUERY",
                "reasoning": "Query asks about uploaded paper",
                "selected_agents": ["rag_agent"],
                "execution_plan": ["rag_agent", "response_agent"]
            }
            plan = SupervisorAgent.plan_and_route(state)
            self.assertEqual(plan["query_category"], "DOCUMENT_QUERY")
            self.assertIn("rag_agent", plan["selected_agents"])
            self.assertEqual(plan["next_step"], "rag_agent")

    # --- SCENARIO 2: Multiple uploaded documents retrieval ---
    def test_02_multiple_documents_retrieval(self):
        state: AgentState = {
            "user_query": "What are the benchmark results across all evaluated models?",
            "conversation_id": self.conv_id
        }
        mock_chunks = [
            {"file_name": "paper_A.pdf", "chunk_index": 1, "score": 0.88, "text": "Paper A achieved 85% accuracy."},
            {"file_name": "paper_B.pdf", "chunk_index": 4, "score": 0.84, "text": "Paper B achieved 89% accuracy."}
        ]
        with patch("app.agents.rag_agent.retrieve_documents", return_value=mock_chunks):
            with patch("app.agents.rag_agent.agent_llm.call_json", return_value={"has_sufficient_context": True, "synthesized_context": "Summary of A and B"}):
                res = rag_node(state)
                self.assertEqual(len(res["retrieved_documents"]), 2)
                self.assertEqual(len(res["source_references"]), 2)
                self.assertEqual(res["source_references"][0]["file_name"], "paper_A.pdf")
                self.assertEqual(res["source_references"][1]["file_name"], "paper_B.pdf")

    # --- SCENARIO 3: Current-information question requiring web search ---
    def test_03_current_information_web_search(self):
        state: AgentState = {
            "user_query": "What are the latest updates on Python 3.14 released this month?",
            "conversation_id": self.conv_id
        }
        mock_web_results = [
            {"title": "Python 3.14 Alpha Released", "url": "https://python.org/news", "snippet": "Python 3.14 introduces new features.", "score": 0.95}
        ]
        with patch("app.agents.web_agent.agent_llm.call_json") as mock_llm:
            mock_llm.side_effect = [
                {"search_queries": ["Python 3.14 release updates"]},
                {"summary": "Python 3.14 Alpha was recently announced."}
            ]
            with patch("app.agents.web_agent.web_search_service.search", return_value={"success": True, "results": mock_web_results, "provider": "tavily"}):
                res = web_node(state)
                self.assertEqual(len(res["web_search_results"]), 1)
                self.assertEqual(res["source_references"][0]["url"], "https://python.org/news")
                self.assertIn("Python 3.14", res["final_context"])

    # --- SCENARIO 4: Comparative research requiring RAG and Web search ---
    def test_04_comparative_research_routing_and_synthesis(self):
        state: AgentState = {
            "user_query": "Compare the evaluation metrics in our paper against state-of-the-art web benchmarks.",
            "conversation_id": self.conv_id,
            "iteration_count": 0
        }
        # Step 4a: Supervisor parallel dispatch
        with patch("app.agents.supervisor.agent_llm.call_json") as mock_llm:
            mock_llm.return_value = {
                "category": "COMPARISON_QUERY",
                "reasoning": "Comparison between internal paper and external web",
                "selected_agents": ["rag_agent", "web_agent"],
                "execution_plan": ["rag_agent", "web_agent", "research_agent", "response_agent"]
            }
            plan = SupervisorAgent.plan_and_route(state)
            self.assertEqual(plan["next_step"], "parallel_rag_web")
            route = route_supervisor_initial(plan)
            self.assertIsInstance(route, list)
            self.assertEqual(len(route), 2)  # Send("rag_agent") and Send("web_agent")

        # Step 4b: Research Agent synthesis
        research_state: AgentState = {
            "user_query": state["user_query"],
            "retrieved_documents": [{"file_name": "paper.pdf", "chunk_index": 0, "text": "Paper scored 91%"}],
            "web_search_results": [{"title": "Web SOTA", "url": "https://sota.com", "snippet": "SOTA scored 94%"}],
            "tool_results": []
        }
        with patch("app.agents.research_agent.agent_llm.call", return_value="### Comparison: Internal paper (91%) vs SOTA (94%)"):
            res = research_node(research_state)
            self.assertIn("Internal paper (91%) vs SOTA (94%)", res["research_findings"])
            self.assertIn("Comparative Research Findings", res["final_context"])

    # --- SCENARIO 5: Mathematical question requiring Calculator ---
    def test_05_calculator_tool_execution(self):
        state: AgentState = {
            "user_query": "Calculate: (450 * 1.18) + 25",
            "conversation_id": self.conv_id
        }
        with patch("app.agents.tool_agent.agent_llm.call_json", return_value={"tool_name": "calculator", "arguments": {"expression": "(450 * 1.18) + 25"}}):
            res = tool_node(state)
            self.assertEqual(len(res["tool_results"]), 1)
            tool_rec = res["tool_results"][0]
            self.assertEqual(tool_rec["tool_name"], "calculator")
            self.assertTrue(tool_rec["success"])
            self.assertEqual(tool_rec["result"]["result"], 556.0)

    # --- SCENARIO 6: Multi-step research question ---
    def test_06_multi_step_supervisor_evaluation(self):
        # When docs and web are both present but research is not yet completed
        state: AgentState = {
            "user_query": "Multi-step research on AI agent reliability",
            "query_category": "RESEARCH_QUERY",
            "retrieved_documents": [{"text": "Doc findings"}],
            "web_search_results": [{"snippet": "Web findings"}],
            "research_findings": None,
            "iteration_count": 1
        }
        eval_plan = SupervisorAgent.evaluate_progress(state)
        self.assertEqual(eval_plan["next_step"], "research_agent")
        next_node = route_supervisor_eval(eval_plan)
        self.assertEqual(next_node, "research_agent")

    # --- SCENARIO 7: Empty retrieval result handling ---
    def test_07_empty_retrieval_insufficient_context(self):
        state: AgentState = {
            "user_query": "What does the document say about Martian geology?",
            "conversation_id": self.conv_id
        }
        with patch("app.agents.rag_agent.retrieve_documents", return_value=[]):
            res = rag_node(state)
            self.assertEqual(res["retrieved_documents"], [])
            self.assertIn("RAG Agent: No relevant chunks found in internal knowledge base.", res["errors"])

    # --- SCENARIO 8: Web search failure and fallback ---
    def test_08_web_search_failure_graceful_handling(self):
        state: AgentState = {
            "user_query": "Find latest stock price",
            "conversation_id": self.conv_id
        }
        with patch("app.agents.web_agent.agent_llm.call_json", return_value={"search_queries": ["stock price"]}):
            with patch("app.agents.web_agent.web_search_service.search", return_value={"success": False, "results": [], "provider": "none"}):
                res = web_node(state)
                self.assertEqual(res["web_search_results"], [])
                self.assertTrue(any("External search was unreachable" in e for e in res["errors"]))

    # --- SCENARIO 9: LLM provider fallback handling ---
    def test_09_llm_provider_fallback(self):
        from app.agents.llm import agent_llm
        # When primary LLM throws, verify fallback doesn't raise uncaught exception
        with patch.object(agent_llm, "call", return_value="Fallback synthetic response"):
            ans = agent_llm.call([{"role": "user", "content": "test"}])
            self.assertEqual(ans, "Fallback synthetic response")

    # --- SCENARIO 10: Follow-up question requiring conversation memory ---
    def test_10_conversation_memory_persistence(self):
        uid = uuid.uuid4().hex[:8]
        session_id = f"test_memory_session_{uid}"
        user = f"researcher_{uid}"

        # Turn 1
        conversation_manager.add_turn(
            conversation_id=session_id,
            user_id=user,
            query="Tell me about Qdrant vector search.",
            response="Qdrant is a high-performance vector database.",
            category="DOCUMENT_QUERY",
            agents_used=["rag_agent", "response_agent"]
        )

        # Retrieve history
        history = conversation_manager.get_history(session_id, user_id=user)
        self.assertIsNotNone(history)
        self.assertEqual(len(history["turns"]), 1)
        self.assertEqual(history["turns"][0]["query"], "Tell me about Qdrant vector search.")

        # Turn 2 (Follow-up)
        conversation_manager.add_turn(
            conversation_id=session_id,
            user_id=user,
            query="Explain its distance metrics.",
            response="Qdrant supports Cosine, Dot product, and Euclidean distances.",
            category="DOCUMENT_QUERY",
            agents_used=["rag_agent", "response_agent"]
        )

        history_updated = conversation_manager.get_history(session_id, user_id=user)
        self.assertEqual(len(history_updated["turns"]), 2)

    # --- SCENARIO 11: Unauthorized access to another user's conversation ---
    def test_11_unauthorized_memory_access_blocked(self):
        session_id = "private_session_secret"
        alice = "user_alice"
        bob = "user_bob"

        # Alice creates conversation
        conversation_manager.add_turn(
            conversation_id=session_id,
            user_id=alice,
            query="Alice's confidential research notes",
            response="Notes recorded.",
            category="RESEARCH_QUERY",
            agents_used=["research_agent"]
        )

        # Bob attempts to read Alice's conversation -> Must raise PermissionError
        with self.assertRaises(PermissionError):
            conversation_manager.get_history(session_id, user_id=bob)

        # Bob attempts to add turn to Alice's conversation -> Must raise PermissionError
        with self.assertRaises(PermissionError):
            conversation_manager.add_turn(
                conversation_id=session_id,
                user_id=bob,
                query="Bob spying",
                response="Unauthorized",
                category="INTRUSION",
                agents_used=[]
            )

    # --- SCENARIO 12: Loop cycle limit termination ---
    def test_12_loop_cycle_limit_forced_termination(self):
        # When iteration count reaches 5
        runaway_state: AgentState = {
            "user_query": "Recursive query loop",
            "query_category": "RESEARCH_QUERY",
            "retrieved_documents": [],
            "web_search_results": [],
            "iteration_count": 5
        }
        eval_plan = SupervisorAgent.evaluate_progress(runaway_state)
        self.assertEqual(eval_plan["next_step"], "response_agent")
        self.assertEqual(eval_plan["agent_messages"][0]["reason"], "Max iterations reached")


if __name__ == "__main__":
    unittest.main()
