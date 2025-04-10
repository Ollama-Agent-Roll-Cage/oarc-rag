"""
Unit tests for the RAGAgent class.
"""
import unittest
import warnings
from unittest.mock import patch, MagicMock, AsyncMock

# Update this import to point to the new location
from oarc_rag.ai.agents.rag_agent import RAGAgent
from oarc_rag.core.engine import Engine
from oarc_rag.core.context import ContextAssembler
from oarc_rag.core.query import QueryFormulator
from oarc_rag.ai.client import OllamaClient

# Filter out the NumPy deprecation warning from FAISS
warnings.filterwarnings(
    "ignore",
    message="numpy.core._multiarray_umath is deprecated",
    category=DeprecationWarning
)

class TestRAGAgent(unittest.TestCase):
    """Test cases for the RAGAgent class."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset any singletons or shared state
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    @patch('oarc_rag.ai.agents.rag_agent.OllamaClient')
    def test_initialization(self, mock_client_class):
        """Test initialization of RAGAgent."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Initialize agent
        agent = RAGAgent(
            name="test_agent",
            model="test-model",
            temperature=0.5,
            max_tokens=500
        )
        
        # Verify initialization
        self.assertEqual(agent.name, "test_agent")
        self.assertEqual(agent.model, "test-model")
        self.assertEqual(agent.temperature, 0.5)
        self.assertEqual(agent.max_tokens, 500)
        self.assertIsNone(agent.rag_engine)
        self.assertIsInstance(agent.context_assembler, ContextAssembler)
        self.assertIsInstance(agent.query_formulator, QueryFormulator)
    
    @patch('oarc_rag.ai.agents.rag_agent.OllamaClient')
    def test_initialization_with_components(self, mock_client_class):
        """Test initialization of RAGAgent with custom components."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        mock_engine = MagicMock(spec=Engine)
        mock_context_assembler = MagicMock(spec=ContextAssembler)
        mock_query_formulator = MagicMock(spec=QueryFormulator)
        
        # Initialize agent with custom components
        agent = RAGAgent(
            name="test_agent",
            model="test-model",
            rag_engine=mock_engine,
            context_assembler=mock_context_assembler,
            query_formulator=mock_query_formulator
        )
        
        # Verify custom components were set
        self.assertEqual(agent.rag_engine, mock_engine)
        self.assertEqual(agent.context_assembler, mock_context_assembler)
        self.assertEqual(agent.query_formulator, mock_query_formulator)
    
    @patch('oarc_rag.ai.client.AsyncClient')
    @patch('oarc_rag.ai.agents.rag_agent.OllamaClient')
    def test_set_rag_engine(self, mock_client_class, mock_async_client):
        """Test setting RAG engine."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create agent
        agent = RAGAgent(name="test_agent")
        
        # Create mock engine and set it
        mock_engine = MagicMock(spec=Engine)
        mock_engine.run_id = "test-run-id"
        
        agent.set_rag_engine(mock_engine)
        
        # Verify engine was set
        self.assertEqual(agent.rag_engine, mock_engine)
    
    @patch('oarc_rag.ai.agents.rag_agent.OllamaClient')
    def test_retrieve_context(self, mock_client_class):
        """Test context retrieval for domain-agnostic use cases."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create mock RAG engine and components
        mock_engine = MagicMock(spec=Engine)
        mock_engine.retrieve.return_value = [
            {"text": "Quantum computing uses quantum bits or qubits.", "similarity": 0.92},
            {"text": "Quantum algorithms can solve certain problems faster than classical algorithms.", "similarity": 0.85}
        ]
        
        mock_query_formulator = MagicMock(spec=QueryFormulator)
        mock_query_formulator.formulate_query.return_value = "Formulated query about quantum computing"
        
        mock_context_assembler = MagicMock(spec=ContextAssembler)
        mock_context_assembler.assemble_context.return_value = "Assembled context about quantum computing"
        
        # Create agent with mocked components
        agent = RAGAgent(
            name="test_agent",
            rag_engine=mock_engine,
            query_formulator=mock_query_formulator,
            context_assembler=mock_context_assembler
        )
        
        # Test retrieving context with domain-agnostic parameters
        context = agent.retrieve_context(
            topic="quantum computing",
            query_type="analysis",
            additional_context={"domain": "physics"},
            top_k=2
        )
        
        # Verify interactions with domain-agnostic parameters
        mock_query_formulator.formulate_query.assert_called_with(
            topic="quantum computing",
            query_type="analysis",
            additional_context={"domain": "physics"}
        )
        
        mock_engine.retrieve.assert_called_with(
            query="Formulated query about quantum computing",
            top_k=2,
            threshold=0.0
        )
        
        self.assertEqual(context, "Assembled context about quantum computing")
        
        # Test retrieval statistics were updated
        self.assertEqual(agent.retrieval_stats["calls"], 1)
        self.assertEqual(agent.retrieval_stats["total_chunks"], 2)
    
    @patch('oarc_rag.ai.agents.rag_agent.OllamaClient')
    def test_retrieve_context_multiple_domains(self, mock_client_class):
        """Test context retrieval works across multiple domains."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create mock RAG engine and components
        mock_engine = MagicMock(spec=Engine)
        mock_engine.retrieve.side_effect = [
            # First call - technical domain
            [
                {"text": "Technical content about APIs", "similarity": 0.91},
                {"text": "RESTful design principles", "similarity": 0.87}
            ],
            # Second call - medical domain  
            [
                {"text": "Medical research on cardiovascular health", "similarity": 0.89},
                {"text": "Recent studies on blood pressure treatment", "similarity": 0.82}
            ]
        ]
        
        mock_query_formulator = MagicMock(spec=QueryFormulator)
        mock_query_formulator.formulate_query.side_effect = [
            "Formulated technical query about REST APIs",
            "Formulated medical query about cardiovascular health"
        ]
        
        mock_context_assembler = MagicMock(spec=ContextAssembler)
        mock_context_assembler.assemble_context.side_effect = [
            "Assembled technical context about APIs", 
            "Assembled medical context about cardiovascular health"
        ]
        
        # Create agent with mocked components
        agent = RAGAgent(
            name="test_agent",
            rag_engine=mock_engine,
            query_formulator=mock_query_formulator,
            context_assembler=mock_context_assembler
        )
        
        # Test retrieving context in technical domain
        tech_context = agent.retrieve_context(
            topic="REST APIs",
            query_type="technical",
            additional_context={"industry": "software"}
        )
        
        # Test retrieving context in medical domain
        medical_context = agent.retrieve_context(
            topic="cardiovascular health",
            query_type="research",
            additional_context={"specialty": "cardiology"}
        )
        
        # Verify both domains work
        self.assertEqual(tech_context, "Assembled technical context about APIs")
        self.assertEqual(medical_context, "Assembled medical context about cardiovascular health")
        
        # Verify formulate_query was called with correct parameters for each domain
        mock_query_formulator.formulate_query.assert_any_call(
            topic="REST APIs",
            query_type="technical",
            additional_context={"industry": "software"}
        )
        
        mock_query_formulator.formulate_query.assert_any_call(
            topic="cardiovascular health", 
            query_type="research",
            additional_context={"specialty": "cardiology"}
        )
    
    @patch('oarc_rag.ai.agents.rag_agent.OllamaClient')
    def test_create_enhanced_prompt(self, mock_client_class):
        """Test creating enhanced prompt with various context strategies."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create agent with mock retrieve_context method
        agent = RAGAgent(name="test_agent")
        agent.retrieve_context = MagicMock(return_value="Relevant context about renewable energy")
        
        # Test prefix strategy (default)
        base_prompt = "Explain the advantages of solar power"
        prefix_prompt = agent.create_enhanced_prompt(
            base_prompt=base_prompt,
            topic="renewable energy",
            query_type="analysis"
        )
        
        # Test suffix strategy
        suffix_prompt = agent.create_enhanced_prompt(
            base_prompt=base_prompt,
            topic="renewable energy",
            query_type="analysis",
            context_strategy="suffix"
        )
        
        # Test combined strategy
        combined_prompt = agent.create_enhanced_prompt(
            base_prompt=base_prompt,
            topic="renewable energy", 
            query_type="analysis",
            context_strategy="combined"
        )
        
        # Verify each strategy produces correct format
        self.assertIn("Context information:", prefix_prompt)
        self.assertIn("Relevant context about renewable energy", prefix_prompt)
        self.assertIn("Based on the above context", prefix_prompt)
        self.assertIn(base_prompt, prefix_prompt)
        
        self.assertIn(base_prompt, suffix_prompt)
        self.assertIn("Use the following context to inform your response", suffix_prompt)
        self.assertIn("Relevant context about renewable energy", suffix_prompt)
        
        self.assertIn("Context information:", combined_prompt)
        self.assertIn("Task:", combined_prompt)
        self.assertIn("Generate a response that uses the context", combined_prompt)
        
        # Verify retrieve_context was called with appropriate parameters
        agent.retrieve_context.assert_called_with(
            topic="renewable energy",
            query_type="analysis",
            additional_context=None,
            top_k=5
        )
    
    @patch('oarc_rag.ai.agents.rag_agent.OllamaClient')
    def test_process(self, mock_client_class):
        """Test full processing workflow with different domains."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.generate.return_value = "Generated response about climate science"
        mock_client_class.return_value = mock_client
        
        # Create agent with mocked methods
        agent = RAGAgent(name="test_agent")
        agent.retrieve_context = MagicMock(return_value="Relevant context about climate change")
        agent.create_enhanced_prompt = MagicMock(return_value="Enhanced prompt with climate context")
        
        # Process a climate science question
        result = agent.process({
            "topic": "climate change",
            "query_type": "scientific",
            "base_prompt": "Explain recent climate modeling advancements",
            "additional_context": {"research_area": "climate science"}
        })
        
        # Verify flow and results
        self.assertEqual(result, "Generated response about climate science")
        agent.create_enhanced_prompt.assert_called_with(
            base_prompt="Explain recent climate modeling advancements",
            topic="climate change",
            query_type="scientific", 
            additional_context={"research_area": "climate science"}
        )
        mock_client.generate.assert_called_with(
            prompt="Enhanced prompt with climate context",
            model=agent.model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens
        )
        self.assertEqual(agent.get_result("last_generation"), "Generated response about climate science")
    
    @patch('oarc_rag.ai.agents.rag_agent.OllamaClient')
    def test_process_invalid_input(self, mock_client_class):
        """Test processing with invalid input data."""
        # Setup mocks
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Create agent
        agent = RAGAgent(name="test_agent")
        
        # Test with missing required fields
        invalid_input = {
            "query_type": "analysis",
            "base_prompt": "Analyze the data"
            # Missing 'topic' field
        }
        
        # Verify raises ValueError with appropriate message about missing field
        with self.assertRaises(ValueError) as cm:
            agent.process(invalid_input)
        
        self.assertIn("topic", str(cm.exception).lower())
        
        # Test with missing base_prompt
        invalid_input = {
            "topic": "data analysis",
            "query_type": "analysis"
            # Missing 'base_prompt' field  
        }
        
        with self.assertRaises(ValueError) as cm:
            agent.process(invalid_input)
            
        self.assertIn("base_prompt", str(cm.exception).lower())


if __name__ == '__main__':
    unittest.main()
