# oarc_rag Curriculum Generation Workflow

This document outlines how oarc_rag generates new curriculums using AI integration with Ollama, retrieval-augmented generation (RAG), and resource collection.

## Workflow Diagram

```mermaid
flowchart TD
    %% Styling
    classDef userInput fill:#d0f4de,stroke:#333,stroke-width:2px
    classDef aiComponent fill:#f9c6d9,stroke:#333,stroke-width:2px
    classDef ragComponent fill:#c8e7ff,stroke:#333,stroke-width:1px
    classDef dataStore fill:#fff2b2,stroke:#333,stroke-width:1px,stroke-dasharray: 5 2
    classDef processingStep fill:#d9d9d9,stroke:#333,stroke-width:1px
    classDef exportComponent fill:#e5c1fd,stroke:#333,stroke-width:1px
    
    %% User Input and Command Flow
    A[User Input]:::userInput --> B[CLI Router]
    B --> C[CreateCommand]
    C --> D[Curriculum.create]
    
    %% Curriculum Generation Path
    subgraph PrepSteps [Preparation]
        direction TB
        RunID[Generate Run ID]:::processingStep
        PrepDir[Prepare Directory Structure]:::processingStep
        E{Ollama Available?}
    end
    
    D --> PrepSteps
    
    PrepSteps --> F[CurriculumWorkflow]
    E -->|No| Error[RuntimeError: Ollama Required]
    Error --> Exit[Exit with Error Code]
    
    %% AI Integration
    subgraph AILayer [AI Integration Layer]
        direction TB
        H[OllamaClient]:::aiComponent
        H1[Generate]
        H2[Chat]
        H3[Embed]
        
        H --> H1 & H2 & H3
    end
    
    F --> AILayer
    
    %% RAG System
    subgraph RAG [Retrieval-Augmented Generation]
        direction TB
        RAGEngine[RAGEngine]:::ragComponent
        
        subgraph Embedding [Embedding Layer]
            direction TB
            EmbGen[EmbeddingGenerator]:::ragComponent
            TextChk[TextChunker]:::ragComponent
            VecDB[(VectorDatabase)]:::dataStore
        end
        
        subgraph Augmentation [Context Augmentation]
            direction TB
            ContAss[ContextAssembler]:::ragComponent
            QueryForm[QueryFormulator]:::ragComponent
            RAGMonitor[RAGMonitor]:::ragComponent
        end
        
        RAGEngine --> Embedding
        RAGEngine -.-> Augmentation
    end
    
    AILayer <--> RAG
    
    %% Resource Collection
    subgraph Resources [Resource Collection]
        direction TB
        RColl[ResourceCollector]
        I1[URLs]
        I2[Local Files]
        
        RColl --> I1 & I2
    end
    
    F --> Resources
    Resources --> RAG
    
    %% Vector Utilities and GPU
    subgraph VecUtils [Vector Utilities]
        direction TB
        VecOps[Vector Operations]:::ragComponent
        FAISS[FAISS Integration]
        GPU{GPU Available?}
        
        VecOps --> FAISS
        GPU -->|Yes| FGPU[FAISS-GPU]
        GPU -->|No| FCPU[FAISS-CPU]
    end
    
    VecUtils -.-> RAG
    
    %% Content Generation with RAG
    F --> AgentInt[RAGEnhancedAgent]:::aiComponent
    RAG --> AgentInt
    
    %% Generation Components
    subgraph GenComponents [Content Generation]
        direction TB
        K1[Overview Generation]:::processingStep
        K2[Learning Path Generation]:::processingStep
        K3[Resources Generation]:::processingStep
        K4[Projects Generation]:::processingStep
    end
    
    F --> GenComponents
    AgentInt --> GenComponents
    
    %% Assembly
    GenComponents --> L[Content Assembly]:::processingStep
    
    %% Export
    subgraph ExportOptions [Export Formats]
        direction TB
        M1[Markdown]:::exportComponent
        M2[JSON]:::exportComponent
        M3[PDF/HTML/etc]:::exportComponent
    end
    
    L --> J[Export Curriculum]
    J --> ExportOptions
```

## Core Components

### Command Line Interface

oarc_rag's CLI provides a user-friendly interface for curriculum generation:

```bash
oarc_rag create \
    --topic "Machine Learning" \
    --title "Introduction to Machine Learning" \
    --level "Beginner" \
    --links "https://scikit-learn.org/" \
    --source "./resources" \
    --format "md"
```

The workflow begins with `CreateCommand`, which processes arguments and calls `Curriculum.create()`, initiating the generation process.

### Curriculum Generation

The `Curriculum` class serves as the primary entry point, which:

1. Generates a unique run ID based on timestamp
2. Creates directory structure for outputs and artifacts
3. **Verifies Ollama availability** (required - will raise RuntimeError if not available)
4. Delegates content generation to `CurriculumWorkflow`
5. Handles exporting to various formats (Markdown, JSON, etc.)

### RAG System Integration

oarc_rag leverages a Retrieval-Augmented Generation (RAG) system that enhances content generation using relevant context:

#### RAG Components

1. **RAGEngine**: Coordinates the entire retrieval process

   ```python
   rag_engine = RAGEngine(run_id="12345", embedding_model="llama3")
   rag_engine.add_document(text="Python is a programming language...", metadata={"source": "documentation"})
   relevant_context = rag_engine.retrieve("How to learn Python?", top_k=5)
   ```

2. **EmbeddingGenerator**: Creates vector embeddings using Ollama's API

   ```python
   embedder = EmbeddingGenerator(model_name="llama3")
   embedding = embedder.embed_text("Python programming basics")
   ```

3. **TextChunker**: Splits documents into manageable pieces for embedding

   ```python
   chunker = TextChunker(chunk_size=512, overlap=50)
   chunks = chunker.chunk_text(long_document)
   ```

4. **VectorDatabase**: Stores and retrieves embeddings using SQLite

   ```python
   vector_db = VectorDatabase("./vectors.db")
   vector_db.add_document(chunks, embeddings, metadata={"source": "textbook"})
   results = vector_db.search(query_vector, top_k=5)
   ```

5. **ContextAssembler**: Formats retrieved chunks for prompt enhancement

   ```python
   assembler = ContextAssembler(format_style="markdown")
   context = assembler.assemble_context(retrieved_chunks=results)
   ```

6. **QueryFormulator**: Generates effective queries for retrieval

   ```python
   formulator = QueryFormulator()
   query = formulator.formulate_query(topic="Python", query_type="learning_path")
   ```

### Vector Operations and GPU Acceleration

Vector operations leverage scientific libraries for efficiency:

1. **Vector Utilities**: Provides functions for vector mathematics using scikit-learn

   ```python
   from oarc_rag.utils.vector import cosine_similarity, normalize_vector
   similarity = cosine_similarity(vec1, vec2)
   ```

2. **FAISS Integration**: Optional acceleration for vector search

   ```python
   from oarc_rag.utils.vector import create_faiss_index, faiss_search
   index = create_faiss_index(vectors, use_gpu=True)
   distances, indices = faiss_search(index, query_vector, k=5)
   ```

3. **GPU Detection**: Automatic GPU detection and utilization

   ```python
   from oarc_rag.utils.utils import detect_gpu, upgrade_faiss_to_gpu
   has_gpu, gpu_info = detect_gpu()
   if has_gpu:
       success, message = upgrade_faiss_to_gpu()
   ```

### Resource Collection

The `ResourceCollector` gathers information from various sources:

1. **URL Processing**: Extracts content from web resources
2. **Local Files**: Analyzes source files and directories
3. **Content Analysis**: Extracts meaningful information for RAG ingestion

### Content Generation

The `CurriculumWorkflow` class coordinates the generation of curriculum components:

1. **Overview Generation**: Creates introduction and learning outcomes
2. **Learning Path Generation**: Designs structured progression of modules
3. **Resources Generation**: Curates learning materials
4. **Projects Generation**: Develops practical exercises

Each generation step is enhanced with RAG context using the `RAGEnhancedAgent`:

```python
agent = RAGEnhancedAgent("learning_path_agent", rag_engine=rag_engine)
context = agent.retrieve_context(topic="Python", query_type="learning_path")
enhanced_prompt = agent.create_enhanced_prompt(base_prompt, topic="Python", query_type="learning_path")
```

## Performance Optimization

oarc_rag includes several optimization strategies:

1. **Efficient Vector Search**: Using FAISS for approximate nearest neighbor search
2. **GPU Acceleration**: Automatic detection and utilization of GPU resources
3. **Asynchronous Processing**: Resources are processed concurrently where possible
4. **Caching**: Embeddings are stored to avoid redundant computation
5. **Performance Monitoring**: RAGMonitor tracks metrics for optimization

## Export Formats

The final curriculum can be exported in multiple formats:

1. **Markdown**: Default format with rich text formatting
2. **JSON**: Structured data format for programmatic use
3. **PDF/HTML**: Rich document formats for sharing and publishing (planned)

Each export includes metadata about the generation process, resources used, and configuration settings.
