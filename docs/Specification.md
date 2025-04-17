# Cognitive RAG (CRAG) Engine: Architecture, Specification & Implementation Plan

This document provides a comprehensive architecture, specification, and implementation plan for the advanced Cognitive Retrieval-Augmented Generation (CRAG) engine. Building upon a foundation of high-performance RAG capabilities—including cutting-edge embedding generation and efficient Approximate Nearest Neighbor (ANN) search via Hierarchical Navigable Small World (HNSW) algorithms—the CRAG engine introduces a novel layer of biologically-inspired self-regulation. This cognitive layer features a dynamic energy and entropy model to manage computational resources, multi-stage sleep cycles for automated system maintenance and optimization, and a sophisticated dual-graph memory system combining an associative Experience Graph with a structured Memory Graph (detailed in `Cognition.md`). Integrated with a recursive agent framework for continuous content refinement and a concurrent FastAPI-based service layer, the CRAG engine is designed for high throughput, robust self-improvement, and adaptive resource management in demanding operational environments.

## Table of Contents

- [1. Overview](#1-overview)
- [2. System Architecture](#2-system-architecture)
  - [2.1 Embedding & Data Processing](#21-embedding--data-processing)
  - [2.2 Similarity Search via ANN/HNSW](#22-similarity-search-via-annhnsw)
    - [2.2.1 Vector Operations](#221-vector-operations)
    - [2.2.2 PCA-Based Indexing Optimization](#222-pca-based-indexing-optimization)
  - [2.3 Recursive Agent Operations](#23-recursive-agent-operations)
  - [2.4 Data Storage and Query Layer](#24-data-storage-and-query-layer)
    - [2.4.1 Vector Database System](#241-vector-database-system)
    - [2.4.2 Database Schema](#242-database-schema)
  - [2.5 API & Concurrency](#25-api--concurrency)
  - [2.6 Visualization Module](#26-visualization-module)
  - [2.7 Resource Collection System](#27-resource-collection-system)
  - [2.8 Performance Monitoring](#28-performance-monitoring)
  - [2.9 Multi-Level Caching System](#29-multi-level-caching-system)
  - [2.10 Design Patterns](#210-design-patterns)
    - [2.10.1 Singleton Pattern](#2101-singleton-pattern)
    - [2.10.2 Factory Pattern](#2102-factory-pattern)
    - [2.10.3 Combined Usage](#2103-combined-usage)
  - [2.11 Agentic Framework](#211-agentic-framework)
    - [2.11.1 Agent Architecture](#2111-agent-architecture)
    - [2.11.2 Agent Types](#2112-agent-types)
    - [2.11.3 Agent Collaboration Model](#2113-agent-collaboration-model)
  - [2.12 Data Acquisition System](#212-data-acquisition-system)
    - [2.12.1 Document Processing Pipeline](#2121-document-processing-pipeline)
    - [2.12.2 Web Crawling & Scraping](#2122-web-crawling--scraping)
    - [2.12.3 Multimodal Content Handling](#2123-multimodal-content-handling)
  - [2.13 Cognitive System Integration (CRAG)](#213-cognitive-system-integration-crag)
    - [2.13.1 Energy and Entropy Management](#2131-energy-and-entropy-management)
    - [2.13.2 Sleep Cycles](#2132-sleep-cycles)
    - [2.13.3 Dual-Graph Memory System](#2133-dual-graph-memory-system)
      - [2.13.3.1 Memory Graph](#21331-memory-graph)
      - [2.13.3.2 Memory Graph vs. Experience Graph](#21332-memory-graph-vs-experience-graph)
      - [2.13.3.3 Dual-Graph Integration](#21333-dual-graph-integration)
  - [2.14 System Dependencies](#214-system-dependencies)
    - [2.14.1 Core Dependencies](#2141-core-dependencies)
    - [2.14.2 LLM Integration Dependencies](#2142-llm-integration-dependencies)
    - [2.14.3 Visualization Dependencies](#2143-visualization-dependencies)
    - [2.14.4 Performance Dependencies](#2144-performance-dependencies)
    - [2.14.5 Testing Dependencies](#2145-testing-dependencies)
    - [2.14.6 Optional Dependencies](#2146-optional-dependencies)
  - [2.15 Error Handling & Recovery](#215-error-handling--recovery)
  - [2.16 Security Considerations](#216-security-considerations)
    - [2.16.1 Internal Data Security](#2161-internal-data-security)
- [3. Implementation Plan](#3-implementation-plan)
  - [3.1 Phase 1: Core Infrastructure](#31-phase-1-core-infrastructure)
  - [3.2 Phase 2: Agent System Development](#32-phase-2-agent-system-development)
  - [3.3 Phase 3: Data Acquisition System](#33-phase-3-data-acquisition-system)
  - [3.4 Phase 4: API and Services](#34-phase-4-api-and-services)
  - [3.5 Phase 5: Optimization & Scaling](#35-phase-5-optimization--scaling)
  - [3.6 Phase 6: Cognitive System Implementation](#36-phase-6-cognitive-system-implementation)
- [4. Testing & Validation](#4-testing--validation)
  - [4.1 Test Methodology](#41-test-methodology)
  - [4.2 Performance Benchmarks](#42-performance-benchmarks)
  - [4.3 Compliance Validation](#43-compliance-validation)
- [5. Operational Procedures](#5-operational-procedures)
  - [5.1 Deployment Guidelines](#51-deployment-guidelines)
  - [5.2 Monitoring & Maintenance](#52-monitoring--maintenance)
  - [5.3 Scaling Considerations](#53-scaling-considerations)
- [6. Conclusion](#6-conclusion)
- [7. Appendices](#7-appendices)
  - [7.1 Glossary of Terms](#71-glossary-of-terms)
  - [7.2 Project Directory Structure](#72-project-directory-structure)

---

## 1. Overview

The RAG engine represents a next-generation approach to information retrieval and generation, specifically engineered for real-time processing while maintaining continuous self-improvement capabilities. The system delivers several cornerstone features:

- **Dynamic Embedding Generation:** Leverages optimized Large Language Models (LLMs) to transform text into sophisticated high-dimensional vector representations (1024+ dimensions), capturing nuanced semantic meaning beyond keywords.

- **Efficient Similarity Search:** Implements production-grade Approximate Nearest Neighbor (ANN) techniques using Hierarchical Navigable Small World (HNSW) graph structures, enabling sub-millisecond retrieval even with millions of vectors.

- **Recursive Data Enhancement:** Deploys an ecosystem of specialized intelligent agents that systematically expand, refine, merge, split, and prune content blocks, creating a self-improving knowledge base that evolves with usage patterns.

- **Pandas-based Query Engine:** Incorporates a lightweight, high-performance query processing system built on Pandas dataframes, enabling rapid filtering, joining, and transformation operations on retrieved content.

- **Multi-Threaded FastAPI Backend:** Provides a modern, fully concurrent service layer capable of handling hundreds of simultaneous queries, agent operations, and real-time WebSocket update streams.

- **Real-Time Visualization:** Features a dedicated visualization application for knowledge graph exploration with dynamic zooming, panning, node selection, and real-time highlighting of information pathways.

- **Cognitive System (CRAG):** Incorporates a biologically-inspired self-regulatory framework with computational analogs to energy management, sleep cycles for maintenance, and a dual-graph memory architecture reflecting human memory consolidation processes.

---

## 2. System Architecture

### 2.1 Embedding & Data Processing

The Embedding and Data Processing system forms the foundation of semantic understanding by transforming natural language into mathematical vector spaces. This crucial component serves as the bridge between human language and machine-processable representations, enabling meaningful similarity comparisons. The system incorporates memory-efficient batching strategies, intelligent caching of frequently embedded texts, and configurable dimensionality reduction techniques to optimize the performance-accuracy tradeoff.

### 2.2 Similarity Search via ANN/HNSW

The Similarity Search system enables efficient retrieval of semantically similar content from potentially millions of vectors. Rather than performing exhaustive comparisons, the system implements Hierarchical Navigable Small World (HNSW) graphs to achieve logarithmic-time lookups. This approach creates a multi-layered graph structure where nodes represent vectors and edges connect similar vectors, allowing for rapid traversal to the nearest neighbors of a query vector. The result is a dramatic reduction in search time while maintaining high recall accuracy.

```
Algorithm: HNSWSearch

SearchHNSW(query_vector, index, top_k=5, ef_search=100):
  // Set search parameters
  index.set_ef(ef_search)
  
  // Perform search
  ids, distances ← index.knn_query(query_vector, k=top_k)
  
  // Convert distances to similarities
  similarities ← [1 - distance for distance in distances]
  
  // Create result objects
  results ← []
  for i from 0 to |ids|-1:
    results.append({id: ids[i], similarity: similarities[i]})
  
  // Sort by similarity
  Sort results by similarity (descending)
  
  return results
```

### 2.2.1 Vector Operations

The Vector Operations subsystem implements fundamental mathematical functions for working with embeddings. These operations form the mathematical foundation for all similarity calculations and vector manipulations throughout the system. Efficient implementations of these operations are critical for system performance, as they are executed millions of times during normal operation. The system includes specialized functions for vector normalization, similarity calculation, and batch processing to maximize computational efficiency.

```
Algorithm: VectorOperations

NormalizeVector(vector):
  norm ← SquareRoot(Sum(vector[i]² for all i))
  if norm > 0:
    return [vector[i]/norm for all i]
  return vector  // Zero vector

CosineSimilarity(vector1, vector2):
  v1 ← NormalizeVector(vector1)
  v2 ← NormalizeVector(vector2)
  return Sum(v1[i] × v2[i] for all i)

BatchSimilarity(query_vector, candidate_vectors):
  norm_query ← NormalizeVector(query_vector)
  similarities ← []
  
  for each candidate in candidate_vectors:
    norm_candidate ← NormalizeVector(candidate)
    similarity ← Sum(norm_query[i] × norm_candidate[i] for all i)
    similarities.append(similarity)
  
  return similarities
```

### 2.2.2 PCA-Based Indexing Optimization

The PCA-Based Indexing Optimization system enhances search performance by reducing the dimensionality of embedding vectors while preserving their semantic relationships. Principal Component Analysis (PCA) identifies the most informative dimensions in the embedding space, allowing the system to represent vectors with fewer dimensions while retaining most of their semantic information. This optimization creates a two-tier search architecture where initial candidate selection uses reduced vectors for speed, followed by precise re-ranking with full vectors for accuracy.

```
Algorithm: PCAIndexOptimization

OptimizePCAIndex(embeddings, target_dimensions=128, variance_threshold=0.9):
  // Compute PCA
  pca_model ← FitPCA(embeddings, n_components=target_dimensions)
  
  // Check explained variance
  variance_ratio ← pca_model.explained_variance_ratio
  cumulative_variance ← Sum(variance_ratio)
  
  // Adjust dimensions if needed
  if cumulative_variance < variance_threshold:
    target_dimensions ← FindMinDimensionsForVariance(variance_ratio, variance_threshold)
    pca_model ← FitPCA(embeddings, n_components=target_dimensions)
  
  // Transform embeddings to reduced space
  reduced_embeddings ← pca_model.transform(embeddings)
  
  return {
    pca_model,
    reduced_embeddings,
    variance_preserved: cumulative_variance,
    dimensions: target_dimensions
  }
```

### 2.3 Recursive Agent Operations

The Recursive Agent Operations system orchestrates a team of specialized AI agents that continuously refine and enhance the knowledge base. Each agent performs specific tasks such as expanding content with additional context, merging related information, splitting mixed content, or pruning irrelevant details. The system coordinates these agents based on the current cognitive state, prioritizing operations according to available energy and system needs. This creates a self-improving knowledge ecosystem that becomes more valuable over time.

```
Algorithm: AgentCoordinator

CoordinateAgents(cognitive_system, document_store, agents):
  // Get system state
  system_state ← cognitive_system.GetSystemState()
  energy_level ← cognitive_system.GetEnergyLevel()
  optimization_level ← cognitive_system.GetOptimizationLevel()
  
  // Prioritize agents based on state
  agent_queue ← PrioritizeAgentsByState(agents, system_state)
  
  // Execute with energy awareness
  results ← []
  for each agent in agent_queue:
    // Check energy requirements
    required_energy ← agent.GetEnergyRequirement()
    if energy_level < required_energy and agent.priority ≠ "critical":
      DeferAgentOperation(agent)
      continue
    
    // Execute agent
    agent.SetOptimizationLevel(optimization_level)
    cognitive_system.RecordOperationStart(agent.type)
    agent_result ← agent.Execute(document_store)
    cognitive_system.RecordOperationComplete(agent.type)
    
    results.append(agent_result)
  
  return results
```

### 2.4 Data Storage and Query Layer

The Data Storage and Query Layer provides the persistent backbone for the RAG engine, managing both the high-dimensional vector embeddings and their associated metadata. It integrates a high-performance vector database system optimized for large-scale similarity search and retrieval operations, alongside mechanisms for querying and managing the textual content and metadata linked to these vectors. This layer ensures data integrity, consistency, and efficient access for both retrieval and agent-based refinement processes.

#### 2.4.1 Vector Database System

The Vector Database System is specifically chosen or designed to store high-dimensional embeddings (e.g., 1024+ dimensions) efficiently and support ultra-fast Approximate Nearest Neighbor (ANN) search operations, leveraging indexing structures like HNSW. It includes features such as indexing for rapid lookups, potential sharding for horizontal scalability across multiple nodes, and replication for fault tolerance and high availability, ensuring the system can handle millions or billions of vectors while maintaining low-latency retrieval.

#### 2.4.2 Database Schema

The Database Schema defines the logical structure for storing embeddings, original content chunks, metadata (source, timestamps, processing history), and relationships within the vector database and potentially associated relational or document stores. It typically includes tables or collections for embeddings (linking vector IDs to content IDs), documents/chunks (storing text and metadata), agent operation logs, and query history, enabling efficient data organization, linkage, and retrieval for various system functions.

### 2.5 API & Concurrency

The API & Concurrency module provides a robust, high-performance service layer built using FastAPI, enabling interaction with the RAG engine's capabilities. It supports asynchronous request handling for high concurrency, offering RESTful endpoints for core functions like embedding generation, similarity search, document ingestion, and agent operations. Additionally, it may utilize WebSocket streams for real-time updates, such as monitoring agent progress or receiving notifications about system state changes from the CRAG component.

### 2.6 Visualization Module

The Visualization Module offers interactive tools, potentially as a separate web application, for exploring the knowledge structures within the RAG engine, particularly the Memory Graph and Experience Graph from the CRAG component. It provides a user-friendly interface featuring dynamic zooming, panning, node selection, and relationship highlighting, allowing users or administrators to visually understand the connections between data points, trace information flow, and gain insights into the system's learned knowledge.

### 2.7 Resource Collection System

The Resource Collection System manages the acquisition and initial storage of content from various sources using the dedicated **`oarc-crawlers`** package. This package provides specialized crawlers and extractors for different source types (e.g., web pages via BeautifulSoup, YouTube videos, GitHub repositories, ArXiv papers, DuckDuckGo results) and content formats. It features integrated **Parquet storage** for efficient data persistence. 

The system operates primarily through the `ResourceCollector` component, which coordinates the data acquisition process in two main steps:

1. **Crawling Job Coordination**: The `ResourceCollector` triggers crawling jobs within the `oarc-crawlers` package based on configuration, agent requests, or scheduled tasks.

2. **Output Monitoring & Processing**: A dedicated background process within the `ResourceCollector` monitors the output directory where `oarc-crawlers` saves Parquet files. When new files are detected, their metadata is extracted and they're enqueued for processing by the Document Processing Pipeline.

This modular approach, centered around `oarc-crawlers`, allows the RAG engine to ingest content from diverse sources, abstracting the complexity of different formats and access methods, and storing the raw or semi-processed data efficiently before it enters the main processing pipeline. Further extensions to `oarc-crawlers` will be implemented to meet any additional source requirements outlined in this specification.

```
Algorithm: ResourceCollector (Coordination & Monitoring)

Initialize(config):
  // Initialize oarc-crawlers components (specific crawlers are part of the package)
  // Example: youtube_crawler = oarc_crawlers.YouTubeDownloader(...)
  // ... initialize other needed crawlers from oarc_crawlers based on config

  // Initialize processing queue (for documents ready for RAG pipeline ingestion)
  document_queue ← new ThreadSafeQueue()
  processed_ids ← new ConcurrentSet() // Tracks IDs processed by RAG pipeline
  monitoring_interval ← config.monitoring_interval or 60 # seconds
  oarc_crawlers_data_dir ← config.oarc_crawlers_data_dir

  // Start background monitoring process/thread
  StartBackgroundMonitor(oarc_crawlers_data_dir, monitoring_interval, document_queue, processed_ids)

TriggerCrawlingJob(source_config):
  // Determine the appropriate crawler from oarc-crawlers based on source_config
  crawler_type ← source_config.type // e.g., "youtube", "github", "web", "arxiv", "ddg"
  crawler_instance ← GetOarcCrawlerInstance(crawler_type)

  if not crawler_instance:
    throw Exception("Unsupported source type for oarc-crawlers: " + crawler_type)

  // oarc-crawlers handles internal rate limiting, async execution, and Parquet saving.
  // This call likely returns quickly after initiating the crawl job within oarc-crawlers.
  try:
    job_id = crawler_instance.StartJob(source_config) # Example method name
    return { status: "triggered", source_type: crawler_type, job_id: job_id }
  catch Exception as e:
    return { status: "error", message: e.message }


// Background process/thread monitors oarc-crawlers output directory
MonitorCrawlerOutput(data_dir, interval, queue, processed_ids):
  while True:
    new_files = CheckForNewParquetFiles(data_dir) # Scans directory for new/unprocessed files
    for file_path in new_files:
      try:
        metadata = ReadParquetMetadata(file_path) // Get source info, crawler type etc.
        doc_id = metadata.get("id") or GenerateIDFromFile(file_path)

        if doc_id not in processed_ids:
           // Enqueue metadata or path for the RAG processing pipeline
          queue.Enqueue({
              "id": doc_id,
              "source_path": file_path,
              "source_type": metadata.get("crawler_type"),
              "metadata": metadata
              })
          processed_ids.Add(doc_id)
      catch Exception as e:
          LogError("Failed to process discovered file:", file_path, e)
          # Optionally move file to an error directory

    Sleep(interval)
```

### 2.8 Performance Monitoring

The Performance Monitoring module continuously tracks key system metrics to ensure operational health and efficiency. It gathers data points such as query latency (end-to-end and component-specific), agent operation durations, resource utilization (CPU, memory, disk I/O, network), cache hit/miss rates, and CRAG energy/entropy levels. This data is exposed through dashboards (e.g., using Grafana) and can trigger alerts based on predefined thresholds, enabling proactive identification of bottlenecks or potential issues.

### 2.9 Multi-Level Caching System

The Multi-Level Caching System significantly improves performance and reduces redundant computations by storing frequently accessed data closer to the point of use. It may include in-memory caches (e.g., using Redis or Python dictionaries) for embedding results, query results, and intermediate agent outputs. It employs intelligent strategies for cache invalidation (e.g., time-to-live, event-based) and prioritization (e.g., LRU - Least Recently Used) to maximize cache hit rates and overall system efficiency, adapting cache behavior based on CRAG's energy state.

### 2.10 Design Patterns

The system architecture leverages established software design patterns to promote modularity, flexibility, scalability, and maintainability, ensuring a robust and extensible codebase.

#### 2.10.1 Singleton Pattern

The Singleton Pattern ensures that certain resource-intensive or stateful components, such as the connection pool to the vector database, the central configuration manager, or the CRAG cognitive model instance, are instantiated only once per application lifecycle. This prevents resource contention, ensures consistent state management, and optimizes memory usage.

#### 2.10.2 Factory Pattern

The Factory Pattern is employed to abstract the creation logic for various objects, particularly for components like agents or data processors where different implementations might exist. It allows the system to dynamically create instances of specific agent types (e.g., `ExpansionAgent`, `RefinementAgent`) or document processors based on configuration or runtime conditions, promoting loose coupling and easier extension.

#### 2.10.3 Combined Usage

The strategic combination of design patterns like Singleton, Factory, Observer (for event handling), and Strategy (for varying algorithms like chunking or embedding) enables the system to achieve a balance between flexibility, efficiency, and maintainability. For instance, a Factory might produce Singleton instances, or different Strategy implementations might be managed as Singletons.

### 2.11 Agentic Framework

The Agentic Framework provides the structure and coordination mechanism for the specialized AI agents responsible for recursive data enhancement and other autonomous tasks within the system.

#### 2.11.1 Agent Architecture

The Agent Architecture defines a standardized structure and interface for all agents. This typically includes methods for initialization (`__init__`), execution (`execute` or `run`), state management, communication with other agents or system components (like the document store or CRAG), and reporting results or errors. It ensures consistency and facilitates the integration of new agents.

#### 2.11.2 Agent Types

The system includes a diverse set of specialized agent types, each designed for a specific task in the data refinement lifecycle. Examples include `ExpansionAgent` (adds context), `RefinementAgent` (improves clarity/accuracy), `MergingAgent` (combines related chunks), `SplittingAgent` (divides mixed-topic chunks), `PruningAgent` (removes irrelevant/outdated info), and potentially agents for summarization, entity linking, or quality assessment.

#### 2.11.3 Agent Collaboration Model

The Agent Collaboration Model defines how agents interact, coordinate their actions, and share information to achieve collective goals, orchestrated by the `AgentCoordinator` (Section 2.3). This might involve direct communication, shared state management, or a blackboard system. The model ensures agents work synergistically, avoiding conflicts and leveraging each other's outputs, often guided by priorities set by the CRAG system's state.

### 2.12 Data Acquisition System

The Data Acquisition System encompasses the end-to-end process of gathering external information and preparing it for integration into the RAG engine's knowledge base. This system clearly differentiates between three distinct stages in the information flow:

1. **Data Acquisition**: The initial gathering of raw, multimodal data from external sources via the `ResourceCollector` component, which coordinates the `oarc-crawlers` package to fetch and save content to Parquet files during the AWAKE state.

2. **Document Processing**: The transformation of acquired raw data (loaded from Parquet) into structured, normalized, and chunked information suitable for embedding and storage, handled by a configurable pipeline with specialized processors for different content types.

3. **Data Enrichment**: The subsequent enhancement of processed information through verification, cross-referencing, and synthesis, executed primarily during specialized SLEEP cycles (NAPPING, SLOW_WAVE, REM) and guided by agent analysis.

Together, these stages ensure a continuous flow of high-quality information into the RAG engine's knowledge base, with each stage adding progressive value to the raw content.

**(See `docs/Enrichment.md` for comprehensive details on all three stages of the Data Acquisition System.)**

#### 2.12.1 Document Processing Pipeline

The Document Processing Pipeline is a crucial stage *following* data acquisition by `oarc-crawlers`. It takes the raw or semi-processed data (often loaded from Parquet files identified by the `ResourceCollector`) and applies a sequence of transformations through configurable stages:

- **Loading:** Reading data from Parquet files using `oarc-crawlers.ParquetStorage`.
- **Normalization:** Cleaning text (e.g., removing HTML tags, standardizing whitespace, case folding).
- **Multimodal Processing:** Extracting text from non-text formats (e.g., PDF text extraction, video transcription via `VideoProcessor`).
- **Metadata Enrichment:** Adding or refining metadata (e.g., timestamps, source URLs, detected language).
- **Chunking:** Dividing documents into smaller, manageable units suitable for embedding, using strategies like fixed-size, sentence-based, or semantic chunking.
- **Feature Extraction:** Optional steps like language detection or named entity recognition (NER).

The output of this pipeline is a set of processed document chunks ready for embedding and storage in the vector database and potentially the Experience Graph.

```
Algorithm: ProcessingPipeline

Initialize(config):
  // Configure pipeline stages (Normalization, Chunking, Metadata, etc.)
  for each stage_config in config.stages:
    processor ← CreateProcessor(stage_config.type, stage_config.config)
    processors[stage_config.name] ← processor
    stages.append(stage_config.name)

  // Initialize Parquet reader from oarc-crawlers
  parquet_storage = oarc_crawlers.ParquetStorage(config.data_dir)

ProcessAcquiredData(source_info): // source_info contains path to Parquet file, metadata
  // Load data collected by oarc-crawlers
  try:
    raw_data_df = parquet_storage.load_data(source_info["source_path"])
    // Convert DataFrame rows to document objects for pipeline processing
    initial_documents = ConvertDataFrameToDocuments(raw_data_df, source_info["metadata"])
  catch Exception as e:
     metrics.RecordError("data_loading", e)
     return {success: false, error: "Failed to load data from Parquet"}

  processed_results = []
  for document in initial_documents:
      current_doc ← document
      stage_results ← {}

      // Process through each pipeline stage
      for each stage_name in stages:
        processor ← processors[stage_name]
        try:
          start_time ← current_time()
          result ← processor.Process(current_doc)
          current_doc ← result.document
          stage_results[stage_name] ← result.metadata
          elapsed ← current_time() - start_time
          metrics.RecordStageTime(stage_name, elapsed)
        catch Exception as e:
          metrics.RecordError(stage_name, e)
          processed_results.append({
            success: false,
            error: e.message,
            stage: stage_name,
            partial_results: stage_results,
            original_doc_id: document.id
          })
          break # Stop processing this document on error
      else: # No break occurred
          processed_results.append({
            success: true,
            document: current_doc, # Final processed document ready for embedding/storage
            stage_results: stage_results
          })

  return processed_results # List of results for each initial document
```

#### 2.12.2 Web Crawling & Scraping

The core functionality for fetching and extracting data from external web sources resides within the dedicated **`oarc-crawlers`** package. This package provides:
- **General Web Crawler:** A configurable crawler (e.g., `BSWebCrawler`) for fetching HTML, respecting `robots.txt`, handling politeness delays, and extracting content/links.
- **Specialized Crawlers:** Modules tailored for specific platforms like YouTube (video metadata, captions, potentially video download), GitHub (repositories, code files, issues), ArXiv (papers, metadata, LaTeX source), and DuckDuckGo (search results).
- **Standardized Output:** Crawlers save their output consistently, typically using `ParquetStorage`, making it easy for the `ResourceCollector` to monitor and process.

The RAG engine's `ResourceCollector` component coordinates these crawlers, initiating jobs and monitoring the output directory, but the heavy lifting of interacting with external sources is encapsulated within `oarc-crawlers`.

**(See `docs/Enrichment.md`, Section 2.3, for more details on `oarc-crawlers` functionality.)**

#### 2.12.3 Multimodal Content Handling

Handling diverse data formats is a two-step process clearly separated between acquisition and processing:

1. **Acquisition (by `oarc-crawlers`):** The specialized crawlers within `oarc-crawlers` are responsible for *downloading* or *fetching* the raw multimodal content (e.g., `.mp4` video files, `.pdf` documents, `.py` code files). They store this raw content or references to it, often alongside extracted metadata, in Parquet files.

2. **Processing (by Document Processing Pipeline):** Once the `ResourceCollector` identifies a Parquet file containing multimodal data, it's passed to the Document Processing Pipeline. Specific processors within this pipeline (e.g., `VideoProcessor`, `PDFProcessor`, `CodeProcessor`) handle the *extraction* of relevant information (like text transcription from audio, text content from PDFs, or function definitions/docstrings from code) from the acquired raw files.

This separation ensures that crawling focuses on efficient data gathering, while processing handles potentially resource-intensive analysis and transformation.

```
Algorithm: MultimodalHandler (Processing Stage in Pipeline)

Initialize(config):
  // Register content processors for pipeline stages
  RegisterProcessor("application/pdf", PDFProcessor) // Extracts text from PDF
  RegisterProcessor("video/*", VideoProcessor)       // Extracts audio, potentially frames/metadata
  RegisterProcessor("audio/*", AudioProcessor)       // Transcribes audio
  RegisterProcessor("text/code", CodeProcessor)      // Analyzes code structure/docstrings
  // ... other processors ...

ProcessContent(document_data, content_type): // document_data loaded from Parquet
  // Find appropriate processor based on content_type (or file extension/metadata)
  processor ← FindMatchingProcessor(content_type)

  if not processor:
    Log warning("No specialized processor for content type: " + content_type)
    // Fallback to basic text extraction if possible
    return ExtractBasicText(document_data)

  // Process the content using the specialized processor
  result ← processor.Process(document_data, config) // e.g., PDFProcessor extracts text

  return {
    text: result.text, // Extracted/Transcribed text
    metadata: result.metadata, // Processor-specific metadata
    confidence: result.confidence
  }
```

### 2.13 Cognitive System Integration (CRAG)

The Cognitive System Integration (CRAG) embeds biologically-inspired self-regulation mechanisms into the RAG architecture. It implements three fundamental cognitive components that work together to create a self-regulating system:

1. **Energy and Entropy Model**: Tracks computational resources (energy) and system disorder (entropy) to enable adaptive operation based on resource availability.

2. **Multi-Phase Sleep Cycles**: Schedules specialized maintenance periods (NAPPING, SLOW_WAVE, REM, RECOVERY) that perform different optimization, verification, and knowledge synthesis tasks.

3. **Dual-Graph Memory Architecture**: Combines associative pattern learning (Experience Graph) with structured knowledge representation (Memory Graph) to enhance retrieval and reasoning capabilities.

This biomimetic approach enables the system to adapt to varying workloads, prioritize operations based on resource availability, perform continuous knowledge organization and enrichment, and maintain long-term system health through automated maintenance cycles.

**(See `Cognition.md` for comprehensive technical specifications of the CRAG components.)**

#### 2.13.1 Energy and Entropy Management

The Energy and Entropy Management system tracks computational resources (energy) and system disorder (entropy), enabling adaptive operation scheduling and maintenance. It consists of:

1. **Energy Tracking**: Monitoring computational resource consumption and availability through a configurable energy model.
2. **Entropy Accumulation**: Measuring system disorder that increases during operations and requires periodic reduction.
3. **State Transitions**: Automatically transitioning between operational states (OPTIMAL, FATIGUED, OVERTIRED, CRITICAL, PROTECTIVE) based on current energy and entropy levels.
4. **Optimization Levels**: Mapping system states to optimization levels (0-4) that components can use to adjust their resource usage accordingly.

```
Algorithm: CognitiveEnergyModel

Initialize(config): // Load parameters from config
  max_energy ← config.max_energy
  max_entropy ← config.max_entropy
  depletion_rates ← config.depletion_rates // Dict mapping operation_type to energy cost
  entropy_rate ← config.entropy_rate // Factor applied to energy cost for entropy increase
  state_thresholds ← config.state_thresholds // Energy percentages for state transitions
  entropy_degradation_threshold ← config.entropy_degradation_threshold // e.g., 90%

  current_energy ← max_energy
  current_entropy ← 0
  system_state ← OPTIMAL // Initial state
  current_sleep_stage ← AWAKE

RecordOperation(operation_type, count=1):
  if operation_type not in depletion_rates:
    Log warning("Unknown operation type for energy tracking:", operation_type)
    return GetCurrentStatus()

  energy_cost ← depletion_rates[operation_type] × count
  current_energy ← max(0, current_energy - energy_cost)
  current_entropy ← min(max_entropy, current_entropy + energy_cost * entropy_rate)
  UpdateSystemState()
  return GetCurrentStatus()

// ... additional methods defined in Cognition.md ...

GetOptimizationLevel():
  Match system_state:
    OPTIMAL:    return 0
    FATIGUED:   return 1
    OVERTIRED:  return 2
    CRITICAL:   return 3
    PROTECTIVE: return 4
```

#### 2.13.2 Sleep Cycles

The Sleep Cycles module schedules maintenance periods to optimize system performance and reduce entropy. Each sleep stage serves distinct purposes for cognitive maintenance:

1. **NAPPING**: Light recovery with minimal system impact, suitable for brief periods of inactivity. Performs quick energy restoration and light maintenance like cache refreshing.

2. **SLOW_WAVE**: Deep memory consolidation and system optimization. Performs resource-intensive tasks like PCA tuning, index rebuilding, vector cleanup, and deep fact verification against external sources.

3. **REM**: Knowledge synthesis and creative connection forming. Analyzes patterns in the Experience Graph, generates abstractions, creates links between related information, and performs enrichment involving synthesis and external exploration.

4. **RECOVERY**: Emergency restorative sleep triggered when the system reaches critically low energy levels. Prioritizes rapid energy restoration followed by critical system repairs.

The sleep scheduling logic determines which sleep stage to enter based on the current system state and entropy levels, and the `ExecuteSleepCycle` function dispatches the appropriate sleep function.

**(See `Cognition.md`, Section 4, for detailed algorithms of each sleep cycle.)**

#### 2.13.3 Dual-Graph Memory System

The Dual-Graph Memory System combines associative and structured knowledge representation for enhanced retrieval and reasoning capabilities. This architecture features two complementary memory structures:

##### 2.13.3.1 Memory Graph

The Memory Graph provides structured knowledge representation through a semantic network of entities and typed relationships. Unlike vector embeddings that capture similarity through proximity in vector space, the Memory Graph explicitly models knowledge as triples (subject-predicate-object) with defined relationship types. It supports operations such as entity and relationship management, path-based querying, and logical inference. This explicit representation enables precise factual recall, ontological reasoning, and structured knowledge extraction that complements the pattern-matching strengths of vector embeddings.

**(Detailed algorithms for Memory Graph operations like `AddEntity`, `AddRelationship`, and `Query` are specified in `Cognition.md`, Section 5.2.2.)**

##### 2.13.3.2 Memory Graph vs. Experience Graph

The Memory Graph and Experience Graph serve complementary functions in the CRAG system:

| Feature | Experience Graph (EG) | Memory Graph (MG) |
|---------|----------------------|-------------------|
| **Primary Purpose** | Episodic/associative memory for patterns | Semantic memory for structured facts |
| **Structure** | Flexible graph with various node/edge types | Structured entity-relationship graph |
| **Node Types** | Content chunks, queries, responses, abstractions | Entities with types and properties |
| **Edge Types** | Semantic similarity, usage patterns, temporal sequence | Typed predicates (e.g., "founded_by", "located_in") |
| **Retrieval Mechanism** | Similarity search + graph traversal | Triple pattern matching + path finding |
| **Update Frequency** | Continually during AWAKE phase | Primarily during REM sleep via Knowledge Bridge |
| **State Over Time** | Short to medium-term, refreshes between AWAKE phases | Long-term, persistent across cycles |

##### 2.13.3.3 Dual-Graph Integration

The Dual-Graph Integration system combines the complementary strengths of vector-based similarity search and structured knowledge representation. It enhances retrieval by leveraging both the Experience Graph's associative pattern matching and the Memory Graph's explicit semantic relationships. When processing a query, the system extracts relevant entities, retrieves related knowledge from both graphs, and combines the results based on relevance to the query. This integrated approach provides more comprehensive and contextually relevant information than either system could deliver independently.

```
Algorithm: EnhancedDualGraphRetrieval

EnhancedDualGraphRetrieval(query, search_config):
  // Enhanced retrieval utilizing both graphs
  
  // 1. Initial analysis
  query_analysis ← AnalyzeQuery(query)
  query_embedding ← query_analysis.embedding
  extracted_entities ← query_analysis.entities
  query_intents ← query_analysis.intents
  
  // 2. Determine if query is more factual or experiential
  is_factual_query ← IsPrimarilyFactual(query_intents)
  
  // 3. Primary search in most relevant graph
  if is_factual_query:
    // Start with Memory Graph for factual queries
    primary_results ← SearchMemoryGraph(extracted_entities, search_config)
    secondary_results ← SearchExperienceGraph(query_embedding, search_config)
  else:
    // Start with Experience Graph for experiential queries
    primary_results ← SearchExperienceGraph(query_embedding, search_config)
    secondary_results ← SearchMemoryGraph(extracted_entities, search_config)
  
  // 4. Cross-reference between graphs
  enriched_results ← CrossReferenceResults(primary_results, secondary_results, query_analysis)
  
  // 5. Perform graph traversal to find additional context if needed
  if search_config.include_graph_traversal and len(enriched_results) < search_config.min_results:
    traversal_results ← PerformGraphTraversal(enriched_results, search_config.traversal_depth)
    enriched_results ← MergeResults(enriched_results, traversal_results)
  
  // 6. Apply final ranking and format results
  final_results ← RankAndFormatResults(enriched_results, query_analysis)
  
  return final_results
```

### 2.14 System Dependencies

The System Dependencies section outlines the required libraries and frameworks for the RAG engine.

#### 2.14.1 Core Dependencies

Core dependencies include libraries for embedding generation, similarity search, data processing, and data acquisition.
- `fastapi`: For the API layer.
- `uvicorn`: ASGI server.
- `pandas`: For data manipulation and the query engine.
- `pyarrow`: For Parquet file handling (used by `oarc-crawlers` and potentially internally).
- `numpy`: For numerical operations, especially vector math.
- `scikit-learn`: For PCA and potentially other ML utilities.
- `hnswlib` or `faiss`: For ANN/HNSW indexing.
- **`oarc-crawlers`**: For data acquisition from web sources, YouTube, GitHub, etc., and Parquet storage.
- `sentence-transformers` or similar: For generating embeddings.
- `nltk` or `spacy`: For text processing (tokenization, entity extraction).

#### 2.14.2 LLM Integration Dependencies

LLM integration dependencies provide the necessary libraries to interact with Large Language Models for tasks like embedding generation, text summarization, or agentic reasoning. Examples include `transformers` (Hugging Face), `openai`, `anthropic`, or specific libraries for chosen embedding models like `sentence-transformers`.

#### 2.14.3 Visualization Dependencies

Visualization dependencies include libraries required for rendering interactive graphs and user interfaces, primarily for the Visualization Module (Section 2.6). This might involve JavaScript libraries like `D3.js`, `vis.js`, or Python libraries like `Plotly Dash` or `Streamlit` if the visualization is served via Python.

#### 2.14.4 Performance Dependencies

Performance dependencies include libraries specifically aimed at optimizing system speed, memory usage, or concurrency. Examples could be `Numba` (for JIT compilation of Python code), `Cython` (for C extensions), or optimized numerical libraries beyond `numpy`.

#### 2.14.5 Testing Dependencies

Testing dependencies encompass the frameworks and libraries used to ensure system reliability, correctness, and compliance through automated testing. Common examples include `pytest` (for writing and running tests), `pytest-asyncio` (for testing async code), `coverage` (for measuring test coverage), and `hypothesis` (for property-based testing).

#### 2.14.6 Optional Dependencies

Optional dependencies provide additional, non-essential features or integrations, such as advanced analytics connectors, specific file format parsers (beyond core needs), or experimental features. These are typically installed separately to keep the core installation lightweight.

### 2.15 Error Handling & Recovery

The Error Handling & Recovery module implements robust mechanisms to gracefully manage exceptions, log errors effectively, and attempt automated recovery where feasible. It ensures system resilience by catching errors at various layers, providing informative logging for debugging, and potentially integrating with the CRAG system (as detailed in `Cognition.md`) for energy-aware retry strategies or deferring recovery actions to sleep cycles.

### 2.16 Security Considerations

The Security Considerations section outlines measures taken to protect the system's data, ensure operational integrity, and prevent unauthorized access or misuse throughout the architecture.

#### 2.16.1 Internal Data Security

Internal Data Security focuses on protecting data within the system's boundaries. Measures include encryption at rest (for vector databases and stored documents), encryption in transit (for internal API communication), role-based access control (RBAC) for accessing sensitive data or operations, secure handling of API keys/credentials, and audit logging to track significant actions or access attempts.

## 3. Implementation Plan

The Implementation Plan outlines a phased approach for developing and deploying the RAG engine, breaking down the complex system into manageable stages.

### 3.1 Phase 1: Core Infrastructure

Phase 1 focuses on establishing the foundational components. This includes setting up the vector database, implementing the core embedding generation service (`EmbeddingGeneration`), developing the basic similarity search functionality (`HNSWSearch`, `VectorOperations`), and defining the initial database schema.

### 3.2 Phase 2: Agent System Development

Phase 2 involves creating and integrating the agentic framework. This includes defining the base `Agent Architecture`, implementing initial key `Agent Types` (e.g., expansion, refinement), developing the `AgentCoordinator`, and establishing the basic `Agent Collaboration Model`.

### 3.3 Phase 3: Data Acquisition System

This phase focuses on implementing the components responsible for bringing external data into the system and preparing it for the RAG core. Key activities include:
- **Integrating `oarc-crawlers`:** Ensuring the RAG engine can correctly invoke crawlers from the `oarc-crawlers` package based on configuration.
- **Implementing `ResourceCollector`:** Building the coordination logic that triggers crawls, monitors the output directory (`config.oarc_crawlers_data_dir`), identifies new Parquet files, avoids duplicates, and enqueues data for processing.
- **Developing Document Processing Pipeline:** Creating the framework for the pipeline and implementing essential stages like text normalization, chunking strategies, and metadata handling.
- **Implementing Multimodal Processors:** Developing or integrating processors for key multimodal formats (e.g., PDF text extraction, basic audio transcription placeholder).
- **Configuration Management:** Defining how data sources and processing stages are configured (likely via files in `configs/`).
- **Testing:** Ensuring the pipeline correctly processes various data types and that the `ResourceCollector` reliably picks up new data.

**(Refer to `docs/Enrichment.md`, Sections 2 and 3, for detailed specifications and algorithms relevant to this phase.)**

### 3.4 Phase 4: API and Services

Phase 4 develops the external-facing API and internal service communication layer using FastAPI. This involves defining API endpoints for search, ingestion, agent interaction, and potentially CRAG monitoring, implementing robust concurrency handling, and ensuring proper request validation and response formatting.

### 3.5 Phase 5: Optimization & Scaling

Phase 5 focuses on enhancing system performance, efficiency, and scalability. Activities include implementing PCA optimization, refining caching strategies, optimizing database queries and indexing, load testing, and potentially setting up mechanisms for horizontal scaling of components like the API layer or embedding service.

### 3.6 Phase 6: Cognitive System Implementation

Phase 6 integrates the CRAG components (Energy/Entropy Management, Sleep Cycles, Dual-Graph Memory) as detailed in `Cognition.md`. This involves:

1. **Core CRAG Framework**: Implementing the foundational components of the cognitive system:
   - Energy and Entropy Model for tracking system resources and state
   - Multi-phase Sleep Cycle mechanisms for maintenance and optimization
   - Dual-Graph Memory architecture combining Experience and Memory Graphs

2. **Integration Points**: Connecting CRAG to core RAG components:
   - Energy tracking for API requests, searches, and agent operations
   - Sleep cycle scheduling based on system workload and state
   - Enhanced retrieval utilizing both memory graphs
   - Energy-aware error handling and recovery strategies

3. **Knowledge Bridge Mechanism**: Developing the bidirectional knowledge transfer between Experience and Memory Graphs, particularly for sleep-cycle knowledge consolidation.

4. **Configuration System**: Creating a flexible configuration framework for CRAG parameters:
   - Energy depletion and recovery rates for different operation types
   - State transition thresholds and optimization levels
   - Sleep cycle durations and trigger conditions
   - Memory graph schema and relationship types

5. **Monitoring and Visualization**: Exposing cognitive state via the API and potentially providing visualization tools for the dual-graph memory system.

**(See `Cognition.md` for comprehensive technical specifications of all CRAG components.)**

## 4. Testing & Validation

The Testing & Validation section describes the comprehensive strategy to ensure the RAG engine is reliable, performs correctly, and meets specified requirements.

### 4.1 Test Methodology

The Test Methodology outlines the overall approach, including unit tests (for individual functions/classes), integration tests (for interactions between components like API-Search-DB), end-to-end tests (simulating user workflows), and performance tests. It emphasizes automated testing within a CI/CD pipeline.

### 4.2 Performance Benchmarks

Performance Benchmarks define specific, measurable targets and tests to evaluate system speed, throughput, and scalability. This includes measuring query latency under load, ingestion throughput, agent processing time, and resource utilization against predefined goals (e.g., p95 query latency < 100ms).

### 4.3 Compliance Validation

Compliance Validation ensures the system adheres to relevant standards, regulations, or organizational policies. This might involve checks for data privacy (like GDPR), security best practices, logging requirements, or specific constraints related to the data being processed.

## 5. Operational Procedures

The Operational Procedures section provides guidelines for deploying, managing, and maintaining the RAG engine in a production environment.

### 5.1 Deployment Guidelines

Deployment Guidelines outline the recommended steps and configurations for installing and setting up the system. This includes infrastructure requirements (servers, databases), dependency installation, configuration file management, initial data loading procedures, and recommended deployment strategies (e.g., Docker, Kubernetes).

### 5.2 Monitoring & Maintenance

Monitoring & Maintenance procedures detail the ongoing tasks required to ensure system reliability and optimal performance. This includes setting up monitoring dashboards and alerts (using tools from Section 2.8), defining regular maintenance tasks (like index rebuilding, log rotation, backups), and procedures for diagnosing and resolving common issues. CRAG's sleep cycles automate some maintenance.

### 5.3 Scaling Considerations

Scaling Considerations address strategies for handling increased workloads and data volumes. This includes options for vertical scaling (increasing resources on existing nodes) and horizontal scaling (adding more nodes) for different components (API servers, embedding services, vector database shards), discussing trade-offs and potential bottlenecks.

## 6. Conclusion

The Conclusion summarizes the key features, architectural design, and anticipated benefits of the proposed Ultra-Fast, Lightweight RAG Engine. It reiterates the value proposition, highlighting the combination of efficient retrieval, recursive self-improvement via agents, and robust cognitive self-regulation provided by the CRAG layer, positioning it as a powerful and adaptable solution for advanced information processing tasks.

## 7. Appendices

### 7.1 Glossary of Terms

| Term                     | Definition                                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **RAG**                  | Retrieval-Augmented Generation, a technique that enhances LLM responses with relevant retrieved information                             |
| **CRAG**                 | Cognitive RAG, the enhanced system with self-regulation and memory capabilities detailed in `Cognition.md`                              |
| **Embedding**            | Vector representation of text in high-dimensional space                                                                               |
| **HNSW**                 | Hierarchical Navigable Small World, an algorithm for approximate nearest neighbor search                                                |
| **ANN**                  | Approximate Nearest Neighbors, techniques for efficient similarity search                                                               |
| **PCA**                  | Principal Component Analysis, dimensionality reduction technique                                                                      |
| **Experience Graph**     | Associative memory structure within CRAG for episodic memory representation and pattern recognition (See `Cognition.md` Section 5.1)    |
| **Memory Graph**         | Structured knowledge representation within CRAG using entities and semantic relationships (See `Cognition.md` Section 5.2)              |
| **Agent**                | Specialized component that performs specific operations on content                                                                    |
| **Chunking**             | Process of dividing documents into smaller sections for processing                                                                    |
| **Energy**               | Computational resource metric in the CRAG cognitive system, depleted by operations and restored during sleep (See `Cognition.md` Section 2.1) |
| **Entropy**              | Measure of system disorder in the CRAG cognitive system, reduced during sleep cycles (See `Cognition.md` Section 2.1)                   |
| **Sleep Cycle**          | Period of system maintenance and optimization in CRAG, analogous to biological sleep (See `Cognition.md` Section 2.3 & 4)               |
| **System State**         | Operational state of the CRAG system based on Energy/Entropy (e.g., OPTIMAL, FATIGUED) (See `Cognition.md` Section 2.2)                |
| **Vector Database**      | Storage system optimized for vector embeddings                                                                                        |
| **LLM**                  | Large Language Model, an AI system trained on vast text data capable of generating human-like text                                      |
| **Semantic Similarity**  | Measure of meaning-based resemblance between texts beyond keyword matching                                                            |
| **Cosine Similarity**    | Vector similarity metric measuring angle between vectors, widely used for embedding comparison                                        |
| **Latent Space**         | Abstract multi-dimensional space where embeddings exist and semantic relationships are preserved                                        |
| **Inference**            | Process of generating predictions or outputs from a trained model given new inputs                                                    |
| **Token**                | Basic unit of text processing in language models, can be words, subwords, or characters                                               |
| **Context Window**       | Maximum amount of text a language model can process at once during operation                                                          |
| **Zero-shot Learning**   | AI's ability to make predictions for classes or tasks it hasn't seen during training                                                  |
| **Few-shot Learning**    | AI's ability to learn from very few examples of a new task or concept                                                                 |
| **Knowledge Graph**      | Network structure representing semantic relationships between entities as a graph (The Memory Graph is a specific type)                 |
| **Attention Mechanism**  | Neural network component that allows models to focus on relevant parts of input data                                                  |
| **Dimensionality Reduction** | Techniques to reduce vector dimensions while preserving information relationships                                                     |

### 7.2 Project Directory Structure

The project follows a standard Python project layout to ensure clarity, maintainability, and ease of navigation.

```plaintext
oarc-rag/
├── .github/              # GitHub Actions workflows (CI/CD, testing)
│   └── workflows/
├── configs/              # Configuration files (e.g., YAML, TOML) for system parameters, energy model, etc.
├── data/                 # Default directory for persistent data (e.g., Parquet files from oarc-crawlers, DB files) - excluded by .gitignore
├── docs/                 # Project documentation (like this file, Cognition.md)
├── examples/             # Usage examples and demonstration scripts
├── src/                  # Main source code for the OARC-RAG engine
│   ├── oarc_rag/         # Core package directory
│   │   ├── core/         # Core components (embedding, search, vector ops)
│   │   ├── agents/       # Agent implementations
│   │   ├── storage/      # Data storage interfaces and implementations (incl. vector DB interaction)
│   │   ├── api/          # FastAPI application and endpoints
│   │   ├── cognitive/    # CRAG system implementation (energy, sleep, graphs, state machine)
│   │   ├── pipeline/     # Document processing pipeline stages (chunking, normalization, multimodal handlers)
│   │   ├── acquisition/  # Coordination logic for oarc-crawlers and monitoring (ResourceCollector)
│   │   ├── utils/        # Utility functions and classes
│   │   └── __init__.py
│   └── __init__.py
├── tests/                # Unit, integration, and performance tests
│   ├── unit/
│   ├── integration/
│   └── performance/
├── .gitignore            # Specifies intentionally untracked files
├── LICENSE               # Project license file (e.g., Apache 2.0)
├── pyproject.toml        # Project metadata, dependencies, and build configuration
├── README.md             # Top-level project description and setup guide
└── requirements.txt      # (Optional) Pinned dependencies for specific environments
```

---
