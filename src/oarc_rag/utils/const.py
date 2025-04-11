"""
Constants used throughout the oarc_rag project.
"""

# Log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s%(context)s"

# Exit codes
SUCCESS = 0
FAILURE = 1

# Version
VERSION = "0.1.0"

# Ollama configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:latest"
DEFAULT_EMBEDDING_MODEL = "llama3.1:latest"

# Default resource limits
DEFAULT_MAX_CONCURRENT_REQUESTS = 5
DEFAULT_MAX_FILE_SIZE_MB = 10.0
DEFAULT_MAX_CONTENT_LENGTH = 10000

# Default filesystem paths
DEFAULT_OUTPUT_DIRNAME = "oarc_rag_output"
DEFAULT_DATA_DIRNAME = "oarc_rag_data"
DEFAULT_CONFIG_DIRNAME = ".oarc_rag"

# Default AI generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TOP_P = 1.0

# Operational modes from Big_Brain.md
OPERATIONAL_MODE_AWAKE = "awake"
OPERATIONAL_MODE_SLEEP = "sleep"
DEFAULT_OPERATIONAL_MODE = OPERATIONAL_MODE_AWAKE

# Vector database and operations defaults
DEFAULT_VECTOR_DIMENSION = 4096
DEFAULT_DISTANCE_METRIC = "cosine"
SUPPORTED_DISTANCE_METRICS = ["cosine", "l2", "ip"]
DEFAULT_HNSW_EF_CONSTRUCTION = 200
DEFAULT_HNSW_EF_SEARCH = 50
DEFAULT_HNSW_M = 16

# Chunking defaults
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
CHUNK_STRATEGIES = ["fixed", "sentence", "paragraph", "semantic"]
DEFAULT_CHUNK_STRATEGY = "paragraph"

# Cache settings
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_CACHE_SIZE = 1000
CACHE_EVICTION_POLICIES = ["lru", "lfu", "fifo", "random", "ttl"]
DEFAULT_CACHE_EVICTION = "lru"

# Monitoring thresholds
DEFAULT_LATENCY_WARNING = 2.0  # seconds
DEFAULT_LATENCY_CRITICAL = 5.0  # seconds
DEFAULT_MEMORY_WARNING = 80.0  # percent
DEFAULT_MEMORY_CRITICAL = 95.0  # percent

# File operations
SUPPORTED_FILE_EXTENSIONS = [
    '.txt', '.md', '.tex', '.rst', '.html',
    '.py', '.js', '.java', '.cpp', '.c',
    '.json', '.yaml', '.yml', '.csv',
    '.pdf', '.docx', '.pptx', '.xlsx'
]

# Context assembly settings
DEFAULT_MAX_CONTEXT_LENGTH = 10000
DEFAULT_SIMILARITY_THRESHOLD = 0.85
OPTIMIZATION_LEVELS = ["low", "balanced", "high"]
DEFAULT_OPTIMIZATION_LEVEL = "balanced"