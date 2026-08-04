# guards against giant functions (up to 484k chars seen in the dataset)
MAX_CODE_BYTES = 30_000
MAX_NODES = 2_000

# heuristic edge types derived from the AST (no external tool)
# AST_CHILD    : syntactic parent -> child
# NEXT_SIBLING : consecutive named siblings (execution-order approximation)
# USE_DEF      : consecutive occurrences of the same identifier (data-flow approximation)
AST_CHILD = 0
NEXT_SIBLING = 1
USE_DEF = 2

# number of base (forward) edge types; reverse edges get type + NUM_BASE_EDGE_TYPES
NUM_BASE_EDGE_TYPES = 3

# total edge-type vocabulary once forward+reverse are distinguished (see _to_data)
NUM_EDGE_TYPES = 2 * NUM_BASE_EDGE_TYPES
