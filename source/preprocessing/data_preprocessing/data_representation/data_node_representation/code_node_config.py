import os
import zlib


VOCAB_PATH = os.path.join(os.path.dirname(__file__), "node_type_vocab.json")

# C functions commonly known as sources of memory-safety vulnerabilities
DANGEROUS_FUNCS = {
    "strcpy", "strcat", "sprintf", "vsprintf", "gets", "scanf", "sscanf",
    "memcpy", "memmove", "memset", "strncpy", "strncat", "snprintf",
    "malloc", "calloc", "realloc", "free", "alloca",
    "system", "exec", "execl", "execlp", "execvp", "popen",
}

# number of handcrafted token features (see _token_features)
NUM_TOKEN_FEATURES = 5

# learned token embedding: tokens are hashed into a fixed number of buckets so the
# model can learn token semantics without an external tokenizer. Bucket 0 is reserved
# for the empty token (non-leaf / unnamed nodes).
NUM_TOKEN_BUCKETS = 4096


def token_bucket(token: str) -> int:
    if not token:
        return 0
    # crc32 is deterministic across runs (unlike Python's salted hash)
    return 1 + zlib.crc32(token.encode("utf-8", "replace")) % (NUM_TOKEN_BUCKETS - 1)
