import os


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