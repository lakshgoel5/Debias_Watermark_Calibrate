"""Creates the secret key (and records the logit-vocab size)"""
import json, os, secrets, sys
from transformers import AutoConfig
from common import MODEL

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
key = {
    "secret": secrets.token_hex(16),
    "vocab_size": AutoConfig.from_pretrained(MODEL).vocab_size,
}
json.dump(key, open(os.path.join(out, "key.json"), "w"))
print("wrote", os.path.join(out, "key.json"))