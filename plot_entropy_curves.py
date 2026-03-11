#!/usr/bin/env python3
import json
import sys
import matplotlib.pyplot as plt

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    rows = json.load(f)

# pick first successful example
row = next(r for r in rows if "error" not in r and r.get("token_entropy"))
ent = [x for x in row["token_entropy"] if x is not None]

plt.figure(figsize=(8, 4))
plt.plot(range(len(ent)), ent)
plt.xlabel("Generated token position")
plt.ylabel("Entropy")
plt.title(f"{row['question_id']} | max_tokens={row['max_tokens']}")
plt.tight_layout()
plt.show()

print("Question:", row["question"])
print("Response:", row["response_text"])