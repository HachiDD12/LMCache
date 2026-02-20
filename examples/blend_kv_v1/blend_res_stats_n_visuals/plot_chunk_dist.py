import json
import matplotlib.pyplot as plt

with open("chunk_dist_v037_django_ttft_chunk_20251208_052234.json") as f:
    data = json.load(f)

plt.hist(data["chunk_sizes"], bins=50)
plt.xlabel("Chunk Size (tokens)")
plt.ylabel("Frequency")
plt.title(f"Chunk Size Distribution (n={data['total_chunks']}, mean={data['mean_size']:.1f})")
plt.savefig("chunk_dist.png", dpi=300, bbox_inches='tight')