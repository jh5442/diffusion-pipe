from datasets import Dataset
import glob

# Load all shards together
files = sorted(glob.glob("/picassox/intelligent-cpfs/pixocial/jin.huang/data/diff_pipe/"
                         "wan_2.2_debug/cache/wan/cache_1.587x1216x768x33/latents_f987cfe3c6ec35b0_*.arrow"))

# Each .arrow file is a shard → load one or combine
print("Shards found:", files)

# Example: load first shard
ds = Dataset.from_file(files[0])
print("Schema:", ds.features)     # column names + types
print("Num rows:", len(ds))
# print("First row:", ds[0])        # sample row

row = ds[0]   # first (and only) row
print(row.keys())
print(row["image_spec"])         # full list of strings
print(len(row["image_spec"]))    # how many entries
print(row["image_spec"][:5])     # first 5 items

import numpy as np

latents = np.array(row["latents"], dtype=np.float32)
print("Latents shape:", latents.shape) # Latents shape: (16, 9, 96, 152)
print("Latents dtype:", latents.dtype)


print("Num samples:", len(row["image_spec"]))
print("Example image_spec:", row["image_spec"][:3])
print("Example caption:", row["caption"][:3])
print("Latent[0] shape:", latents[0].shape)

print("Row keys:", row.keys())
print("Latent[0] stats:", latents[0].shape, latents[0].min(), latents[0].max())
print("Latent[1] stats:", latents[1].shape, latents[1].min(), latents[1].max())
print("Num clips in this shard:", len(row["latents"]))


