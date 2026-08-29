#!/usr/bin/env python3
"""Generate the router cache with local BAAI/bge-small-en-v1.5 embeddings."""
import argparse
import json
import os
import tempfile
from pathlib import Path

from fastembed import TextEmbedding

VERSION = 1
DIMENSIONS = 384


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    documents = json.loads(Path(args.input).read_text())
    output = Path(args.output)
    if output.exists():
        cache = json.loads(output.read_text())
        if cache.get("schema_version") != VERSION or cache.get("dimensions") != DIMENSIONS:
            raise SystemExit("existing cache schema/model dimensions are incompatible")
    else:
        cache = {"schema_version": VERSION, "model": "BAAI/bge-small-en-v1.5", "dimensions": DIMENSIONS, "records": []}
    records = {(r["kind"], r["id"]): r for r in cache["records"]}
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=os.environ.get("FASTEMBED_CACHE_DIR"))
    for start in range(0, len(documents), args.batch_size):
        batch = documents[start : start + args.batch_size]
        vectors = model.embed([d["text"] or f'{d["kind"]} {d["id"]}' for d in batch], batch_size=len(batch))
        for document, vector in zip(batch, vectors):
            records[(document["kind"], document["id"])] = {
                "kind": document["kind"],
                "id": document["id"],
                "fingerprint": document["fingerprint"],
                "vector": [float(value) for value in vector],
            }
        cache["records"] = [records[key] for key in sorted(records)]
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=output.parent, prefix=f".{output.name}.", delete=False) as handle:
            json.dump(cache, handle, separators=(",", ":"))
            temporary = Path(handle.name)
        temporary.replace(output)
        print(f"embedded local batch {start // args.batch_size + 1}/{(len(documents) + args.batch_size - 1) // args.batch_size}; cache now has {len(records)} records", flush=True)
    print(f"local embedding cache complete: {len(records)} records, {DIMENSIONS} dimensions")


if __name__ == "__main__":
    main()
