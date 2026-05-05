from __future__ import annotations

import argparse

from .argon_adapter import load_argon_graph
from .text_embeddings import BowConfig, build_text_index, embed_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for argon text embeddings.")
    parser.add_argument("--argon-edge-path", required=True)
    parser.add_argument("--argon-text-path", required=True)
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--vocab-path", type=str, default=None)
    args = parser.parse_args()

    argon = load_argon_graph(
        args.argon_edge_path,
        args.argon_text_path,
        drop_missing_text=False,
        max_nodes=args.max_nodes,
    )

    bow_config = BowConfig(max_features=5000, min_df=1)
    result = embed_texts(argon.node_texts, model_name="bow", vocab_path=args.vocab_path, bow_config=bow_config)

    print(f"Embeddings shape: {tuple(result.embeddings.shape)}")

    if argon.has_text.any():
        text_ids = [nid for nid, has_text in zip(argon.node_ids, argon.has_text) if has_text]
        text_values = [text for text, has_text in zip(argon.node_texts, argon.has_text) if has_text]
        text_embeddings = result.embeddings[argon.has_text]
        index = build_text_index(text_embeddings, text_ids, text_values, use_faiss=False)
        nearest = index.query(result.embeddings[:5].cpu().numpy(), k=1)
        for i, row in enumerate(nearest):
            if not row:
                continue
            nid, score, text = row[0]
            print(f"Sample {i}: nearest={nid} score={score:.3f} text={text[:60]}")


if __name__ == "__main__":
    main()
