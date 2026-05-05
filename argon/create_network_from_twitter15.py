import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_bigcn_data(file_path):
    graphs = defaultdict(nx.DiGraph)
    roots = {}

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            root_id = parts[0]
            parent_str = parts[1]
            node_id = parts[2]

            graphs[root_id].add_node(node_id)

            if parent_str == "None":
                roots[root_id] = node_id
            else:
                graphs[root_id].add_edge(parent_str, node_id)

    return graphs, roots


def visualize_graphs(cascade_graphs, cascade_roots):
    for root_id, graph in cascade_graphs.items():
        plt.figure(figsize=(8, 6))

        # Calculate a layout for the nodes (how they are positioned on screen)
        # spring_layout works nicely for small networks
        pos = nx.spring_layout(graph, seed=42)

        # Draw the standard nodes (replies) and edges
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=2000,
            font_size=12,
            font_weight="bold",
            arrowsize=20,
            arrows=True,
        )

        # Highlight the root node (source tweet) in green
        root_node = cascade_roots[root_id]
        nx.draw_networkx_nodes(
            graph, pos, nodelist=[root_node], node_color="lightgreen", node_size=2000
        )

        plt.title(
            f"Propagation Tree for Root ID: {root_id}\n(Green = Source Tweet, Blue = Replies/Retweets)"
        )

        # This function pauses the script and pops up the interactive graphic!
        # You will need to close the window for the next graph to appear.
        plt.show()


if __name__ == "__main__":

    # Parse the data
    cascade_graphs, cascade_roots = parse_bigcn_data(
        "BiGCN/data/Twitter15/data.TD_RvNN.vol_5000.txt"
    )

    # Trigger the graphical pop-ups
    visualize_graphs(cascade_graphs, cascade_roots)
