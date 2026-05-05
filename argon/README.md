## file descriptions

so the `argon` directory is an exploration of the BiGCN datasets, they seem the most promising of already created datasets. see notes in README.md about their potential application.

- `check_message_coverage.py` aims to check which of the tweets in BiGCN graph data have corresponding plain text messages in `source_tweets_t15.tsv` which was sourced from Kaggle
- `create_network_from_twitter15.py` loads networkx graphs from the BiGCN dataset and displays them to the user. as previously noted, there are many individual graphs that are seperate from one another.
    - the typical model for GNN is transductive learning, in which the model learns over the entire graph with some nodes/edges masked out to create test/val/train splits
    - inductive learning on the other hand is something that could be applied here where the model learns on a train graph (or set of graphs) and tries to generalize to other graphs that have not been seen before

## sources

- https://github.com/TianBian95/BiGCN/tree/master/data
- https://www.kaggle.com/datasets/syntheticprogrammer/rumor-detection-acl-2017
