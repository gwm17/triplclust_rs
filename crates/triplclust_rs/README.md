# triplclust_rs

The Rust implementation of [`triplclust`](https://github.com/cdalitz/triplclust).

## Changes over original

### Collapsing labels

When labels are translated from triplets to points, each point may receive more than
one label. In the original implementation, each label is stored and no decision is made
as to how to handle the overlap of clustering labels. In `triplclust_rs`, we implement
a solution where labels are tallied, and the label which is assigned most frequently
to a given point is returned as the most accurate label. Other solutions could be
explored as well.
