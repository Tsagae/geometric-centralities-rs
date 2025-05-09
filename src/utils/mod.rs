use webgraph::prelude::VecGraph;

#[allow(dead_code)]
pub(crate) fn transpose_arc_list(
    arcs: impl IntoIterator<Item = (usize, usize)>,
) -> impl IntoIterator<Item = (usize, usize)> {
    arcs.into_iter().map(|(a, b)| (b, a))
}

#[allow(dead_code)]
pub(crate) fn new_directed_cycle(num_nodes: usize) -> VecGraph {
    let mut graph = VecGraph::new();
    for i in 0..num_nodes {
        graph.add_node(i);
    }
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if (i + 1) % num_nodes == j {
                graph.add_arc(i, j);
            }
        }
    }
    graph
}

#[allow(dead_code)]
pub(crate) fn new_clique(num_nodes: usize) -> VecGraph {
    let mut graph = VecGraph::new();
    for i in 0..num_nodes {
        graph.add_node(i);
    }
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j {
                graph.add_arc(i, j);
            }
        }
    }
    graph
}
