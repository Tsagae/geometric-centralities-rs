use dsi_progress_logger::no_logging;
use geometric_centralities::betweenness::BetweennessCentrality;
use webgraph::prelude::VecGraph;
use geometric_centralities::utils::transpose_arc_list;

fn main() {
    let mut temp_g = VecGraph::new();
    for i in 0..5 {
        temp_g.add_node(i);
    }
    let mut arcs = vec![];
    arcs.append(&mut vec![
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 4),
        (4, 0),
    ]);
    arcs.append(&mut vec![(5, 0), (5, 4), (6, 4), (4, 6), (4, 7)]);

    let graph = VecGraph::from_arcs(arcs.clone());
    let graph_t = VecGraph::from_arcs(transpose_arc_list(arcs));
    let mut betweenness = BetweennessCentrality::new(&graph, 1);
    betweenness.compute(no_logging!());

    betweenness
        .betweenness
        .iter()
        .zip(0..betweenness.betweenness.len())
        .for_each(|(item, i)| println!("{i}: {item}"));
}
