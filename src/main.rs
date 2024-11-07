use crate::geometric_centralities::GeometricCentralities;
use webgraph::prelude::BvGraph;

mod geometric_centralities;

fn main() {
    env_logger::init();
    let path = "/home/matteo/Documents/tesi/example_graphs/uk-2007-05@100000/".to_owned();
    let graph = BvGraph::with_basename(path + "uk-2007-05@100000")
        .load()
        .expect("Failed loading graph");
    let mut geom = GeometricCentralities::new(&graph, 0, true);
    geom.compute_with_par_iter(10);
    println!("Done");
}
