use crate::geometric_centralities::GeometricCentralities;
use dsi_progress_logger::ProgressLogger;
use log::info;
use std::env;
use webgraph::prelude::BvGraph;
use webgraph_algo::traits::Sequential;

mod geometric_centralities;

fn main() {
    env_logger::init();
    let args: Vec<_> = env::args().collect();
    let graph = BvGraph::with_basename(args.get(1).expect("Missing path argument"))
        .load()
        .expect("Failed loading graph");
    let mut geom = GeometricCentralities::new(&graph, 0);

    info!("-------------- Geom generic --------------");
    geom.compute_with_atomic_counter_out_channel_generic(&mut ProgressLogger::default());

    info!("-------------- Geom no generic --------------");
    geom.compute_with_atomic_counter_out_channel(&mut ProgressLogger::default());

    info!("\nDone");
}
