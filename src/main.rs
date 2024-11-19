use crate::geometric_centralities::GeometricCentralities;
use dsi_progress_logger::{ProgressLogger};
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
    info!("-------------- Default reset --------------");
    geom.compute_with_atomic_counter_out_channel(|bfs| bfs.reset(), &mut ProgressLogger::default());

    info!("-------------- Reset no rayon --------------");
    geom.compute_with_atomic_counter_out_channel(|bfs| bfs.reset_no_rayon(), &mut ProgressLogger::default());

    info!("-------------- Reset min_len_iter 10000 --------------");
    geom.compute_with_atomic_counter_out_channel(|bfs| bfs.reset_min_len_iter(10_000), &mut ProgressLogger::default());

    info!("-------------- Reset min_len_iter 10000000 --------------");
    geom.compute_with_atomic_counter_out_channel(|bfs| bfs.reset_min_len_iter(10_000_000), &mut ProgressLogger::default());

    info!("\nDone");
}
