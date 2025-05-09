use clap::Parser;
use dsi_progress_logger::ConcurrentWrapper;
use geometric_centralities::betweenness::BetweennessCentrality;
use log::info;
use webgraph::prelude::BvGraph;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks geometric centralities", long_about = None)]
struct Args {
    #[arg(short = 'p', long)]
    path: String,

    #[arg(short = 't', long, default_value = "0")]
    threads: usize,
}

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();

    info!("Args: {:?}", args);

    let graph = BvGraph::with_basename(args.path)
        .load()
        .expect("Failed loading graph");
    let mut betw = BetweennessCentrality::new(&graph, 0);

    info!("-------------- Computing betweenness --------------");
    betw.compute(&mut ConcurrentWrapper::new());

    info!("Done");

    Ok(())
}
