use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use geometric_centralities::geometric::GeometricCentralities;
use log::info;
use webgraph::prelude::BvGraph;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks geometric centralities", long_about = None)]
struct Args {
    #[arg(short = 'p', long)]
    path: String,

    #[arg(short = 'a', long)]
    all_single_node: bool,

    #[arg(short = 'd', long)]
    disable_normal_compute: bool,

    #[arg(short = 'g', long, default_value = "100")]
    granularity: usize,
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
    let mut geom = GeometricCentralities::new(&graph, 0);

    if !args.disable_normal_compute {
        info!("-------------- Geom generic --------------");
        geom.compute(&mut ProgressLogger::default());
    }

    if args.all_single_node {
        info!("-------------- Geom single node --------------");
        geom.compute_all_single_node(&mut ProgressLogger::default(), args.granularity);
    }

    info!("Done");

    Ok(())
}
