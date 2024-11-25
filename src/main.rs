use crate::geometric_centralities::GeometricCentralities;
use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use log::info;
use webgraph::prelude::BvGraph;


mod geometric_centralities;


#[derive(Parser, Debug)]
#[command(about = "Benchmarks geometric centralities", long_about = None)]
struct Args {
    #[arg(short, long)]
    path: String,

    #[clap(long, short)]
    all: bool,

    #[clap(long, short)]
    generic: bool,

    #[clap(long, short)]
    generic_no_known: bool,

    #[clap(long, short)]
    no_generic: bool,

    #[clap(long, short)]
    generic_dist_vec: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = Args::parse();
    let graph = BvGraph::with_basename(args.path)
        .load()
        .expect("Failed loading graph");
    let mut geom = GeometricCentralities::new(&graph, 0);

    if args.all || args.generic {
        info!("-------------- Geom generic --------------");
        geom.compute_generic(&mut ProgressLogger::default());
    }

    if args.all || args.generic_no_known {
        info!("-------------- Geom generic no known --------------");
        geom.compute_generic_no_known(&mut ProgressLogger::default());
    }

    if args.all || args.generic_dist_vec {
        info!("-------------- Geom generic distance vector --------------");
        geom.compute_generic_dist_vec(&mut ProgressLogger::default());
    }

    if args.all || args.no_generic {
        info!("-------------- Geom no generic --------------");
        geom.compute(&mut ProgressLogger::default());
    }

    info!("Done");

    Ok(())
}
