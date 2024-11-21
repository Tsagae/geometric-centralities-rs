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
    generic: bool,

    #[clap(long, short)]
    generic_no_known: bool,

    #[clap(long, short)]
    no_generic: bool,
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

    if args.generic {
        info!("-------------- Geom generic --------------");
        geom.compute_with_atomic_counter_out_channel_generic(&mut ProgressLogger::default());
    }

    if args.generic_no_known {
        info!("-------------- Geom generic no known --------------");
        geom.compute_with_atomic_counter_out_channel_generic_no_known(&mut ProgressLogger::default());
    }

    if args.no_generic {
        info!("-------------- Geom no generic --------------");
        geom.compute_with_atomic_counter_out_channel(&mut ProgressLogger::default());
    }
   
    info!("Done");

    Ok(())
}
