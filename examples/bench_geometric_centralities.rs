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

    #[arg(long)]
    parallel: bool,

    #[arg(short = 't', long, default_value = "0")]
    threads: usize,

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

    if args.parallel {
        info!("-------------- Computing with parallel visit --------------");
        geom.compute_all_par_visit(&mut ProgressLogger::default(), args.granularity);
    } else {
        info!("-------------- Computing with sequential visit --------------");
        geom.compute(&mut ProgressLogger::default());
    }

    info!("Done");

    Ok(())
}
