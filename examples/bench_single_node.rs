use clap::{arg, Parser};
use criterion::{black_box, Criterion};
use dsi_progress_logger::no_logging;
use geometric_centralities::geometric::GeometricCentralities;
use log::info;
use std::time::Duration;
use webgraph::prelude::BvGraph;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks single node geometric centrality", long_about = None)]
struct Args {
    #[arg(short = 'p', long)]
    path: String,

    #[arg(short = 't', long, default_value = "0")]
    threads: usize,

    #[arg(short = 'g', long, default_value = "100")]
    granularity: usize,

    #[arg(short = 'n', long, default_value = "0")]
    node: usize,

    #[arg(short, long, default_value = "60")]
    duration: usize,
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

    let mut c = Criterion::default()
        .with_output_color(true)
        .measurement_time(Duration::from_secs(args.duration as u64));

    c.bench_function("single node visit", |b| {
        b.iter(|| {
            geom.compute_single_node_par_visit(
                black_box(args.node),
                no_logging!(),
                black_box(args.granularity),
            )
        })
    });

    c.final_summary();
    Ok(())
}
