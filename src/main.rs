use clap::Parser;
use common_traits::Sequence;
use dsi_progress_logger::{ConcurrentWrapper, ProgressLogger};
use geometric_centralities::betweenness::BetweennessCentrality;
use geometric_centralities::geometric::GeometricCentralities;
use log::info;
use std::env;
use std::fmt::Display;
use std::io::Write;
use webgraph::prelude::{BvGraph, VecGraph};
use webgraph::traits::RandomAccessGraph;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks geometric centralities", long_about = None)]
struct MainArgs {
    #[arg(short = 'p', long)]
    path: String,

    #[arg(short = 's', long)]
    save: bool,

    #[arg(long)]
    parallel: bool,

    #[arg(short = 't', long, default_value = "0")]
    threads: usize,

    #[arg(short = 'g', long, default_value = "100")]
    granularity: usize,

    #[arg(long)]
    geometric: bool,

    #[arg(long)]
    betweenness: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = MainArgs::parse();
    info!("Args: {:?}", args);

    let graph_name = args.path.split("/").last().unwrap();
    let path = env::current_dir()?;
    let current_path = path.display().to_string();
    let mut results_dir = current_path.clone();

    if args.save {
        println!("current path: {current_path}");
        results_dir = format!("{current_path}/{graph_name}_rustresults");
        println!("results_dir: {results_dir}");
        std::fs::create_dir(&results_dir).expect("Can't create directory. It may already exist");
    }

    let graph = BvGraph::with_basename(&args.path)
        .load()
        .expect("Failed loading graph");

    if args.geometric {
        let mut geom = GeometricCentralities::new(&graph, args.threads);

        if args.parallel {
            println!("Computing geometric centralities with parallel visit");
            geom.compute_all_par_visit(&mut ProgressLogger::default(), args.granularity);
        } else {
            println!("Computing geometric centralities with sequential visit");
            geom.compute(&mut ConcurrentWrapper::new());
        }

        if args.save {
            write_nums_to_file(&results_dir, "closeness", geom.closeness.iter());
            write_nums_to_file(&results_dir, "lin", geom.lin.iter());
            write_nums_to_file(&results_dir, "exponential", geom.exponential.iter());
            write_nums_to_file(&results_dir, "harmonic", geom.harmonic.iter());
            write_nums_to_file(&results_dir, "reachable", geom.reachable.iter());
        }
    }

    if args.betweenness {
        let mut betw = BetweennessCentrality::new(&graph, args.threads);
        betw.compute(&mut ConcurrentWrapper::with_threshold(500));
        
        if args.save {
            write_nums_to_file(&results_dir, "betweenness", betw.betweenness.iter());
        }
    }
    info!("Done");

    Ok(())
}

fn decompress_graph(g: impl RandomAccessGraph) -> VecGraph {
    let mut vg = VecGraph::new();
    for node in 0..g.num_nodes() {
        vg.add_node(node);
        vg.add_arcs(
            g.successors(node)
                .into_iter()
                .map(|successor| (node, successor)),
        );
    }
    vg
}

fn write_nums_to_file<T: Display>(base_path: &str, filename: &str, nums: impl Iterator<Item = T>) {
    let filepath = format!("{base_path}/{filename}");
    println!("filepath: {filepath}");
    let mut f = std::fs::File::create(&filepath).expect(&format!("Cannot create file {filename}"));
    let mut string_to_write: String = String::new();
    nums.for_each(|n| string_to_write.push_str(&format!("{}\n", n)));
    f.write(string_to_write.as_bytes())
        .expect(&format!("Failed writing to file {filename}"));
}
