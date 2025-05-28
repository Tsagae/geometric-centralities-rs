use clap::Parser;
use dsi_progress_logger::{ConcurrentWrapper, ProgressLogger};
use geometric_centralities::betweenness::betweenness_centrality;
use geometric_centralities::geometric;
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

    #[arg(short = 'd', long)]
    decompress: bool,

    #[arg(long)]
    parallel: bool,

    #[arg(long)]
    parallel_single_node: bool,
    
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

    let bv_graph = BvGraph::with_basename(&args.path)
        .load()
        .expect("Failed loading graph");

    if args.decompress {
        info!("Decompressing graph...");
        let graph = decompress_graph(bv_graph);
        info!("Graph ready");
        run(graph, args, &results_dir);
    } else {
        run(bv_graph, args, &results_dir);
    }

    info!("Done");

    Ok(())
}

fn run(graph: impl RandomAccessGraph + Sync, args: MainArgs, results_dir: &str) {
    if args.geometric {
        if args.parallel_single_node {
            geometric::compute_single_node_par_visit(&graph, args.threads, 0, &mut ProgressLogger::default());
            return;
        }
        
        let res = if args.parallel {
            geometric::compute_all_par_visit(&graph, args.threads, &mut ProgressLogger::default())
        } else {
            geometric::compute(
                &graph,
                args.threads,
                &mut ConcurrentWrapper::with_threshold(1000),
            )
        };

        if args.save {
            write_to_file(
                &results_dir,
                "closeness",
                float_to_string_iter(res.closeness.iter().map(|&f| f)),
            );
            write_to_file(
                &results_dir,
                "lin",
                float_to_string_iter(res.lin.iter().map(|&f| f)),
            );
            write_to_file(
                &results_dir,
                "exponential",
                float_to_string_iter(res.exponential.iter().map(|&f| f)),
            );
            write_to_file(
                &results_dir,
                "harmonic",
                float_to_string_iter(res.harmonic.iter().map(|&f| f)),
            );
            write_to_file(&results_dir, "reachable", res.reachable.iter());
        }
    }

    if args.betweenness {
        let betweenness = betweenness_centrality::compute(
            &graph,
            args.threads,
            &mut ConcurrentWrapper::with_threshold(1000),
        )
        .unwrap();

        if args.save {
            write_to_file(
                &results_dir,
                "betweenness",
                float_to_string_iter(betweenness.iter().map(|&f| f)),
            );
        }
    }
}

fn float_to_string_iter(iter: impl Iterator<Item = f64>) -> impl Iterator<Item = String> {
    iter.into_iter().map(|f| format!("{f:.20}"))
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

fn write_to_file(base_path: &str, filename: &str, iter: impl Iterator<Item = impl Display>) {
    let filepath = format!("{base_path}/{filename}");
    println!("filepath: {filepath}");
    let mut f = std::fs::File::create(&filepath).expect(&format!("Cannot create file {filename}"));
    let mut string_to_write: String = String::new();
    iter.for_each(|n| string_to_write.push_str(&format!("{}\n", n)));
    f.write(string_to_write.as_bytes())
        .expect(&format!("Failed writing to file {filename}"));
}
