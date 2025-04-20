use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use geometric_centralities::geometric::GeometricCentralities;
use log::info;
use std::env;
use std::fmt::Display;
use std::io::Write;
use webgraph::prelude::BvGraph;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks geometric centralities", long_about = None)]
struct MainArgs {
    #[arg(short = 'p', long)]
    path: String,
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

    println!("current path: {current_path}");
    let results_dir = format!("{current_path}/{graph_name}_rustresults");
    println!("results_dir: {results_dir}");
    std::fs::create_dir(&results_dir).expect("Can't create directory. It may already exist");

    let graph = BvGraph::with_basename(args.path)
        .load()
        .expect("Failed loading graph");
    let mut geom = GeometricCentralities::new(&graph, 0);

    geom.compute(&mut ProgressLogger::default());

    write_nums_to_file(&results_dir, "closeness", geom.closeness.iter());
    write_nums_to_file(&results_dir, "lin", geom.lin.iter());
    write_nums_to_file(&results_dir, "exponential", geom.exponential.iter());
    write_nums_to_file(&results_dir, "harmonic", geom.harmonic.iter());
    write_nums_to_file(&results_dir, "reachable", geom.reachable.iter());
    info!("Done");

    Ok(())
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
