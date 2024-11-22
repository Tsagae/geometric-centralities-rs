/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */
use anyhow::Result;
use clap::Parser;
use common_traits::Number;
use criterion::Criterion;
use dsi_progress_logger::ProgressLogger;
use geometric_centralities_rs::geometric_centralities::{GeometricCentralityResult, DEFAULT_ALPHA};
use std::collections::VecDeque;
use std::time::Duration;
use webgraph::prelude::{BvGraph, RandomAccessGraph};
use webgraph::traits::SequentialLabeling;
use webgraph_algo::prelude::breadth_first::{EventPred, Seq};
use webgraph_algo::traits::Sequential;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks bfs", long_about = None)]
struct Args {
    #[arg(short, long, default_value = "5")]
    duration: usize,

    #[arg(short, long)]
    path: String,

    #[arg(short, long, default_value = "0")]
    start_node: usize,
}

pub fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let mut args = Args::parse();

    let graph = BvGraph::with_basename(&args.path)
        .load()
        .expect("Failed loading graph");

    graph_wrapper(&graph, args);

    Ok(())
}

fn graph_wrapper(graph: &impl RandomAccessGraph, args: Args) {
    let mut c = Criterion::default()
        .with_output_color(true)
        .measurement_time(Duration::from_secs(args.duration as u64));

    let mut distances = Vec::new();
    distances.resize(graph.num_nodes(), -1);

    c.bench_function("non-generic-bfs", |_| {
        single_visit(&graph, args.start_node, &mut VecDeque::new(), &mut distances);
    });

    let mut bfs = Seq::new(graph);
    c.bench_function("generic-bfs", |_| {
        single_visit_generic(args.start_node, &mut bfs);
    });

    c.final_summary();
}

fn single_visit<G: RandomAccessGraph>(
    graph: &G,
    start: usize,
    queue: &mut VecDeque<usize>,
    distances: &mut [i64],
) -> GeometricCentralityResult {
    let mut closeness = 0f64;
    let mut harmonic = 0f64;
    let lin;
    let mut exponential = 0f64;
    let mut reachable: usize = 0;

    let base = DEFAULT_ALPHA;

    queue.clear();
    distances.fill(-1);

    distances[start] = 0;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        reachable += 1;
        let d = distances[node] + 1;
        let hd = 1f64 / d as f64;
        let ed = base.pow(d as f64);

        for successor in graph.successors(node) {
            if distances[successor] == -1 {
                queue.push_back(successor);
                distances[successor] = d;
                closeness += d as f64;
                harmonic += hd;
                exponential += ed;
            }
        }
    }
    if closeness == 0f64 {
        lin = 1f64;
    } else {
        closeness = 1f64 / closeness;
        lin = reachable as f64 * reachable as f64 * closeness;
    }
    GeometricCentralityResult {
        closeness,
        harmonic,
        lin,
        exponential,
        reachable,
    }
}

fn single_visit_generic<G: RandomAccessGraph>(
    start: usize,
    bfs: &mut Seq<&G>,
) -> GeometricCentralityResult {
    let mut closeness = 0f64;
    let mut harmonic = 0f64;
    let lin;
    let mut exponential = 0f64;
    let mut reachable: usize = 0;

    bfs.reset_no_rayon();
    bfs.visit(
        start,
        |args| {
            let base = DEFAULT_ALPHA;
            match args {
                EventPred::Init { root } => {
                    eprintln!("starting from {root}");
                    Ok::<(), ()>(())
                }
                EventPred::Known { .. } => Ok(()),
                EventPred::Unknown { distance, .. } => {
                    let d = distance;
                    reachable += 1;
                    if d == 0 {
                        //Skip first
                        return Ok(());
                    }
                    let hd = 1f64 / d as f64;
                    let ed = base.pow(d as f64);
                    closeness += d as f64;
                    harmonic += hd;
                    exponential += ed;
                    Ok(())
                }
            }
        },
        &mut Option::<ProgressLogger>::None,
    )
        .expect("Error in bfs");
    if closeness == 0f64 {
        lin = 1f64;
    } else {
        closeness = 1f64 / closeness;
        lin = reachable as f64 * reachable as f64 * closeness;
    }
    GeometricCentralityResult {
        closeness,
        harmonic,
        lin,
        exponential,
        reachable,
    }
}