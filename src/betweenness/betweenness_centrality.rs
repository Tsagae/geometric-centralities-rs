use dsi_progress_logger::ProgressLog;
use itertools::Itertools;
use openmp_reducer::Reducer;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::ThreadPool;
use std::cell::Cell;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Mutex;
use std::thread::available_parallelism;
use webgraph::traits::RandomAccessGraph;
use webgraph_algo::prelude::breadth_first::EventPred;
use webgraph_algo::prelude::Parallel;

pub struct BetweennessCentrality<'a, G: RandomAccessGraph> {
    pub betweenness: Vec<f64>,
    graph: &'a G,
    thread_pool: ThreadPool,
}

impl<G: RandomAccessGraph + Sync> BetweennessCentrality<'_, G> {
    pub fn new(graph: &G, num_of_threads: usize) -> BetweennessCentrality<G> {
        let num_threads = if num_of_threads == 0 {
            usize::from(available_parallelism().unwrap())
        } else {
            num_of_threads
        };
        let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
        thread_pool_builder = thread_pool_builder.num_threads(num_threads);
        let thread_pool = thread_pool_builder
            .build()
            .expect("Error in building thread pool");
        BetweennessCentrality {
            betweenness: Vec::new(),
            graph,
            thread_pool,
        }
    }

    pub fn compute(&mut self, pl: &mut impl ProgressLog) {
        let num_nodes = self.graph.num_nodes();
        let thread_pool = &self.thread_pool;

        pl.item_name("visit")
            .display_memory(false) //TODO: check if this can be enabled https://docs.rs/dsi-progress-logger/latest/dsi_progress_logger/trait.ConcurrentProgressLog.html
            .local_speed(true)
            .expected_updates(Some(num_nodes));

        pl.start(format!(
            "Computing betweenness centrality with {} threads...",
            &self.thread_pool.current_num_threads()
        ));

        let betweenness: Mutex<Vec<f64>> = Mutex::new(vec![0.; num_nodes]);

        let mut bfs = webgraph_algo::visits::breadth_first::ParFairPred::new(self.graph, 100);
        let chunk_size = 1000;
        let cache_node_capacity = 30000;
        for mut chunk in (0..num_nodes).chunks(chunk_size).into_iter() {
            let first_node_in_chunk = chunk.next().unwrap();
            let last_node_in_chunk = chunk.last().unwrap();
            let mut node_cache: HashMap<usize, Vec<usize>> = HashMap::new(); // TODO: Change this with Vec<Option<Vec<usize>>> of size num_of_nodes
            let reducer: Reducer<HashMap<usize, Vec<usize>>> =
                Reducer::new(node_cache, |global, local| {
                    for (&key, val) in local {
                        global.entry(key).or_default().extend(val);
                    }
                });
            bfs.reset();
            bfs.par_visit_with(
                [first_node_in_chunk],
                reducer.share(),
                |shared_reducer, event| {
                    let cache = shared_reducer.as_mut();
                    match event {
                        EventPred::Init { .. } => {}
                        EventPred::Unknown {
                            node,
                            pred,
                            distance: _,
                        } => {
                            if !cache.contains_key(&pred) && cache.len() >= cache_node_capacity {
                                return ControlFlow::<()>::Break(());
                            }
                            cache.entry(pred).or_default().push(node);
                        }
                        EventPred::Known { node, pred } => {
                            if !cache.contains_key(&pred) && cache.len() >= cache_node_capacity {
                                return ControlFlow::<()>::Break(());
                            }
                            cache.entry(pred).or_default().push(node);
                        }
                        EventPred::Done { .. } => {}
                    }
                    ControlFlow::<()>::Continue(())
                },
                thread_pool,
            );

            //let cache_hits = AtomicUsize::new(0);
            //let cache_misses = AtomicUsize::new(0);

            node_cache = reducer.get();
            let par_iter = (first_node_in_chunk..last_node_in_chunk + 1).into_par_iter();
            par_iter.for_each(|curr| {
                thread_local! {
                    static DISTANCE: Cell<Vec<i32>> = Cell::new(Vec::new());
                    static DELTA: Cell<Vec<f64>> = Cell::new(Vec::new());
                    static SIGMA: Cell<Vec<i64>> = Cell::new(Vec::new());
                    static QUEUE: Cell<Vec<usize>> = Cell::new(Vec::new());
                }

                let graph = self.graph;

                let mut distance = DISTANCE.take();
                let mut delta = DELTA.take();
                let mut sigma = SIGMA.take();
                let mut queue = QUEUE.take();

                if graph.outdegree(curr) == 0 {
                    return;
                }
                distance.resize(num_nodes, -1);
                sigma.resize(num_nodes, 0);
                delta.resize(num_nodes, 0.);
                queue.clear();

                distance[curr] = 0;
                sigma[curr] = 1;

                queue.push(curr);
                let mut overflow = false;

                let mut i = 0;
                while i != queue.len() {
                    let node = queue[i];
                    let d = distance[node];
                    debug_assert_ne!(d, -1);
                    let curr_sigma = sigma[node];
                    let mut successors_func = |s| {
                        if distance[s] == -1 {
                            distance[s] = d + 1;
                            queue.push(s);
                            //TODO: maybe use a different error handling. Not debug assert?
                            debug_assert!(Self::check_overflow(&sigma, node, curr_sigma, s));
                            overflow |= sigma[s] > i64::MAX - curr_sigma;
                            sigma[s] += curr_sigma;
                        } else if distance[s] == d + 1 {
                            debug_assert!(Self::check_overflow(&sigma, node, curr_sigma, s));
                            overflow |= sigma[s] > i64::MAX - curr_sigma;
                            sigma[s] += curr_sigma;
                        }
                    };
                    match node_cache.get(&node) {
                        None => {
                            //cache_misses.fetch_add(1, Ordering::Relaxed);
                            graph
                                .successors(node)
                                .into_iter()
                                .for_each(|s| successors_func(s));
                        }
                        Some(cache_successors) => {
                            //cache_hits.fetch_add(1, Ordering::Relaxed);
                            cache_successors.iter().for_each(|&s| successors_func(s))
                        }
                    };
                    i += 1;
                }

                if overflow {
                    panic!("Path count overflow")
                }

                let mut delta_func = |s, d, node, sigma_node| {
                    if distance[s] == d + 1 {
                        delta[node] += (1. + delta[s]) * sigma_node / sigma[s] as f64;
                    }
                };

                for &node in queue[1..].iter().rev() {
                    let d = distance[node];
                    let sigma_node = sigma[node] as f64;
                    match node_cache.get(&node) {
                        None => {
                            //cache_misses.fetch_add(1, Ordering::Relaxed);
                            graph
                                .successors(node)
                                .into_iter()
                                .for_each(|s| delta_func(s, d, node, sigma_node));
                        }
                        Some(cache_successors) => {
                            //cache_hits.fetch_add(1, Ordering::Relaxed);
                            cache_successors
                                .iter()
                                .for_each(|&s| delta_func(s, d, node, sigma_node));
                        }
                    };
                }

                {
                    //TODO: try lock, if it fails update betweenness at next step after overflow check
                    let mut lock_betweenness = betweenness.lock().unwrap();
                    for &node in &queue[1..] {
                        lock_betweenness[node] += delta[node];
                    }
                }
            });
            pl.update_with_count(last_node_in_chunk + 1 - first_node_in_chunk);
            //println!("avg cache hits: {}", cache_hits.into_inner() / chunk_size);
            //println!("avg cache misses: {}", cache_misses.into_inner() / chunk_size);
            //println!("cache size: {}", node_cache.len());
        }

        pl.done_with_count(num_nodes);
        self.betweenness = betweenness.into_inner().unwrap();
    }

    fn check_overflow(sigma: &[i64], node: usize, curr_sigma: i64, s: usize) -> bool {
        if sigma[s] > i64::MAX - curr_sigma {
            panic!("{} > {} ({node} -> {s})", sigma[s], i64::MAX - curr_sigma);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::betweenness::BetweennessCentrality;
    use crate::utils::{new_clique, new_directed_cycle};
    use assert_approx_eq::assert_approx_eq;
    use dsi_progress_logger::no_logging;
    use webgraph::graphs::random::ErdosRenyi;
    use webgraph::prelude::VecGraph;
    use webgraph::traits::SequentialLabeling;

    #[test]
    fn test_path() {
        let g = VecGraph::from_arcs([(0, 1), (1, 2)]);
        let mut centrality = BetweennessCentrality::new(&g, 0);
        centrality.compute(no_logging!());

        assert_approx_eq!(centrality.betweenness[0], 0., 1E-5);
        assert_approx_eq!(centrality.betweenness[1], 1., 1E-5);
        assert_approx_eq!(centrality.betweenness[2], 0., 1E-5);
    }

    #[test]
    fn test_lozenge() {
        let g = VecGraph::from_arcs([(0, 1), (0, 2), (1, 3), (2, 3)]);
        let mut centrality = BetweennessCentrality::new(&g, 0);
        centrality.compute(no_logging!());

        assert_approx_eq!(centrality.betweenness[0], 0., 1E-5);
        assert_approx_eq!(centrality.betweenness[1], 0.5, 1E-5);
        assert_approx_eq!(centrality.betweenness[2], 0.5, 1E-5);
        assert_approx_eq!(centrality.betweenness[3], 0., 1E-5);
    }

    #[test]
    fn test_cycle() {
        for size in [10, 50, 100] {
            let graph = new_directed_cycle(size);
            let mut centrality = BetweennessCentrality::new(&graph, 0);
            centrality.compute(no_logging!());

            let mut expected = Vec::new();
            expected.resize(size, ((size - 1) * (size - 2)) as f64 / 2.0);

            (0..size)
                .for_each(|i| assert_approx_eq!(centrality.betweenness[i], expected[i], 1E-12));
        }
    }

    #[test]
    fn test_clique() {
        for size in [10, 50, 100] {
            let graph = new_clique(size);
            let mut centrality = BetweennessCentrality::new(&graph, 0);
            centrality.compute(no_logging!());

            let expected = vec![0f64; size];

            (0..size)
                .for_each(|i| assert_approx_eq!(centrality.betweenness[i], expected[i], 1E-12));
        }
    }

    #[test]
    fn test_clique_no_bridge_cycle() {
        for p in [10, 50, 100] {
            for k in [10, 50, 100] {
                let mut arcs = Vec::new();
                let mut graph = new_clique(k);
                for i in 0..p {
                    arcs.push((k + i, k + (i + 1) % p));
                }
                graph.add_arcs(arcs);

                let mut centrality = BetweennessCentrality::new(&graph, 0);
                centrality.compute(no_logging!());

                let mut expected = vec![0f64; k + p];
                (0..k).for_each(|i| expected[i] = 0.);
                (k..k + p).for_each(|i| expected[i] = ((p - 1) * (p - 2)) as f64 / 2.0);

                (0..k + p)
                    .for_each(|i| assert_approx_eq!(centrality.betweenness[i], expected[i], 1E-12));
            }
        }
    }

    #[test]
    fn test_clique_forward_bridge_cycle() {
        for p in [10, 50, 100] {
            for k in [10, 50, 100] {
                let mut arcs = Vec::new();
                let mut graph = new_clique(k);
                for i in 0..p {
                    arcs.push((k + i, k + (i + 1) % p));
                }
                arcs.push((k - 1, k));
                graph.add_arcs(arcs);

                let mut centrality = BetweennessCentrality::new(&graph, 0);
                centrality.compute(no_logging!());

                let mut expected = vec![0f64; k + p];
                (0..k - 1).for_each(|i| expected[i] = 0.);
                expected[k - 1] = (p * (k - 1)) as f64;
                (0..p).for_each(|d| {
                    expected[k + d] = (k as i32 * (p as i32 - d as i32 - 1)) as f64
                        + ((p as i32 - 1) * (p as i32 - 2)) as f64 / 2.0;
                });

                (0..k + p).for_each(|i| {
                    assert_approx_eq!(centrality.betweenness[i], expected[i], 1E-12);
                });
            }
        }
    }

    #[test]
    fn test_clique_back_bridge_cycle() {
        for p in [10, 50, 100] {
            for k in [10, 50, 100] {
                let mut arcs = Vec::new();
                let mut graph = new_clique(k);
                for i in 0..p {
                    arcs.push((k + i, k + (i + 1) % p));
                }
                arcs.push((k, k - 1));
                graph.add_arcs(arcs);

                let mut centrality = BetweennessCentrality::new(&graph, 0);
                centrality.compute(no_logging!());

                let mut expected = vec![0f64; k + p];
                (0..k - 1).for_each(|i| expected[i] = 0.);
                expected[k - 1] = (p * (k - 1)) as f64;
                (0..p).for_each(|d| {
                    let t = match d {
                        0 => p,
                        _ => 0,
                    };
                    expected[k + d] = (k as i32 * (d as i32 - 1 + t as i32)) as f64
                        + ((p as i32 - 1) * (p as i32 - 2)) as f64 / 2.0;
                });

                (0..k + p)
                    .for_each(|i| assert_approx_eq!(centrality.betweenness[i], expected[i], 1E-12));
            }
        }
    }

    #[test]
    fn test_clique_bi_bridge_cycle() {
        for p in [10, 50, 100] {
            for k in [10, 50, 100] {
                let mut arcs = Vec::new();
                let mut graph = new_clique(k);
                for i in 0..p {
                    arcs.push((k + i, k + (i + 1) % p));
                }
                arcs.push((k, k - 1));
                arcs.push((k - 1, k));
                graph.add_arcs(arcs);

                let mut centrality = BetweennessCentrality::new(&graph, 0);
                centrality.compute(no_logging!());

                let mut expected = vec![0f64; k + p];
                (0..k - 1).for_each(|i| expected[i] = 0.);
                expected[k - 1] = (2 * p * (k - 1)) as f64;
                expected[k] = (2 * k * (p - 1)) as f64 + ((p - 1) * (p - 2)) as f64 / 2.0;

                (1..p).for_each(|d| {
                    expected[k + d] =
                        (k * (p - 2)) as f64 + ((p as i32 - 1) * (p as i32 - 2)) as f64 / 2.0
                });

                (0..k + p)
                    .for_each(|i| assert_approx_eq!(centrality.betweenness[i], expected[i], 1E-12));
            }
        }
    }

    #[test]
    fn test_random() {
        for p in [0.1, 0.2, 0.5, 0.7] {
            for size in [10, 50, 100] {
                let graph = VecGraph::from_lender(ErdosRenyi::new(size, p, 0).iter());

                let mut centrality_multiple_visits = BetweennessCentrality::new(&graph, 0);
                centrality_multiple_visits.compute(no_logging!());

                let mut centrality = BetweennessCentrality::new(&graph, 0);
                centrality.compute(no_logging!());

                let size = graph.num_nodes();
                (0..size).for_each(|i| {
                    assert_approx_eq!(
                        centrality.betweenness[i],
                        centrality_multiple_visits.betweenness[i],
                        1E-12
                    )
                });
            }
        }
    }

    #[test]
    fn test_overflow_ok() {
        let blocks = 20;
        let block_size = 10;

        overflow_test(blocks, block_size);
    }

    #[test]
    #[should_panic] //TODO: maybe change with a different error handling
    fn test_overflow_not_ok() {
        let blocks = 40;
        let block_size = 10;
        overflow_test(blocks, block_size);
    }

    fn overflow_test(blocks: usize, block_size: usize) {
        let n = blocks * block_size;
        let mut arcs = Vec::new();
        let mut graph = VecGraph::new();

        let mut i = blocks;
        while i != 0 {
            i -= 1;
            let mut j = block_size - 1;
            while j != 0 {
                j -= 1;
                arcs.push((i * block_size, i * block_size + j + 1));
                arcs.push((i * block_size + j + 1, (i + 1) * block_size % n));
            }
        }
        graph.add_arcs(arcs);

        let mut centrality = BetweennessCentrality::new(&graph, 0);
        centrality.compute(no_logging!());
    }
}
