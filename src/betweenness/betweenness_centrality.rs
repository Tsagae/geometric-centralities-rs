use atomic_counter::AtomicCounter;
use dsi_progress_logger::ProgressLog;
use std::sync::Mutex;
use std::thread::available_parallelism;
use webgraph::traits::RandomAccessGraph;

pub fn compute(
    graph: &(impl RandomAccessGraph + Sync),
    num_of_threads: usize,
    pl: &mut impl ProgressLog,
) -> Box<[f64]> {
    let num_nodes = graph.num_nodes();

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

    let mut cpl = pl.concurrent();

    cpl.item_name("visit").expected_updates(Some(num_nodes));

    cpl.start(format!(
        "Computing betweenness centrality with {} threads...",
        &thread_pool.current_num_threads()
    ));

    let atomic_counter = atomic_counter::RelaxedCounter::new(0);
    let betweenness = Mutex::new(vec![0.; num_nodes].into_boxed_slice());

    thread_pool.in_place_scope(|scope| {
        for _ in 0..thread_pool.current_num_threads() {
            scope.spawn(|s| {
                let mut cpl = cpl.clone();
                let mut distance = vec![-1i32; num_nodes].into_boxed_slice();
                let mut delta = vec![0f64; num_nodes].into_boxed_slice();
                let mut sigma = vec![0i64; num_nodes].into_boxed_slice();
                let mut queue = Vec::new();

                loop {
                    let curr = atomic_counter.inc();
                    if curr >= num_nodes {
                        break;
                    }
                    if graph.outdegree(curr) == 0 {
                        cpl.update();
                        continue;
                    }
                    queue.clear();
                    distance.fill(-1);
                    sigma.fill(0);
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
                        for s in graph.successors(node) {
                            if distance[s] == -1 {
                                distance[s] = d + 1;
                                queue.push(s);
                                //TODO: maybe use a different error handling. Not debug assert?
                                debug_assert!(check_overflow(&sigma, node, curr_sigma, s));
                                overflow |= sigma[s] > i64::MAX - curr_sigma;
                                sigma[s] += curr_sigma;
                            } else if distance[s] == d + 1 {
                                debug_assert!(check_overflow(&sigma, node, curr_sigma, s));
                                overflow |= sigma[s] > i64::MAX - curr_sigma;
                                sigma[s] += curr_sigma;
                            }
                        }
                        i += 1;
                    }

                    if overflow {
                        panic!("Overflow error");
                        //return Err(BetweennessError::PathCountOverflow);
                    }

                    for &node in queue[1..].iter().rev() {
                        let d = distance[node];
                        let sigma_node = sigma[node] as f64;
                        delta[node] = 0.;
                        for s in graph.successors(node) {
                            if distance[s] == d + 1 {
                                delta[node] += (1. + delta[s]) * sigma_node / sigma[s] as f64;
                            }
                        }
                    }

                    {
                        let mut lock_betweenness = betweenness.lock().unwrap();
                        for &node in &queue[1..] {
                            lock_betweenness[node] += delta[node];
                        }
                    }
                    cpl.update();
                }
            });
        }
    });

    cpl.done_with_count(num_nodes);
    betweenness.into_inner().unwrap()
}

fn check_overflow(sigma: &[i64], node: usize, curr_sigma: i64, s: usize) -> bool {
    if sigma[s] > i64::MAX - curr_sigma {
        panic!("{} > {} ({node} -> {s})", sigma[s], i64::MAX - curr_sigma);
    }
    true
}

#[cfg(test)]
mod tests {
    use crate::betweenness::compute;
    use crate::utils::{new_clique, new_directed_cycle};
    use assert_approx_eq::assert_approx_eq;
    use dsi_progress_logger::no_logging;
    use webgraph::graphs::random::ErdosRenyi;
    use webgraph::prelude::VecGraph;
    use webgraph::traits::SequentialLabeling;

    #[test]
    fn test_path() {
        let g = VecGraph::from_arcs([(0, 1), (1, 2)]);
        let betweenness = compute(&g, 0, no_logging!());
        
        assert_approx_eq!(betweenness[0], 0., 1E-5);
        assert_approx_eq!(betweenness[1], 1., 1E-5);
        assert_approx_eq!(betweenness[2], 0., 1E-5);
    }

    #[test]
    fn test_lozenge() {
        let g = VecGraph::from_arcs([(0, 1), (0, 2), (1, 3), (2, 3)]);
        let betweenness = compute(&g, 0, no_logging!());

        assert_approx_eq!(betweenness[0], 0., 1E-5);
        assert_approx_eq!(betweenness[1], 0.5, 1E-5);
        assert_approx_eq!(betweenness[2], 0.5, 1E-5);
        assert_approx_eq!(betweenness[3], 0., 1E-5);
    }

    #[test]
    fn test_cycle() {
        for size in [10, 50, 100] {
            let graph = new_directed_cycle(size);
            let betweenness = compute(&graph, 0, no_logging!());

            let mut expected = Vec::new();
            expected.resize(size, ((size - 1) * (size - 2)) as f64 / 2.0);

            (0..size).for_each(|i| assert_approx_eq!(betweenness[i], expected[i], 1E-12));
        }
    }

    #[test]
    fn test_clique() {
        for size in [10, 50, 100] {
            let graph = new_clique(size);
            let betweenness = compute(&graph, 0, no_logging!());

            let expected = vec![0f64; size];

            (0..size).for_each(|i| assert_approx_eq!(betweenness[i], expected[i], 1E-12));
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

                let betweenness = compute(&graph, 0, no_logging!());

                let mut expected = vec![0f64; k + p];
                (0..k).for_each(|i| expected[i] = 0.);
                (k..k + p).for_each(|i| expected[i] = ((p - 1) * (p - 2)) as f64 / 2.0);

                (0..k + p).for_each(|i| assert_approx_eq!(betweenness[i], expected[i], 1E-12));
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

                let betweenness = compute(&graph, 0, no_logging!());

                let mut expected = vec![0f64; k + p];
                (0..k - 1).for_each(|i| expected[i] = 0.);
                expected[k - 1] = (p * (k - 1)) as f64;
                (0..p).for_each(|d| {
                    expected[k + d] = (k as i32 * (p as i32 - d as i32 - 1)) as f64
                        + ((p as i32 - 1) * (p as i32 - 2)) as f64 / 2.0;
                });

                (0..k + p).for_each(|i| {
                    assert_approx_eq!(betweenness[i], expected[i], 1E-12);
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

                let betweenness = compute(&graph, 0, no_logging!());

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

                (0..k + p).for_each(|i| assert_approx_eq!(betweenness[i], expected[i], 1E-12));
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

                let betweenness = compute(&graph, 0, no_logging!());

                let mut expected = vec![0f64; k + p];
                (0..k - 1).for_each(|i| expected[i] = 0.);
                expected[k - 1] = (2 * p * (k - 1)) as f64;
                expected[k] = (2 * k * (p - 1)) as f64 + ((p - 1) * (p - 2)) as f64 / 2.0;

                (1..p).for_each(|d| {
                    expected[k + d] =
                        (k * (p - 2)) as f64 + ((p as i32 - 1) * (p as i32 - 2)) as f64 / 2.0
                });

                (0..k + p).for_each(|i| assert_approx_eq!(betweenness[i], expected[i], 1E-12));
            }
        }
    }

    #[test]
    fn test_random() {
        for p in [0.1, 0.2, 0.5, 0.7] {
            for size in [10, 50, 100] {
                let graph = VecGraph::from_lender(ErdosRenyi::new(size, p, 0).iter());

                let betweenness_multiple_visits = compute(&graph, 0, no_logging!());

                let betweenness = compute(&graph, 0, no_logging!());

                let size = graph.num_nodes();
                (0..size).for_each(|i| {
                    assert_approx_eq!(betweenness[i], betweenness_multiple_visits[i], 1E-12)
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

        compute(&graph, 0, no_logging!());
    }
}
