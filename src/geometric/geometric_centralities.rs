use atomic_counter::AtomicCounter;
use common_traits::Number;
use dsi_progress_logger::ProgressLog;
use no_break::NoBreak;
use openmp_reducer::SharedReducer;
use rayon::ThreadPool;
use std::num::NonZero;
use std::ops::ControlFlow::Continue;
use std::thread;
use std::thread::available_parallelism;
use sync_cell_slice::{SyncCell, SyncSlice};
use webgraph::traits::RandomAccessGraph;
use webgraph_algo::prelude::breadth_first::{EventNoPred, EventPred};
use webgraph_algo::visits::breadth_first::{ParFair, Seq};
use webgraph_algo::visits::{Parallel, Sequential};

const DEFAULT_ALPHA: f64 = 0.5;

#[derive(Clone, Debug, Default)]
pub struct SingleNodeResult {
    pub closeness: f64,
    pub harmonic: f64,
    pub lin: f64,
    pub exponential: f64,
    pub reachable: usize,
}

#[derive(Debug)]
pub struct GeometricCentralitiesResult {
    pub closeness: Box<[f64]>,
    pub harmonic: Box<[f64]>,
    pub lin: Box<[f64]>,
    pub exponential: Box<[f64]>,
    pub reachable: Box<[usize]>,
}

pub fn compute_custom<T: Default + Clone + Sync, F: Fn(&mut T, usize, usize) + Sync>(
    graph: &(impl RandomAccessGraph + Sync),
    num_of_threads: usize,
    pl: &mut impl ProgressLog,
    op: F,
) -> Box<[T]> {
    let num_nodes = graph.num_nodes();

    let num_of_threads = if num_of_threads == 0 {
        available_parallelism().unwrap()
    } else {
        NonZero::new(num_of_threads).unwrap()
    };
    let num_of_threads = num_of_threads.get();

    let mut cpl = pl.concurrent(); //TODO: pl.concurrent_with_threshold(n)
    cpl.item_name("visit").expected_updates(Some(num_nodes));

    cpl.start(format!(
        "Computing geometric centralities (custom) with {} threads...",
        &num_of_threads
    ));

    let atomic_counter = atomic_counter::RelaxedCounter::new(0);

    let mut data = vec![T::default(); num_nodes].into_boxed_slice();
    let data_sync_cell = data.as_sync_slice();

    thread::scope(|scope| {
        for _ in 0..num_of_threads {
            scope.spawn(|| {
                let mut cpl = cpl.clone();
                let mut bfs = Seq::new(graph);

                let mut target_node = atomic_counter.inc();
                while target_node < num_nodes {
                    unsafe {
                        data_sync_cell[target_node].set(single_visit_sequential_custom(
                            target_node,
                            &mut bfs,
                            &op,
                        ));
                    }
                    target_node = atomic_counter.inc();
                    cpl.update();
                }
            });
        }
    });

    cpl.done_with_count(num_nodes);
    data
}

pub fn compute(
    graph: &(impl RandomAccessGraph + Sync),
    num_of_threads: usize,
    pl: &mut impl ProgressLog,
) -> Box<[SingleNodeResult]> {
    let alpha: f64 = DEFAULT_ALPHA;
    let mut res = compute_custom(
        &graph,
        num_of_threads,
        pl,
        |anything: &mut SingleNodeResult, num_nodes, distance| {
            anything.reachable += num_nodes;
            if distance == 0 {
                return;
            }
            let hd = 1f64 / distance as f64;
            let ed = alpha.pow(distance as f64);
            anything.closeness += (distance * num_nodes) as f64;
            anything.harmonic += hd * num_nodes as f64;
            anything.exponential += ed * num_nodes as f64;
        },
    );

    for item in &mut res {
        if item.closeness == 0f64 {
            item.lin = 1f64;
        } else {
            item.closeness = 1f64 / item.closeness;
            item.lin = item.reachable as f64 * item.reachable as f64 * item.closeness;
        }
    }

    res
}

pub fn compute_single_node_par_visit(
    graph: &(impl RandomAccessGraph + Sync),
    num_of_threads: usize,
    node: usize,
    pl: &mut impl ProgressLog,
) -> SingleNodeResult {
    let num_nodes = graph.num_nodes();

    let num_of_threads = if num_of_threads == 0 {
        available_parallelism().unwrap()
    } else {
        NonZero::new(num_of_threads).unwrap()
    };
    let num_of_threads = num_of_threads.get();

    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_of_threads)
        .build()
        .expect("Error in building thread pool");

    pl.item_name("visit").expected_updates(Some(num_nodes));

    pl.start(format!(
        "Computing geometric centralities of node {node} with {} threads...",
        &thread_pool.current_num_threads()
    ));

    let mut bfs = ParFair::new(graph);
    let res = single_visit_parallel(node, &mut bfs, &thread_pool);

    pl.done();
    res
}

pub fn compute_all_par_visit(
    graph: &(impl RandomAccessGraph + Sync),
    num_of_threads: usize,
    pl: &mut impl ProgressLog,
) -> GeometricCentralitiesResult {
    let num_nodes = graph.num_nodes();

    let num_of_threads = if num_of_threads == 0 {
        available_parallelism().unwrap()
    } else {
        NonZero::new(num_of_threads).unwrap()
    };
    let num_of_threads = num_of_threads.get();

    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_of_threads)
        .build()
        .expect("Error in building thread pool");

    pl.item_name("visit").expected_updates(Some(num_nodes));

    pl.start(format!(
        "Computing geometric centralities (parallel bfs) with {} threads...",
        &thread_pool.current_num_threads()
    ));

    let mut closeness = vec![0f64; num_nodes].into_boxed_slice();
    let mut harmonic = vec![0f64; num_nodes].into_boxed_slice();
    let mut lin = vec![0f64; num_nodes].into_boxed_slice();
    let mut exponential = vec![0f64; num_nodes].into_boxed_slice();
    let mut reachable = vec![0usize; num_nodes].into_boxed_slice();

    let mut bfs = ParFair::new(graph);

    for node in 0..num_nodes {
        let result = single_visit_parallel(node, &mut bfs, &thread_pool);
        closeness[node] = result.closeness;
        harmonic[node] = result.harmonic;
        lin[node] = result.lin;
        exponential[node] = result.exponential;
        reachable[node] = result.reachable;
        pl.update();
    }

    pl.done_with_count(num_nodes);

    GeometricCentralitiesResult {
        closeness,
        harmonic,
        lin,
        exponential,
        reachable,
    }
}

// TODO: add example to doc
/// Performs a sequential breadth-first visit from a start node and applies a custom operation
/// to compute metrics based on the distances of visited nodes.
///
/// # Arguments
///
/// * `start`: The source node to start the visit from
/// * `bfs`: The sequential bfs
/// * `op`: Callback called at every change of distance with a reference to the mutable item, the number of nodes and the distance
///
/// returns: A custom type T containing the computed metrics for the start node
fn single_visit_sequential_custom<T: Default + Clone>(
    start: usize,
    bfs: &mut Seq<&(impl RandomAccessGraph + Sync)>,
    op: impl Fn(&mut T, usize, usize),
) -> T {
    let mut anything = T::default();
    bfs.reset();
    bfs.visit([start], |event| {
        match event {
            EventPred::DistanceChanged { nodes, distance } => op(&mut anything, nodes, distance),
            _ => {}
        }
        Continue(())
    })
    .continue_value_no_break();
    anything
}

// TODO: add example to doc
/// Performs a parallel breadth-first visit from a start node and applies a custom operation
/// to compute metrics based on the distances of visited nodes.
///
/// # Arguments
///
/// * `start`: The source node to start the visit from
/// * `bfs`: The sequential bfs
/// * `op`: Callback called at every change of distance with a reference to the mutable item, the number of nodes and the distance
///
/// returns: A custom type T containing the computed metrics for the start node
fn single_visit_parallel_custom<T: Default + Clone + Sync>(
    start: usize,
    visit: &mut ParFair<&(impl RandomAccessGraph + Sync)>,
    thread_pool: &ThreadPool,
    op: impl Fn(&mut T, usize, usize) + Sync,
) -> T {
    let anything = SyncCell::new(T::default());
    visit.reset();

    visit
        .par_visit(
            [start],
            |event| {
                match event {
                    EventNoPred::DistanceChanged { nodes, distance } => unsafe {
                        op(&mut *anything.as_ptr(), nodes, distance)
                    },
                    _ => {}
                }
                Continue(())
            },
            thread_pool,
        )
        .continue_value_no_break();
    anything.into_inner()
}

fn single_visit_parallel(
    start: usize,
    visit: &mut ParFair<&(impl RandomAccessGraph + Sync)>,
    thread_pool: &ThreadPool,
) -> SingleNodeResult {
    let alpha: f64 = DEFAULT_ALPHA;
    let mut res = single_visit_parallel_custom(
        start,
        visit,
        thread_pool,
        |anything: &mut SingleNodeResult, num_nodes, distance| {
            anything.reachable += num_nodes;
            if distance == 0 {
                return;
            }
            let hd = 1f64 / distance as f64;
            let ed = alpha.pow(distance as f64);
            anything.closeness += (distance * num_nodes) as f64;
            anything.harmonic += hd * num_nodes as f64;
            anything.exponential += ed * num_nodes as f64;
        },
    );

    if res.closeness == 0f64 {
        res.lin = 1f64;
    } else {
        res.closeness = 1f64 / res.closeness;
        res.lin = res.reachable as f64 * res.reachable as f64 * res.closeness;
    }

    res
}

#[cfg(test)]
mod tests {
    use crate::geometric::{geometric_centralities, GeometricCentralitiesResult};
    use crate::utils::{new_directed_cycle, transpose_arc_list};
    use assert_approx_eq::assert_approx_eq;
    use webgraph::prelude::VecGraph;
    use webgraph::traits::SequentialLabeling;

    fn standard_strategy(graph: &VecGraph) -> GeometricCentralitiesResult {
        let res = geometric_centralities::compute(&graph, 0, dsi_progress_logger::no_logging!());
        GeometricCentralitiesResult {
            closeness: res.iter().map(|item| item.closeness).collect(),
            harmonic: res.iter().map(|item| item.harmonic).collect(),
            lin: res.iter().map(|item| item.lin).collect(),
            exponential: res.iter().map(|item| item.exponential).collect(),
            reachable: res.iter().map(|item| item.reachable).collect(),
        }
    }

    fn all_par_visit_strategy(graph: &VecGraph) -> GeometricCentralitiesResult {
        geometric_centralities::compute_all_par_visit(&graph, 0, dsi_progress_logger::no_logging!())
    }

    fn single_node_par_visit_strategy(graph: &VecGraph) -> GeometricCentralitiesResult {
        let num_nodes = graph.num_nodes();
        let mut res = GeometricCentralitiesResult {
            closeness: vec![0f64; num_nodes].into_boxed_slice(),
            harmonic: vec![0f64; num_nodes].into_boxed_slice(),
            lin: vec![0f64; num_nodes].into_boxed_slice(),
            exponential: Box::new([]),
            reachable: Box::new([]),
        };
        for n in 0..num_nodes {
            let single_node_res = geometric_centralities::compute_single_node_par_visit(
                &graph,
                0,
                n,
                dsi_progress_logger::no_logging!(),
            );
            res.closeness[n] = single_node_res.closeness;
            res.harmonic[n] = single_node_res.harmonic;
            res.lin[n] = single_node_res.lin;
        }
        res
    }

    fn compute_generic(strategy: fn(&VecGraph) -> GeometricCentralitiesResult) {
        let graph = VecGraph::from_arcs(transpose_arc_list([(0, 1), (1, 2)]));
        let res = strategy(&graph);

        assert_eq!(res.closeness[0], 0f64);
        assert_eq!(res.closeness[1], 1f64);
        assert_eq!(res.closeness[2], 1f64 / 3f64);

        assert_eq!(res.lin[0], 1f64);
        assert_eq!(res.lin[1], 4f64);
        assert_eq!(res.lin[2], 3f64);

        assert_eq!(res.harmonic[0], 0f64);
        assert_eq!(res.harmonic[1], 1f64);
        assert_eq!(res.harmonic[2], 3f64 / 2f64);
    }

    fn compute_cycle_generic(strategy: fn(&VecGraph) -> GeometricCentralitiesResult) {
        for size in [10, 50, 100] {
            let graph = new_directed_cycle(size);
            let res = strategy(&graph);

            let mut expected = Vec::new();

            expected.resize(size, 2. / (size as f64 * (size as f64 - 1.)));
            (0..size).for_each(|i| assert_approx_eq!(res.closeness[i], expected[i], 1E-15f64));

            expected.fill(size as f64 * 2. / (size as f64 - 1.));
            (0..size).for_each(|i| assert_approx_eq!(res.lin[i], expected[i], 1E-15f64));

            let s = (1..size).fold(0f64, |acc, i| acc + 1. / (i as f64));
            expected.fill(s);
            (0..size).for_each(|i| assert_approx_eq!(res.harmonic[i], expected[i], 1E-14f64));
        }
    }

    #[test]
    fn test_compute_standard() {
        compute_generic(standard_strategy);
    }

    #[test]
    fn test_compute_cycle_standard() {
        compute_cycle_generic(standard_strategy);
    }

    #[test]
    fn test_compute_all_par_visit() {
        compute_generic(all_par_visit_strategy);
    }

    #[test]
    fn test_compute_cycle_all_par_visit() {
        compute_cycle_generic(all_par_visit_strategy);
    }

    #[test]
    fn test_compute_single_node_par_visit() {
        compute_generic(single_node_par_visit_strategy);
    }

    #[test]
    fn test_compute_cycle_single_node_par_visit() {
        compute_cycle_generic(single_node_par_visit_strategy);
    }
}
