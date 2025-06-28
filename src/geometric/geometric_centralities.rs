use atomic_counter::AtomicCounter;
use common_traits::Number;
use dsi_progress_logger::ProgressLog;
use no_break::NoBreak;
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
pub struct DefaultGeometric {
    pub closeness: f64,
    pub harmonic: f64,
    pub lin: f64,
    pub exponential: f64,
    pub reachable: usize,
}

fn default_geometric_operation(alpha: f64) -> impl Fn(&mut DefaultGeometric, usize, usize) {
    move |anything: &mut DefaultGeometric, num_nodes: usize, distance: usize| {
        anything.reachable += num_nodes;
        if distance == 0 {
            return;
        }
        let hd = 1f64 / distance as f64;
        let ed = alpha.pow(distance as f64);
        anything.closeness += (distance * num_nodes) as f64;
        anything.harmonic += hd * num_nodes as f64;
        anything.exponential += ed * num_nodes as f64;
    }
}

fn default_geometric_post_op(res: &mut DefaultGeometric) {
    if res.closeness == 0f64 {
        res.lin = 1f64;
    } else {
        res.closeness = 1f64 / res.closeness;
        res.lin = res.reachable as f64 * res.reachable as f64 * res.closeness;
    }
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
) -> Box<[DefaultGeometric]> {
    let mut res = compute_custom(
        &graph,
        num_of_threads,
        pl,
        default_geometric_operation(DEFAULT_ALPHA),
    );

    for item in &mut res {
        default_geometric_post_op(item);
    }

    res
}

pub fn compute_single_node_par_visit_custom<
    T: Default + Clone + Sync,
    F: Fn(&mut T, usize, usize) + Sync,
>(
    graph: &(impl RandomAccessGraph + Sync),
    num_of_threads: usize,
    node: usize,
    pl: &mut impl ProgressLog,
    op: F,
) -> T {
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
    let res = single_visit_parallel_custom(node, &mut bfs, &thread_pool, op);

    pl.done();
    res
}

pub fn compute_single_node_par_visit(
    graph: &(impl RandomAccessGraph + Sync),
    num_of_threads: usize,
    node: usize,
    pl: &mut impl ProgressLog,
) -> DefaultGeometric {
    let mut res = compute_single_node_par_visit_custom(
        graph,
        num_of_threads,
        node,
        pl,
        default_geometric_operation(DEFAULT_ALPHA),
    );

    default_geometric_post_op(&mut res);
    res
}


pub fn compute_single_node_seq_visit_custom<
    T: Default + Clone + Sync,
    F: Fn(&mut T, usize, usize),
>(
    graph: &(impl RandomAccessGraph + Sync),
    node: usize,
    pl: &mut impl ProgressLog,
    op: F,
) -> T {
    let num_nodes = graph.num_nodes();

    pl.item_name("visit").expected_updates(Some(num_nodes));

    pl.start(format!(
        "Computing geometric centralities of node {node} with single-threaded sequential visit...",
    ));

    let mut bfs = Seq::new(graph);
    let res = single_visit_sequential_custom(node, &mut bfs, op);

    pl.done();
    res
}

pub fn compute_single_node_seq_visit(
    graph: &(impl RandomAccessGraph + Sync),
    node: usize,
    pl: &mut impl ProgressLog,
) -> DefaultGeometric {
    let mut res = compute_single_node_seq_visit_custom(
        graph,
        node,
        pl,
        default_geometric_operation(DEFAULT_ALPHA),
    );

    default_geometric_post_op(&mut res);
    res
}

pub fn compute_all_par_visit(
    graph: &(impl RandomAccessGraph + Sync),
    num_of_threads: usize,
    pl: &mut impl ProgressLog,
) -> Box<[DefaultGeometric]> {
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

    let mut bfs = ParFair::new(graph);
    let mut data = vec![DefaultGeometric::default(); num_nodes].into_boxed_slice();

    for node in 0..num_nodes {
        data[node] = single_visit_parallel_custom(
            node,
            &mut bfs,
            &thread_pool,
            default_geometric_operation(DEFAULT_ALPHA),
        );
        default_geometric_post_op(&mut data[node]);
        pl.update();
    }

    pl.done_with_count(num_nodes);

    data
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
            EventPred::FrontierSize { distance, size } => op(&mut anything, size, distance),
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
                    EventNoPred::FrontierSize {  distance, sizes } => unsafe {
                        op(&mut *anything.as_ptr(), sizes, distance)
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

#[cfg(test)]
mod tests {
    use crate::geometric::{geometric_centralities, DefaultGeometric};
    use crate::utils::{new_directed_cycle, transpose_arc_list};
    use assert_approx_eq::assert_approx_eq;
    use webgraph::prelude::VecGraph;
    use webgraph::traits::SequentialLabeling;

    fn standard_strategy(graph: &VecGraph) -> Box<[DefaultGeometric]> {
        geometric_centralities::compute(&graph, 0, dsi_progress_logger::no_logging!())
    }

    fn all_par_visit_strategy(graph: &VecGraph) -> Box<[DefaultGeometric]> {
        geometric_centralities::compute_all_par_visit(&graph, 0, dsi_progress_logger::no_logging!())
    }

    fn single_node_par_visit_strategy(graph: &VecGraph) -> Box<[DefaultGeometric]> {
        let num_nodes = graph.num_nodes();
        let mut res = vec![DefaultGeometric::default(); num_nodes].into_boxed_slice();
        for n in 0..num_nodes {
            res[n] = geometric_centralities::compute_single_node_par_visit(
                &graph,
                1,
                n,
                dsi_progress_logger::no_logging!(),
            );
        }
        res
    }

    fn single_node_seq_visit_strategy(graph: &VecGraph) -> Box<[DefaultGeometric]> {
        let num_nodes = graph.num_nodes();
        let mut res = vec![DefaultGeometric::default(); num_nodes].into_boxed_slice();
        for n in 0..num_nodes {
            res[n] = geometric_centralities::compute_single_node_seq_visit(
                &graph,
                n,
                dsi_progress_logger::no_logging!(),
            );
        }
        res
    }

    fn compute_generic(strategy: fn(&VecGraph) -> Box<[DefaultGeometric]>) {
        let graph = VecGraph::from_arcs(transpose_arc_list([(0, 1), (1, 2)]));
        let res = strategy(&graph);

        assert_eq!(res[0].closeness, 0f64);
        assert_eq!(res[1].closeness, 1f64);
        assert_eq!(res[2].closeness, 1f64 / 3f64);

        assert_eq!(res[0].lin, 1f64);
        assert_eq!(res[1].lin, 4f64);
        assert_eq!(res[2].lin, 3f64);

        assert_eq!(res[0].harmonic, 0f64);
        assert_eq!(res[1].harmonic, 1f64);
        assert_eq!(res[2].harmonic, 3f64 / 2f64);
    }

    fn compute_cycle_generic(strategy: fn(&VecGraph) -> Box<[DefaultGeometric]>) {
        for size in [10, 50, 100] {
            let graph = new_directed_cycle(size);
            let res = strategy(&graph);

            let mut expected = Vec::new();

            expected.resize(size, 2. / (size as f64 * (size as f64 - 1.)));
            (0..size).for_each(|i| assert_approx_eq!(res[i].closeness, expected[i], 1E-15f64));

            expected.fill(size as f64 * 2. / (size as f64 - 1.));
            (0..size).for_each(|i| assert_approx_eq!(res[i].lin, expected[i], 1E-15f64));

            let s = (1..size).fold(0f64, |acc, i| acc + 1. / (i as f64));
            expected.fill(s);
            (0..size).for_each(|i| assert_approx_eq!(res[i].harmonic, expected[i], 1E-14f64));
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

    #[test]
    fn test_compute_single_node_seq_visit() {
        compute_generic(single_node_seq_visit_strategy);
    }

    #[test]
    fn test_compute_cycle_single_node_seq_visit() {
        compute_cycle_generic(single_node_seq_visit_strategy);
    }
}
