use atomic_counter::AtomicCounter;
use common_traits::Number;
use dsi_progress_logger::ProgressLog;
use no_break::NoBreak;
use openmp_reducer::{Reducer, SharedReducer};
use rayon::ThreadPool;
use std::num::NonZero;
use std::ops::ControlFlow::Continue;
use std::thread;
use std::thread::available_parallelism;
use sync_cell_slice::SyncSlice;
use webgraph::traits::RandomAccessGraph;
use webgraph_algo::prelude::breadth_first::{EventNoPred, EventPred};
use webgraph_algo::visits::breadth_first::{ParFair, Seq};
use webgraph_algo::visits::{Parallel, Sequential};

const DEFAULT_ALPHA: f64 = 0.5;

#[derive(Clone, Debug)]
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

#[derive(Clone)]
struct ReducerCollection<'a> {
    closeness: SharedReducer<'a, usize, usize>,
    harmonic: SharedReducer<'a, f64, f64>,
    exponential: SharedReducer<'a, f64, f64>,
    reachable: SharedReducer<'a, usize, usize>,
}

pub fn compute(
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

    let mut cpl = pl.concurrent(); //TODO: pl.concurrent_with_threshold(n)
    cpl.item_name("visit").expected_updates(Some(num_nodes));

    cpl.start(format!(
        "Computing geometric centralities with {} threads...",
        &num_of_threads
    ));

    let atomic_counter = atomic_counter::RelaxedCounter::new(0);

    let mut closeness = vec![0f64; num_nodes].into_boxed_slice();
    let mut harmonic = vec![0f64; num_nodes].into_boxed_slice();
    let mut lin = vec![0f64; num_nodes].into_boxed_slice();
    let mut exponential = vec![0f64; num_nodes].into_boxed_slice();
    let mut reachable = vec![0usize; num_nodes].into_boxed_slice();

    let closeness_sync_cell = closeness.as_sync_slice();
    let harmonic_sync_cell = harmonic.as_sync_slice();
    let lin_sync_cell = lin.as_sync_slice();
    let exponential_sync_cell = exponential.as_sync_slice();
    let reachable_sync_cell = reachable.as_sync_slice();

    thread::scope(|scope| {
        for _ in 0..num_of_threads {
            scope.spawn(|| {
                let mut cpl = cpl.clone();
                let mut bfs = Seq::new(graph);

                let mut target_node = atomic_counter.inc();
                while target_node < num_nodes {
                    let centralities = single_visit_sequential(target_node, &mut bfs);
                    unsafe {
                        closeness_sync_cell[target_node].set(centralities.closeness);
                        harmonic_sync_cell[target_node].set(centralities.harmonic);
                        lin_sync_cell[target_node].set(centralities.lin);
                        exponential_sync_cell[target_node].set(centralities.exponential);
                        reachable_sync_cell[target_node].set(centralities.reachable);
                    }
                    target_node = atomic_counter.inc();
                    cpl.update();
                }
            });
        }
    });

    cpl.done_with_count(num_nodes);
    GeometricCentralitiesResult {
        closeness,
        harmonic,
        lin,
        exponential,
        reachable,
    }
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

fn single_visit_sequential(
    start: usize,
    bfs: &mut Seq<&(impl RandomAccessGraph + Sync)>,
) -> SingleNodeResult {
    let mut closeness = 0f64;
    let mut harmonic = 0f64;
    let lin;
    let mut exponential = 0f64;
    let mut reachable: usize = 0;

    bfs.reset();
    bfs.visit([start], |event| {
        let base = DEFAULT_ALPHA;
        match event {
            EventPred::Init { .. } => {}
            EventPred::Unknown {
                node: _node,
                pred: _pred,
                distance,
            } => {
                let d = distance;
                reachable += 1;
                if d == 0 {
                    //Skip first
                    return Continue(());
                }
                let hd = 1f64 / d as f64;
                let ed = base.pow(d as f64);
                closeness += d as f64;
                harmonic += hd;
                exponential += ed;
            }
            EventPred::Known { .. } => {}
            EventPred::Done { .. } => {}
        }
        Continue(())
    })
    .continue_value_no_break();

    if closeness == 0f64 {
        lin = 1f64;
    } else {
        closeness = 1f64 / closeness;
        lin = reachable as f64 * reachable as f64 * closeness;
    }
    SingleNodeResult {
        closeness,
        harmonic,
        lin,
        exponential,
        reachable,
    }
}

fn single_visit_parallel(
    start: usize,
    visit: &mut ParFair<&(impl RandomAccessGraph + Sync)>,
    thread_pool: &ThreadPool,
) -> SingleNodeResult {
    let base = DEFAULT_ALPHA;
    visit.reset();

    let usize_reducer_func = |global: &mut usize, local: &usize| *global += *local;
    let float_reducer_func = |global: &mut f64, local: &f64| *global += *local;

    let closeness_reducer = Reducer::<usize>::new(0, usize_reducer_func);
    let harmonic_reducer = Reducer::<f64>::new(0f64, float_reducer_func);
    let exponential_reducer = Reducer::<f64>::new(0f64, float_reducer_func);
    let reachable_reducer = Reducer::<usize>::new(0, usize_reducer_func);

    visit
        .par_visit_with(
            [start],
            ReducerCollection {
                closeness: closeness_reducer.share(),
                harmonic: harmonic_reducer.share(),
                exponential: exponential_reducer.share(),
                reachable: reachable_reducer.share(),
            },
            |cloned_reducer_collection, event| {
                match event {
                    EventNoPred::Unknown { distance, .. } => {
                        let d = distance;
                        *cloned_reducer_collection.reachable.as_mut() += 1;
                        if d == 0 {
                            //Skip first
                            return Continue(());
                        }
                        let hd = 1f64 / d as f64;
                        let ed = base.pow(d as f64);
                        *cloned_reducer_collection.closeness.as_mut() += d;
                        *cloned_reducer_collection.harmonic.as_mut() += hd;
                        *cloned_reducer_collection.exponential.as_mut() += ed;
                    }
                    _ => {}
                }
                Continue(())
            },
            thread_pool,
        )
        .continue_value_no_break();

    let mut closeness = closeness_reducer.get() as f64;
    let harmonic = harmonic_reducer.get();
    let lin;
    let exponential = exponential_reducer.get();
    let reachable = reachable_reducer.get();

    if closeness == 0f64 {
        lin = 1f64;
    } else {
        closeness = 1f64 / closeness;
        lin = reachable as f64 * reachable as f64 * closeness;
    }

    SingleNodeResult {
        closeness,
        harmonic,
        lin,
        exponential,
        reachable,
    }
}

#[cfg(test)]
mod tests {
    use crate::geometric::geometric_centralities;
    use crate::utils::{new_directed_cycle, transpose_arc_list};
    use assert_approx_eq::assert_approx_eq;
    use webgraph::prelude::VecGraph;

    #[test]
    fn test_compute() {
        let graph = VecGraph::from_arcs(transpose_arc_list([(0, 1), (1, 2)]));
        let res = geometric_centralities::compute(&graph, 0, dsi_progress_logger::no_logging!());

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

    #[test]
    fn test_compute_cycle() {
        for size in [10, 50, 100] {
            let graph = new_directed_cycle(size);
            let res =
                geometric_centralities::compute(&graph, 0, dsi_progress_logger::no_logging!());

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
    fn test_compute_all_par_visit() {
        let graph = VecGraph::from_arcs(transpose_arc_list([(0, 1), (1, 2)]));
        let res = geometric_centralities::compute_all_par_visit(
            &graph,
            0,
            dsi_progress_logger::no_logging!(),
        );

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

    #[test]
    fn test_compute_all_par_visit_cycle() {
        for size in [10, 50, 100] {
            let graph = new_directed_cycle(size);
            let res = geometric_centralities::compute_all_par_visit(
                &graph,
                0,
                dsi_progress_logger::no_logging!(),
            );

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
}
