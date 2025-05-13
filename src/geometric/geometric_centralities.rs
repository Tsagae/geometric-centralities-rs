use atomic_counter::AtomicCounter;
use common_traits::Number;
use dsi_progress_logger::{ConcurrentProgressLog, ProgressLog};
use no_break::NoBreak;
use openmp_reducer::{Reducer, SharedReducer};
use rayon::ThreadPool;
use std::ops::ControlFlow::Continue;
use std::thread::available_parallelism;
use sync_cell_slice::SyncSlice;
use webgraph::traits::RandomAccessGraph;
use webgraph_algo::prelude::breadth_first::{EventNoPred, EventPred};
use webgraph_algo::visits::breadth_first::{ParFair, Seq};
use webgraph_algo::visits::{Parallel, Sequential};

const DEFAULT_ALPHA: f64 = 0.5;

pub struct GeometricCentralities<'a, G: RandomAccessGraph> {
    pub closeness: Vec<f64>,
    pub harmonic: Vec<f64>,
    pub lin: Vec<f64>,
    pub exponential: Vec<f64>,
    pub reachable: Vec<usize>,
    graph: &'a G,
    thread_pool: ThreadPool,
    alpha: f64,
}

#[derive(Clone, Debug)]
pub struct GeometricCentralityResult {
    pub closeness: f64,
    pub harmonic: f64,
    pub lin: f64,
    pub exponential: f64,
    pub reachable: usize,
}

#[derive(Clone)]
struct ReducerCollection<'a> {
    closeness: SharedReducer<'a, usize, usize>,
    harmonic: SharedReducer<'a, f64, f64>,
    exponential: SharedReducer<'a, f64, f64>,
    reachable: SharedReducer<'a, usize, usize>,
}

impl<G: RandomAccessGraph + Sync> GeometricCentralities<'_, G> {
    pub fn new(graph: &G, num_of_threads: usize) -> GeometricCentralities<G> {
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
        GeometricCentralities {
            graph,
            thread_pool: thread_pool,
            closeness: Vec::new(),
            harmonic: Vec::new(),
            lin: Vec::new(),
            exponential: Vec::new(),
            reachable: Vec::new(),
            alpha: DEFAULT_ALPHA,
        }
    }

    pub fn compute(&mut self, pl: &mut impl ConcurrentProgressLog) {
        self.init(pl);
        let atomic_counter = atomic_counter::ConsistentCounter::new(0);
        
        let num_of_nodes = self.graph.num_nodes();
        let thread_pool = &self.thread_pool;

        let closeness = self.closeness.as_sync_slice();
        let harmonic = self.harmonic.as_sync_slice();
        let lin = self.lin.as_sync_slice();
        let exponential = self.exponential.as_sync_slice();
        let reachable = self.reachable.as_sync_slice();

        pl.display_memory(false); //TODO: check if this can be enabled https://docs.rs/dsi-progress-logger/latest/dsi_progress_logger/trait.ConcurrentProgressLog.html

        pl.start(format!(
            "Computing geometric centralities with {} threads...",
            &self.thread_pool.current_num_threads()
        ));

        thread_pool.in_place_scope(|scope| {
            for _ in 0..thread_pool.current_num_threads() {
                scope.spawn(|_| {   
                    let mut bfs = Seq::new(self.graph);
                    let num_of_nodes = num_of_nodes;
                    let mut pl = pl.clone();
                    
                    let mut target_node = atomic_counter.inc();
                    while target_node < num_of_nodes {
                        let centralities =
                            Self::single_visit_sequential(self.alpha, target_node, &mut bfs);
                        unsafe {
                            closeness[target_node].set(centralities.closeness);
                            harmonic[target_node].set(centralities.harmonic);
                            lin[target_node].set(centralities.lin);
                            exponential[target_node].set(centralities.exponential);
                            reachable[target_node].set(centralities.reachable);
                        }
                        target_node = atomic_counter.inc();
                        pl.update();
                    }
                });
            }
        });
        pl.done_with_count(num_of_nodes);
    }

    pub fn compute_single_node_par_visit(
        &mut self,
        start_node: usize,
        pl: &mut impl ProgressLog,
        granularity: usize,
    ) -> GeometricCentralityResult {
        let num_of_nodes = self.graph.num_nodes();

        pl.item_name("visit")
            .display_memory(true)
            .local_speed(true)
            .expected_updates(Some(num_of_nodes));
        pl.start(format!(
            "Computing geometric centralities only on node {} with {} threads...",
            start_node,
            self.thread_pool.current_num_threads()
        ));

        let mut bfs = ParFair::new(self.graph, granularity);
        let res = Self::single_visit_parallel(self.alpha, start_node, &mut bfs, &self.thread_pool);

        pl.done();
        res
    }

    pub fn compute_all_par_visit(&mut self, pl: &mut impl ProgressLog, granularity: usize) {
        self.init(pl);
        let num_of_nodes = self.graph.num_nodes();

        pl.start(format!(
            "Computing geometric centralities on all nodes with parallel bfs with {} threads...",
            self.thread_pool.current_num_threads()
        ));

        let mut bfs = ParFair::new(self.graph, granularity);

        for node in 0..num_of_nodes {
            let result = Self::single_visit_parallel(self.alpha, node, &mut bfs, &self.thread_pool);
            self.closeness[node] = result.closeness;
            self.harmonic[node] = result.harmonic;
            self.lin[node] = result.lin;
            self.exponential[node] = result.exponential;
            self.reachable[node] = result.reachable;
            pl.update();
        }

        pl.done_with_count(num_of_nodes)
    }

    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    fn init(&mut self, pl: &mut impl ProgressLog) {
        let n = self.graph.num_nodes();
        self.closeness = vec![-1f64; n];
        self.harmonic = vec![-1f64; n];
        self.lin = vec![-1f64; n];
        self.exponential = vec![-1f64; n];
        self.reachable = vec![0; n];

        pl.item_name("visit")
            .display_memory(true)
            .local_speed(true)
            .expected_updates(Some(self.graph.num_nodes()));
    }

    fn single_visit_sequential(
        alpha: f64,
        start: usize,
        bfs: &mut Seq<&G>,
    ) -> GeometricCentralityResult {
        let mut closeness = 0f64;
        let mut harmonic = 0f64;
        let lin;
        let mut exponential = 0f64;
        let mut reachable: usize = 0;

        bfs.reset();
        bfs.visit([start], |event| {
            let base = alpha;
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
        GeometricCentralityResult {
            closeness,
            harmonic,
            lin,
            exponential,
            reachable,
        }
    }

    fn single_visit_parallel(
        alpha: f64,
        start: usize,
        visit: &mut ParFair<&G>,
        thread_pool: &ThreadPool,
    ) -> GeometricCentralityResult {
        let base = alpha;
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

        GeometricCentralityResult {
            closeness,
            harmonic,
            lin,
            exponential,
            reachable,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::geometric::GeometricCentralities;
    use crate::utils::{new_directed_cycle, transpose_arc_list};
    use assert_approx_eq::assert_approx_eq;
    use webgraph::prelude::VecGraph;

    #[test]
    fn test_compute() {
        let g = VecGraph::from_arcs(transpose_arc_list([(0, 1), (1, 2)]));
        let mut centralities = GeometricCentralities::new(&g, 0);
        centralities.compute(dsi_progress_logger::no_logging!());

        assert_eq!(centralities.closeness[0], 0f64);
        assert_eq!(centralities.closeness[1], 1f64);
        assert_eq!(centralities.closeness[2], 1f64 / 3f64);

        assert_eq!(centralities.lin[0], 1f64);
        assert_eq!(centralities.lin[1], 4f64);
        assert_eq!(centralities.lin[2], 3f64);

        assert_eq!(centralities.harmonic[0], 0f64);
        assert_eq!(centralities.harmonic[1], 1f64);
        assert_eq!(centralities.harmonic[2], 3f64 / 2f64);
    }

    #[test]
    fn test_compute_cycle() {
        for size in [10, 50, 100] {
            let graph = new_directed_cycle(size);
            let mut centralities = GeometricCentralities::new(&graph, 0);
            centralities.compute(dsi_progress_logger::no_logging!());

            let mut expected = Vec::new();

            expected.resize(size, 2. / (size as f64 * (size as f64 - 1.)));
            (0..size)
                .for_each(|i| assert_approx_eq!(centralities.closeness[i], expected[i], 1E-15f64));

            expected.fill(size as f64 * 2. / (size as f64 - 1.));
            (0..size).for_each(|i| assert_approx_eq!(centralities.lin[i], expected[i], 1E-15f64));

            let s = (1..size).fold(0f64, |acc, i| acc + 1. / (i as f64));
            expected.fill(s);
            (0..size)
                .for_each(|i| assert_approx_eq!(centralities.harmonic[i], expected[i], 1E-14f64));
        }
    }

    #[test]
    fn test_compute_all_par_visit() {
        let graph = VecGraph::from_arcs(transpose_arc_list([(0, 1), (1, 2)]));
        let mut centralities = GeometricCentralities::new(&graph, 0);
        centralities.compute_all_par_visit(&mut dsi_progress_logger::ProgressLogger::default(), 1);

        assert_eq!(centralities.closeness[0], 0f64);
        assert_eq!(centralities.closeness[1], 1f64);
        assert_eq!(centralities.closeness[2], 1f64 / 3f64);

        assert_eq!(centralities.lin[0], 1f64);
        assert_eq!(centralities.lin[1], 4f64);
        assert_eq!(centralities.lin[2], 3f64);

        assert_eq!(centralities.harmonic[0], 0f64);
        assert_eq!(centralities.harmonic[1], 1f64);
        assert_eq!(centralities.harmonic[2], 3f64 / 2f64);
    }

    #[test]
    fn test_compute_all_par_visit_cycle() {
        for size in [10, 50, 100] {
            let graph = new_directed_cycle(size);
            let mut centralities = GeometricCentralities::new(&graph, 0);
            centralities
                .compute_all_par_visit(&mut dsi_progress_logger::ProgressLogger::default(), 1);

            let mut expected = Vec::new();

            expected.resize(size, 2. / (size as f64 * (size as f64 - 1.)));
            (0..size)
                .for_each(|i| assert_approx_eq!(centralities.closeness[i], expected[i], 1E-15f64));

            expected.fill(size as f64 * 2. / (size as f64 - 1.));
            (0..size).for_each(|i| assert_approx_eq!(centralities.lin[i], expected[i], 1E-15f64));

            let s = (1..size).fold(0f64, |acc, i| acc + 1. / (i as f64));
            expected.fill(s);
            (0..size)
                .for_each(|i| assert_approx_eq!(centralities.harmonic[i], expected[i], 1E-14f64));
        }
    }
}
