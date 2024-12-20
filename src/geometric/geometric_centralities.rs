use atomic_counter::AtomicCounter;
use atomic_float::AtomicF64;
use common_traits::Number;
use dsi_progress_logger::{no_logging, ProgressLog};
use rayon::ThreadPool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::sync::{Arc, Mutex};
use std::thread::available_parallelism;
use sync_cell_slice::SyncSlice;
use webgraph::traits::RandomAccessGraph;
use webgraph_algo::prelude::breadth_first::{EventPred, Seq};
use webgraph_algo::traits::{Parallel, Sequential};

const DEFAULT_ALPHA: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct GeometricCentralityResult {
    pub closeness: f64,
    pub harmonic: f64,
    pub lin: f64,
    pub exponential: f64,
    pub reachable: usize,
}

pub struct GeometricCentralities<'a, G: RandomAccessGraph> {
    graph: &'a G,
    num_of_threads: usize,
    atomic_counter: Arc<atomic_counter::ConsistentCounter>,
    thread_pool: ThreadPool,
    pub closeness: Vec<f64>,
    pub harmonic: Vec<f64>,
    pub lin: Vec<f64>,
    pub exponential: Vec<f64>,
    pub reachable: Vec<usize>,
    alpha: f64,
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
            num_of_threads: num_threads,
            atomic_counter: Arc::new(atomic_counter::ConsistentCounter::new(0)),
            thread_pool: thread_pool,
            closeness: Vec::new(),
            harmonic: Vec::new(),
            lin: Vec::new(),
            exponential: Vec::new(),
            reachable: Vec::new(),
            alpha: DEFAULT_ALPHA,
        }
    }

    fn init<'a, P: ProgressLog + Send + Sync>(&mut self, pl: &'a mut P) -> Arc<Mutex<&'a mut P>> {
        let n = self.graph.num_nodes();
        self.closeness = vec![-1f64; n];
        self.harmonic = vec![-1f64; n];
        self.lin = vec![-1f64; n];
        self.exponential = vec![-1f64; n];
        self.reachable = vec![0; n];
        self.atomic_counter.reset();

        pl.display_memory(true)
            .item_name("visit")
            .local_speed(true)
            .expected_updates(Some(n));

        Arc::new(Mutex::new(pl))
    }

    pub fn compute<P: ProgressLog + Send + Sync>(&mut self, pl: &mut P) {
        let num_of_nodes = self.graph.num_nodes();
        let shared_pl = self.init::<P>(pl);
        let thread_pool = &self.thread_pool;

        let closeness = self.closeness.as_sync_slice();
        let harmonic = self.harmonic.as_sync_slice();
        let lin = self.lin.as_sync_slice();
        let exponential = self.exponential.as_sync_slice();
        let reachable = self.reachable.as_sync_slice();

        {
            let mut pl = shared_pl.lock().expect("Error in taking mut pl");
            pl.start(format!(
                "Computing geometric centralities with {} threads...",
                &self.thread_pool.current_num_threads()
            ));
        }

        thread_pool.in_place_scope(|scope| {
            for _ in 0..thread_pool.current_num_threads() {
                let thread_atomic_counter = Arc::clone(&self.atomic_counter);
                let num_of_nodes = self.graph.num_nodes();
                let graph = self.graph;
                let local_pl = Arc::clone(&shared_pl);
                let alpha = self.alpha;

                scope.spawn(move |_| {
                    let mut bfs = Seq::new(graph);
                    let atom_counter = thread_atomic_counter;
                    let num_of_nodes = num_of_nodes;

                    let mut target_node = atom_counter.inc();
                    while target_node < num_of_nodes {
                        let centralities =
                            Self::single_visit_sequential(alpha, target_node, &mut bfs);
                        unsafe {
                            closeness[target_node].set(centralities.closeness);
                            harmonic[target_node].set(centralities.harmonic);
                            lin[target_node].set(centralities.lin);
                            exponential[target_node].set(centralities.exponential);
                            reachable[target_node].set(centralities.reachable);
                        }
                        target_node = atom_counter.inc();
                        {
                            let mut pl = local_pl.lock().expect("Error in taking mut pl");
                            pl.update();
                        }
                    }
                });
            }
        });
        {
            let mut pl = shared_pl.lock().expect("Error in taking mut pl");
            pl.done_with_count(num_of_nodes);
        }
    }

    pub fn compute_single_node<P: ProgressLog + Send + Sync>(
        &mut self,
        start_node: usize,
        pl: &mut P,
        granularity: usize,
    ) -> GeometricCentralityResult {
        let num_of_nodes = self.graph.num_nodes();

        pl.start(format!(
            "Computing geometric centralities only on node {} with {} threads...",
            start_node,
            self.thread_pool.current_num_threads()
        ));

        pl.display_memory(true)
            .item_name("node")
            .local_speed(true)
            .expected_updates(Some(num_of_nodes));

        let mut bfs =
            webgraph_algo::algo::visits::breadth_first::ParFairBase::new(self.graph, granularity);
        Self::single_visit_parallel(self.alpha, start_node, &mut bfs, &self.thread_pool, pl)
    }

    pub fn compute_all_single_node<P: ProgressLog + Send + Sync>(
        &mut self,
        pl: &mut P,
        granularity: usize,
    ) -> Vec<GeometricCentralityResult> {
        let num_of_nodes = self.graph.num_nodes();

        pl.start(format!(
            "Computing geometric centralities on all nodes with parallel bfs with {} threads...",
            self.thread_pool.current_num_threads()
        ));

        pl.display_memory(true)
            .item_name("visit")
            .local_speed(true)
            .expected_updates(Some(num_of_nodes));

        let mut bfs =
            webgraph_algo::algo::visits::breadth_first::ParFairBase::new(self.graph, granularity);
        let mut results = Vec::with_capacity(self.num_of_threads);
        for node in 0..num_of_nodes {
            let result = Self::single_visit_parallel(
                self.alpha,
                node,
                &mut bfs,
                &self.thread_pool,
                no_logging!(),
            );
            results.push(result);
            pl.update();
        }
        pl.done_with_count(num_of_nodes);
        results
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

        webgraph_algo::traits::Sequential::reset(bfs);
        bfs.visit(
            start,
            |args| {
                let base = alpha;
                match args {
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
                    _ => Ok::<(), ()>(()),
                }
            },
            no_logging!(),
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

    fn single_visit_parallel(
        alpha: f64,
        start: usize,
        visit: &mut impl Parallel<EventPred>,
        thread_pool: &ThreadPool,
        pl: &mut impl ProgressLog,
    ) -> GeometricCentralityResult {
        let atomic_closeness = AtomicUsize::new(0);
        let atomic_harmonic = AtomicF64::new(0f64);
        let atomic_exponential = AtomicF64::new(0f64);
        let atomic_reachable = AtomicUsize::new(0);

        let base = alpha;
        visit.reset();

        visit
            .par_visit(
                start,
                |args| match args {
                    EventPred::Unknown { distance, .. } => {
                        let d = distance;
                        atomic_reachable.fetch_add(1, Relaxed);
                        if d == 0 {
                            //Skip first
                            return Ok(());
                        }
                        let hd = 1f64 / d as f64;
                        let ed = base.pow(d as f64);
                        atomic_closeness.fetch_add(d, Relaxed);
                        atomic_harmonic.fetch_add(hd, Relaxed);
                        atomic_exponential.fetch_add(ed, Relaxed);
                        Ok(())
                    }
                    _ => Ok::<(), ()>(()),
                },
                thread_pool,
                pl,
            )
            .expect("Error in bfs");

        let mut closeness = atomic_closeness.load(SeqCst) as f64;
        let harmonic = atomic_harmonic.load(SeqCst);
        let lin;
        let exponential = atomic_exponential.load(SeqCst);
        let reachable = atomic_reachable.load(SeqCst);

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

    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }
}

#[cfg(test)]
mod tests {
    use crate::geometric::GeometricCentralities;
    use crate::utils::{new_directed_cycle, transpose_arc_list};
    use assert_approx_eq::assert_approx_eq;
    use webgraph::labels::Left;
    use webgraph::prelude::VecGraph;
    use webgraph::traits::SequentialLabeling;

    #[test]
    fn test_compute_generic() {
        let g = VecGraph::from_arc_list(transpose_arc_list([(0, 1), (1, 2)]));
        let l = &Left(g);
        let mut centralities = GeometricCentralities::new(&l, 0);
        centralities.compute(dsi_progress_logger::no_logging!());

        assert_eq!(0f64, centralities.closeness[0]);
        assert_eq!(1f64, centralities.closeness[1]);
        assert_eq!(1f64 / 3f64, centralities.closeness[2]);

        assert_eq!(1f64, centralities.lin[0]);
        assert_eq!(4f64, centralities.lin[1]);
        assert_eq!(3f64, centralities.lin[2]);

        assert_eq!(0f64, centralities.harmonic[0]);
        assert_eq!(1f64, centralities.harmonic[1]);
        assert_eq!(3f64 / 2f64, centralities.harmonic[2]);
    }

    #[test]
    fn test_compute() {
        let g = VecGraph::from_arc_list(transpose_arc_list([(0, 1), (1, 2)]));
        let l = &Left(g);
        let mut centralities = GeometricCentralities::new(&l, 0);
        centralities.compute(dsi_progress_logger::no_logging!());

        assert_eq!(0f64, centralities.closeness[0]);
        assert_eq!(1f64, centralities.closeness[1]);
        assert_eq!(1f64 / 3f64, centralities.closeness[2]);

        assert_eq!(1f64, centralities.lin[0]);
        assert_eq!(4f64, centralities.lin[1]);
        assert_eq!(3f64, centralities.lin[2]);

        assert_eq!(0f64, centralities.harmonic[0]);
        assert_eq!(1f64, centralities.harmonic[1]);
        assert_eq!(3f64 / 2f64, centralities.harmonic[2]);
    }

    #[test]
    fn test_cycle() {
        for size in [10, 50, 100] {
            let graph = Left(new_directed_cycle(size));
            let mut centralities = GeometricCentralities::new(&graph, 0);
            centralities.compute(dsi_progress_logger::no_logging!());

            let mut expected = Vec::new();

            expected.resize(size, 2. / (size as f64 * (size as f64 - 1.)));
            (0..size)
                .for_each(|i| assert_approx_eq!(expected[i], centralities.closeness[i], 1E-15f64));

            expected.fill(size as f64 * 2. / (size as f64 - 1.));
            (0..size).for_each(|i| assert_approx_eq!(expected[i], centralities.lin[i], 1E-15f64));

            let s = (1..size).fold(0f64, |acc, i| acc + 1. / (i as f64));
            expected.fill(s);
            (0..size)
                .for_each(|i| assert_approx_eq!(expected[i], centralities.harmonic[i], 1E-14f64));
        }
    }

    #[test]
    fn test_compute_single_node() {
        let graph = Left(VecGraph::from_arc_list(transpose_arc_list([
            (0, 1),
            (1, 2),
        ])));
        let num_of_nodes = graph.num_nodes();

        let mut closeness = vec![-1f64; num_of_nodes];
        let mut harmonic = vec![-1f64; num_of_nodes];
        let mut lin = vec![-1f64; num_of_nodes];
        let mut exponential = vec![-1f64; num_of_nodes];
        let mut reachable = vec![0; num_of_nodes];

        let mut centralities = GeometricCentralities::new(&graph, 0);

        for node in 0..num_of_nodes {
            let results = centralities.compute_single_node(
                node,
                &mut dsi_progress_logger::ProgressLogger::default(),
                1,
            );
            closeness[node] = results.closeness;
            harmonic[node] = results.harmonic;
            lin[node] = results.lin;
            exponential[node] = results.exponential;
            reachable[node] = results.reachable;
        }

        assert_eq!(0f64, closeness[0]);
        assert_eq!(1f64, closeness[1]);
        assert_eq!(1f64 / 3f64, closeness[2]);

        assert_eq!(1f64, lin[0]);
        assert_eq!(4f64, lin[1]);
        assert_eq!(3f64, lin[2]);

        assert_eq!(0f64, harmonic[0]);
        assert_eq!(1f64, harmonic[1]);
        assert_eq!(3f64 / 2f64, harmonic[2]);
    }

    #[test]
    fn test_cycle_single_node() {
        for size in [10, 50, 100] {
            let graph = Left(new_directed_cycle(size));
            let mut centralities = GeometricCentralities::new(&graph, 0);

            let num_of_nodes = graph.num_nodes();
            let mut closeness = vec![-1f64; num_of_nodes];
            let mut harmonic = vec![-1f64; num_of_nodes];
            let mut lin = vec![-1f64; num_of_nodes];
            let mut exponential = vec![-1f64; num_of_nodes];
            let mut reachable = vec![0; num_of_nodes];

            for node in 0..graph.num_nodes() {
                let results =
                    centralities.compute_single_node(node, dsi_progress_logger::no_logging!(), 1);
                closeness[node] = results.closeness;
                harmonic[node] = results.harmonic;
                lin[node] = results.lin;
                exponential[node] = results.exponential;
                reachable[node] = results.reachable;
            }

            let mut expected = Vec::new();

            expected.resize(size, 2. / (size as f64 * (size as f64 - 1.)));
            (0..size).for_each(|i| assert_approx_eq!(expected[i], closeness[i], 1E-15f64));

            expected.fill(size as f64 * 2. / (size as f64 - 1.));
            (0..size).for_each(|i| assert_approx_eq!(expected[i], lin[i], 1E-15f64));

            let s = (1..size).fold(0f64, |acc, i| acc + 1. / (i as f64));
            expected.fill(s);
            (0..size).for_each(|i| assert_approx_eq!(expected[i], harmonic[i], 1E-14f64));
        }
    }
}
