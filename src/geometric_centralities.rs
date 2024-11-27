use atomic_counter::AtomicCounter;
use common_traits::Number;
use dsi_progress_logger::{no_logging, progress_logger, ProgressLog};
use rayon::ThreadPool;
use std::cmp::min;
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
    pub closeness: Vec<f64>,
    pub harmonic: Vec<f64>,
    pub lin: Vec<f64>,
    pub exponential: Vec<f64>,
    pub reachable: Vec<usize>,
    graph: &'a G,
    num_of_threads: usize,
    atomic_counter: Arc<atomic_counter::ConsistentCounter>,
    alpha: f64,
}

impl<G: RandomAccessGraph + Sync> GeometricCentralities<'_, G> {
    pub fn new(graph: &G, num_of_threads: usize) -> GeometricCentralities<G> {
        GeometricCentralities {
            graph,
            num_of_threads: min(
                graph.num_nodes(),
                if num_of_threads == 0 {
                    usize::from(available_parallelism().unwrap())
                } else {
                    num_of_threads
                },
            ),
            closeness: Vec::new(),
            harmonic: Vec::new(),
            lin: Vec::new(),
            exponential: Vec::new(),
            reachable: Vec::new(),
            atomic_counter: Arc::new(atomic_counter::ConsistentCounter::new(0)),
            alpha: DEFAULT_ALPHA,
        }
    }

    pub fn compute<P: ProgressLog + Send + Sync>(&mut self, pl: &mut P) {
        let num_of_nodes = self.graph.num_nodes();
        let (shared_pl, thread_pool, num_threads) = self.init::<P>(pl);

        let closeness = self.closeness.as_sync_slice();
        let harmonic = self.harmonic.as_sync_slice();
        let lin = self.lin.as_sync_slice();
        let exponential = self.exponential.as_sync_slice();
        let reachable = self.reachable.as_sync_slice();

        thread_pool.in_place_scope(|scope| {
            for _ in 0..num_threads {
                let thread_atomic_counter = Arc::clone(&self.atomic_counter);
                let num_of_nodes = self.graph.num_nodes();
                let graph = self.graph;
                let local_pl = Arc::clone(&shared_pl);
                let alpha = self.alpha;

                scope.spawn(move |_| {
                    let mut bfs = Seq::new(graph);
                    //println!("Started thread id: {:?}", thread::current().id());
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

        let num_threads = min(self.graph.num_nodes(), self.num_of_threads);
        pl.start(format!(
            "Computing geometric centralities only on node {start_node} with {num_threads} threads..."
        ));

        pl.display_memory(true)
            .item_name("node")
            .local_speed(true)
            .expected_updates(Some(num_of_nodes));

        let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
        thread_pool_builder = thread_pool_builder.num_threads(num_threads);
        let thread_pool = thread_pool_builder
            .build()
            .expect("Error in building thread pool");

        let mut bfs =
            webgraph_algo::algo::visits::breadth_first::ParFairBase::new(self.graph, granularity);
        Self::single_visit_parallel(self.alpha, start_node, &mut bfs, &thread_pool, pl)
    }

    pub fn compute_all_single_node<P: ProgressLog + Send + Sync>(
        &mut self,
        pl: &mut P,
        granularity: usize,
    ) -> Vec<GeometricCentralityResult> {
        let num_of_nodes = self.graph.num_nodes();

        let num_threads = min(self.graph.num_nodes(), self.num_of_threads);
        pl.start(format!(
            "Computing geometric centralities on all nodes with parallel bfs with {num_threads} threads..."
        ));

        pl.display_memory(true)
            .item_name("visit")
            .local_speed(true)
            .expected_updates(Some(num_of_nodes));

        let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
        thread_pool_builder = thread_pool_builder.num_threads(num_threads);
        let thread_pool = thread_pool_builder
            .build()
            .expect("Error in building thread pool");

        let mut bfs =
            webgraph_algo::algo::visits::breadth_first::ParFairBase::new(self.graph, granularity);
        let mut results = Vec::with_capacity(self.num_of_threads);
        for node in 0..num_of_nodes {
            let result = Self::single_visit_parallel(self.alpha, node, &mut bfs, &thread_pool, no_logging!());
            results.push(result);
            pl.update();
        }
        results
    }

    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    fn init<'a, P: ProgressLog + Send + Sync>(
        &mut self,
        pl: &'a mut P,
    ) -> (Arc<Mutex<&'a mut P>>, ThreadPool, usize) {
        let num_of_nodes = self.graph.num_nodes();
        self.closeness = vec![-1f64; num_of_nodes];
        self.harmonic = vec![-1f64; num_of_nodes];
        self.lin = vec![-1f64; num_of_nodes];
        self.exponential = vec![-1f64; num_of_nodes];
        self.reachable = vec![0; num_of_nodes];
        self.atomic_counter.reset();

        let num_threads = min(self.graph.num_nodes(), self.num_of_threads);
        pl.start(format!(
            "Computing geometric centralities with {num_threads} threads..."
        ));

        pl.display_memory(true)
            .item_name("visit")
            .local_speed(true)
            .expected_updates(Some(num_of_nodes));

        let shared_pl = Arc::new(Mutex::new(pl));
        let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
        thread_pool_builder = thread_pool_builder.num_threads(num_threads);
        let thread_pool = thread_pool_builder
            .build()
            .expect("Error in building thread pool");
        (shared_pl, thread_pool, num_threads)
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
            dsi_progress_logger::no_logging!(),
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
        let mut closeness = 0f64;
        let mut harmonic = 0f64;
        let mut lin;
        let mut exponential = 0f64;
        let mut reachable: usize = 0;

        let (sender, receiver) = crossbeam_channel::unbounded();

        let base = alpha;
        visit.reset();

        thread_pool.in_place_scope(|scope| {
            scope.spawn(|_| {
                for distance in receiver {
                    let d = distance;
                    reachable += 1;
                    if d == 0 {
                        //Skip first
                        continue;
                    }
                    let hd = 1f64 / d as f64;
                    let ed = base.pow(d as f64);
                    closeness += d as f64;
                    harmonic += hd;
                    exponential += ed;
                }
            });
            visit
                .par_visit(
                    start,
                    |args| match args {
                        EventPred::Unknown { distance, .. } => {
                            sender.send(distance).expect("Failed sending distance");
                            Ok(())
                        }
                        _ => Ok::<(), ()>(()),
                    },
                    thread_pool,
                    pl,
                )
                .expect("Error in bfs");
            drop(sender);
        });

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
    use crate::geometric_centralities::GeometricCentralities;
    use assert_approx_eq::assert_approx_eq;
    use webgraph::labels::Left;
    use webgraph::prelude::VecGraph;
    use webgraph::traits::SequentialLabeling;

    fn transpose_arc_list(
        arcs: impl IntoIterator<Item = (usize, usize)>,
    ) -> impl IntoIterator<Item = (usize, usize)> {
        arcs.into_iter().map(|(a, b)| (b, a))
    }

    fn new_directed_cycle(num_nodes: usize) -> VecGraph {
        let mut graph = VecGraph::new();
        for i in 0..num_nodes {
            graph.add_node(i);
        }
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if (i + 1) % num_nodes == j {
                    graph.add_arc(i, j);
                }
            }
        }
        graph
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
                10,
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
