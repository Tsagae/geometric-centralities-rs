use atomic_counter::AtomicCounter;
use common_traits::Number;
use crossbeam_channel::unbounded;
use dsi_progress_logger::{ProgressLog, ProgressLogger};
use rayon::ThreadPool;
use std::cmp::min;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread::available_parallelism;
use sux::bits::BitVec;
use webgraph::traits::RandomAccessGraph;
use webgraph_algo::prelude::breadth_first::{EventPred, Seq, SeqNoKnown};
use webgraph_algo::traits::Sequential;

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
    pub results: Vec<GeometricCentralityResult>,
    graph: &'a G,
    num_of_threads: usize,
    atomic_counter: Arc<atomic_counter::ConsistentCounter>,
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
            closeness: vec![],
            harmonic: vec![],
            lin: vec![],
            exponential: vec![],
            reachable: vec![],
            results: vec![],
            atomic_counter: Arc::new(atomic_counter::ConsistentCounter::new(0)),
        }
    }

    fn init<'a, P: ProgressLog + Send + Sync>(&mut self, pl: &'a mut P) -> (Arc<Mutex<&'a mut P>>, ThreadPool, usize) {
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

    fn collect_results(&mut self, num_of_nodes: usize, receive_from_thread: crossbeam_channel::Receiver<(usize, GeometricCentralityResult)>) {
        for _ in 0..num_of_nodes {
            let (
                node,
                GeometricCentralityResult {
                    closeness,
                    harmonic,
                    lin,
                    exponential,
                    reachable,
                },
            ) = receive_from_thread
                .recv()
                .expect("Failed receiving from thread");
            self.closeness[node] = closeness;
            self.harmonic[node] = harmonic;
            self.lin[node] = lin;
            self.exponential[node] = exponential;
            self.reachable[node] = reachable;
        }
    }

    pub fn compute_with_atomic_counter_out_channel_generic<P: ProgressLog + Send + Sync>(&mut self, pl: &mut P) {
        let num_of_nodes = self.graph.num_nodes();
        let (shared_pl, thread_pool, num_threads) = self.init::<P>(pl);
        let (send_out_of_thread, receive_from_thread) = unbounded();
        thread_pool.in_place_scope(|scope| {
            for i in 0..num_threads {
                let local_send_out_of_thread = send_out_of_thread.clone();
                let thread_atomic_counter = Arc::clone(&self.atomic_counter);
                let num_of_nodes = self.graph.num_nodes();
                let graph = self.graph;
                let local_pl = Arc::clone(&shared_pl);
                scope.spawn(move |_| {
                    let mut bfs = Seq::new(graph);

                    //println!("Started thread id: {:?}", thread::current().id());
                    let atom_counter = thread_atomic_counter;
                    let num_of_nodes = num_of_nodes;

                    let mut target_node = atom_counter.inc();
                    while target_node < num_of_nodes {
                        let centralities = Self::single_visit_generic(target_node, &mut bfs);
                        local_send_out_of_thread
                            .send((target_node, centralities))
                            .unwrap_or_else(|_| panic!("Failed send out of thread {i} target_node: {target_node}"));
                        target_node = atom_counter.inc();
                        {
                            let mut pl = local_pl.lock().expect("Error in taking mut pl");
                            pl.update();
                        }
                    }
                });
            }
            self.collect_results(num_of_nodes, receive_from_thread);
        });
        {
            let mut pl = shared_pl.lock().expect("Error in taking mut pl");
            pl.done_with_count(num_of_nodes);
        }
    }

    pub fn compute_with_atomic_counter_out_channel_generic_no_known<P: ProgressLog + Send + Sync>(&mut self, pl: &mut P) {
        let num_of_nodes = self.graph.num_nodes();
        let (shared_pl, thread_pool, num_threads) = self.init::<P>(pl);
        let (send_out_of_thread, receive_from_thread) = unbounded();
        thread_pool.in_place_scope(|scope| {
            for i in 0..num_threads {
                let local_send_out_of_thread = send_out_of_thread.clone();
                let thread_atomic_counter = Arc::clone(&self.atomic_counter);
                let num_of_nodes = self.graph.num_nodes();
                let graph = self.graph;
                let local_pl = Arc::clone(&shared_pl);
                scope.spawn(move |_| {
                    let mut bfs = SeqNoKnown::new(graph);

                    //println!("Started thread id: {:?}", thread::current().id());
                    let atom_counter = thread_atomic_counter;
                    let num_of_nodes = num_of_nodes;

                    let mut target_node = atom_counter.inc();
                    while target_node < num_of_nodes {
                        let centralities = Self::single_visit_generic_no_known(target_node, &mut bfs);
                        local_send_out_of_thread
                            .send((target_node, centralities))
                            .unwrap_or_else(|_| panic!("Failed send out of thread {i} target_node: {target_node}"));
                        target_node = atom_counter.inc();
                        {
                            let mut pl = local_pl.lock().expect("Error in taking mut pl");
                            pl.update();
                        }
                    }
                });
            }
            self.collect_results(num_of_nodes, receive_from_thread);
        });
        {
            let mut pl = shared_pl.lock().expect("Error in taking mut pl");
            pl.done_with_count(num_of_nodes);
        }
    }

    pub fn compute_with_atomic_counter_out_channel<P: ProgressLog + Send + Sync>(&mut self, pl: &mut P) {
        let num_of_nodes = self.graph.num_nodes();
        let (shared_pl, thread_pool, num_threads) = self.init::<P>(pl);
        let (send_out_of_thread, receive_from_thread) = unbounded();
        thread_pool.in_place_scope(|scope| {
            for i in 0..num_threads {
                let local_send_out_of_thread = send_out_of_thread.clone();
                let thread_atomic_counter = Arc::clone(&self.atomic_counter);
                let num_of_nodes = self.graph.num_nodes();
                let graph = self.graph;
                let local_pl = Arc::clone(&shared_pl);
                scope.spawn(move |_| {
                    //println!("Started thread id: {:?}", thread::current().id());
                    let atom_counter = thread_atomic_counter;
                    let num_of_nodes = num_of_nodes;

                    let mut queue: VecDeque<usize> = VecDeque::new();
                    let mut distances: Vec<i64> = Vec::new();
                    distances.resize(num_of_nodes, -1);

                    let mut target_node = atom_counter.inc();
                    while target_node < num_of_nodes {
                        let centralities = Self::single_visit(graph, target_node, &mut queue, &mut distances);
                        local_send_out_of_thread
                            .send((target_node, centralities))
                            .unwrap_or_else(|_| panic!("Failed send out of thread {i} target_node: {target_node}"));
                        target_node = atom_counter.inc();
                        {
                            let mut pl = local_pl.lock().expect("Error in taking mut pl");
                            pl.update();
                        }
                    }
                });
            }
            self.collect_results(num_of_nodes, receive_from_thread);
        });
        {
            let mut pl = shared_pl.lock().expect("Error in taking mut pl");
            pl.done_with_count(num_of_nodes);
        }
    }

    pub fn compute_with_atomic_counter_out_channel_no_dist_vec<P: ProgressLog + Send + Sync>(&mut self, pl: &mut P) {
        let num_of_nodes = self.graph.num_nodes();
        let (shared_pl, thread_pool, num_threads) = self.init::<P>(pl);
        let (send_out_of_thread, receive_from_thread) = unbounded();
        thread_pool.in_place_scope(|scope| {
            for i in 0..num_threads {
                let local_send_out_of_thread = send_out_of_thread.clone();
                let thread_atomic_counter = Arc::clone(&self.atomic_counter);
                let num_of_nodes = self.graph.num_nodes();
                let graph = self.graph;
                let local_pl = Arc::clone(&shared_pl);
                scope.spawn(move |_| {
                    //println!("Started thread id: {:?}", thread::current().id());
                    let atom_counter = thread_atomic_counter;
                    let num_of_nodes = num_of_nodes;

                    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
                    let mut visited = BitVec::with_value(num_of_nodes, false);

                    let mut target_node = atom_counter.inc();
                    while target_node < num_of_nodes {
                        let centralities = Self::single_visit_no_dist_vec(graph, target_node, &mut queue, &mut visited);
                        local_send_out_of_thread
                            .send((target_node, centralities))
                            .unwrap_or_else(|_| panic!("Failed send out of thread {i} target_node: {target_node}"));
                        target_node = atom_counter.inc();
                        {
                            let mut pl = local_pl.lock().expect("Error in taking mut pl");
                            pl.update();
                        }
                    }
                });
            }
            self.collect_results(num_of_nodes, receive_from_thread);
        });
        {
            let mut pl = shared_pl.lock().expect("Error in taking mut pl");
            pl.done_with_count(num_of_nodes);
        }
    }

    fn single_visit_generic(
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

    fn single_visit_generic_no_known(
        start: usize,
        bfs: &mut SeqNoKnown<&G>,
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

    fn single_visit(
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

    fn single_visit_no_dist_vec(
        graph: &G,
        start: usize,
        queue: &mut VecDeque<(usize, usize)>,
        visited: &mut BitVec,
    ) -> GeometricCentralityResult {
        let mut closeness = 0f64;
        let mut harmonic = 0f64;
        let lin;
        let mut exponential = 0f64;
        let mut reachable: usize = 0;

        let base = DEFAULT_ALPHA;

        queue.clear();
        visited.fill_no_rayon(false);

        queue.push_back((start, 0));

        while let Some((node, distance)) = queue.pop_front() {
            reachable += 1;
            let d = distance + 1;
            let hd = 1f64 / d as f64;
            let ed = base.pow(d as f64);

            for successor in graph.successors(node) {
                if !visited[successor] {
                    visited.set(successor, true);
                    queue.push_back((successor, d));
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
}


#[cfg(test)]
mod tests {
    use crate::geometric_centralities::GeometricCentralities;
    use dsi_progress_logger::ProgressLogger;
    use webgraph::labels::Left;
    use webgraph::prelude::VecGraph;

    fn transpose_arc_list(
        arcs: impl IntoIterator<Item=(usize, usize)>,
    ) -> impl IntoIterator<Item=(usize, usize)> {
        arcs.into_iter().map(|(a, b)| (b, a))
    }

    #[test]
    fn test_geom_atom_out_chan() {
        let g = VecGraph::from_arc_list(transpose_arc_list([(0, 1), (1, 2)]));
        let l = &Left(g);
        let mut centralities = GeometricCentralities::new(&l, 0);
        centralities.compute_with_atomic_counter_out_channel(&mut Option::<ProgressLogger>::None);

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
    fn test_geom_atom_out_chan_generic() {
        let g = VecGraph::from_arc_list(transpose_arc_list([(0, 1), (1, 2)]));
        let l = &Left(g);
        let mut centralities = GeometricCentralities::new(&l, 0);
        centralities.compute_with_atomic_counter_out_channel_generic(&mut Option::<ProgressLogger>::None);

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
    fn test_geom_atom_out_chan_generic_no_kown() {
        let g = VecGraph::from_arc_list(transpose_arc_list([(0, 1), (1, 2)]));
        let l = &Left(g);
        let mut centralities = GeometricCentralities::new(&l, 0);
        centralities.compute_with_atomic_counter_out_channel_generic_no_known(&mut Option::<ProgressLogger>::None);

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
}
