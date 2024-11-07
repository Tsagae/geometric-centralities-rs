use atomic_counter::AtomicCounter;
use common_traits::Number;
use crossbeam_channel::unbounded;
use dsi_progress_logger::{ProgressLog, ProgressLogger};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::cmp::min;
use std::sync::{Arc, Mutex};
use std::thread::available_parallelism;
use webgraph::traits::RandomAccessGraph;
use webgraph_algo::prelude::breadth_first::{QueueItem, SingleThreadedBreadthFirstVisit};
use webgraph_algo::traits::SeqVisit;

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
    pl: ProgressLogger,
    logging: bool,
    atomic_counter: Arc<atomic_counter::ConsistentCounter>,
}

impl<G: RandomAccessGraph + Sync> GeometricCentralities<'_, G> {
    pub fn new(graph: &G, num_of_threads: usize, logging: bool) -> GeometricCentralities<G> {
        let mut pl = ProgressLogger::default();
        pl.display_memory(true)
            .item_name("visit")
            .local_speed(true)
            .expected_updates(Some(graph.num_nodes()));

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
            pl,
            logging,
            closeness: vec![],
            harmonic: vec![],
            lin: vec![],
            exponential: vec![],
            reachable: vec![],
            results: vec![],
            atomic_counter: Arc::new(atomic_counter::ConsistentCounter::new(0)),
        }
    }

    pub fn compute_with_atomic_counter_out_channel(&mut self) {
        let num_of_nodes = self.graph.num_nodes();
        self.closeness = vec![-1f64; num_of_nodes];
        self.harmonic = vec![-1f64; num_of_nodes];
        self.lin = vec![-1f64; num_of_nodes];
        self.exponential = vec![-1f64; num_of_nodes];
        self.reachable = vec![0; num_of_nodes];

        let logging = self.logging;
        let num_threads = min(self.graph.num_nodes(), self.num_of_threads);
        if logging {
            self.pl.start(&format!(
                "Computing geometric centralities with {num_threads} threads..."
            ));
        }
        let shared_pl = Arc::new(Mutex::new(&mut self.pl));

        let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
        thread_pool_builder = thread_pool_builder.num_threads(num_threads);
        let thread_pool = thread_pool_builder
            .build()
            .expect("Error in building thread pool");

        let (send_out_of_thread, receive_from_thread) = unbounded();
        thread_pool.in_place_scope(|scope| {
            for i in 0..num_threads {
                let local_send_out_of_thread = (&send_out_of_thread).clone();
                let thread_atomic_counter = Arc::clone(&self.atomic_counter);
                let num_of_nodes = self.graph.num_nodes();
                let graph = self.graph;
                let local_pl = Arc::clone(&shared_pl);
                scope.spawn(move |_| {
                    let mut bfs: SingleThreadedBreadthFirstVisit<(), &G> =
                        SingleThreadedBreadthFirstVisit::new(graph);

                    //println!("Started thread {}", i);
                    let atom_counter = thread_atomic_counter;
                    let num_of_nodes = num_of_nodes;

                    let mut target_node = atom_counter.inc();
                    while target_node < num_of_nodes {
                        let centralities = Self::single_visit(target_node, &mut bfs);
                        local_send_out_of_thread
                            .send((target_node, centralities))
                            .expect(&format!(
                                "Failed send out of thread {i} target_node: {target_node}"
                            ));
                        target_node = atom_counter.inc();
                        if logging {
                            let mut pl = local_pl.lock().expect("Error in taking mut pl");
                            pl.update();
                        }
                    }
                });
            }

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
        });
        if logging {
            let mut pl = shared_pl.lock().expect("Error in taking mut pl");
            pl.done();
            eprintln!("{}", pl.to_string());
        }
    }

    /*
    pub fn compute_with_2_channels(&mut self) {
        let num_of_nodes = self.graph.num_nodes();
        self.closeness = vec![-1f64; num_of_nodes];
        self.harmonic = vec![-1f64; num_of_nodes];
        self.lin = vec![-1f64; num_of_nodes];
        self.exponential = vec![-1f64; num_of_nodes];
        self.reachable = vec![0; num_of_nodes];

        let logging = self.logging;
        let num_threads = min(self.graph.num_nodes(), self.num_of_threads);
        if logging {
            self.pl.start(&format!(
                "Computing geometric centralities with {num_threads} threads..."
            ));
        }
        let shared_pl = Arc::new(Mutex::new(&mut self.pl));

        let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
        thread_pool_builder = thread_pool_builder.num_threads(num_threads);
        let thread_pool = thread_pool_builder
            .build()
            .expect("Error in building thread pool");

        let (send_to_thread, receive_in_thread) = unbounded();
        let (send_out_of_thread, receive_from_thread) = unbounded();

        let mut sent = 0;
        let mut received = 0;
        for i in 0..min(num_threads * 2, num_of_nodes) {
            send_to_thread.send((false, i)).unwrap();
            sent += 1;
        }

        thread_pool.in_place_scope(|scope| {
            for i in 0..num_threads {
                let local_send_out_of_thread = (&send_out_of_thread).clone();
                let local_receive_in_thread = (&receive_in_thread).clone();
                let num_of_nodes = self.graph.num_nodes();
                let graph = self.graph;
                let local_pl = Arc::clone(&shared_pl);
                scope.spawn(move |_| {
                    //println!("Started thread {}", i);
                    let num_of_nodes = num_of_nodes;
                    let mut distances = vec![-1; num_of_nodes];
                    let mut queue = VecDeque::new();
                    loop {
                        let (stop, target_node) = local_receive_in_thread
                            .recv()
                            .expect("Failed receiving in thread");
                        if stop {
                            break;
                        }

                        let centralities =
                            Self::single_visit(&graph, target_node, &mut distances, &mut queue);
                        local_send_out_of_thread
                            .send((target_node, centralities))
                            .expect(&format!(
                                "Failed send out of thread {i} target_node: {target_node}"
                            ));
                        if logging {
                            let mut pl = local_pl.lock().expect("Error in taking mut pl");
                            pl.update();
                        }
                    }
                });
            }
            let first_sent = sent;
            for _ in 0..(num_of_nodes - first_sent) {
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
                received += 1;

                send_to_thread
                    .send((false, sent))
                    .expect("Failed sending to thread");
                sent += 1;
            }

            for _ in 0..first_sent {
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
                received += 1;
            }

            for i in 0..num_threads {
                send_to_thread
                    .send((true, i))
                    .expect("Failed sending to thread");
            }
        });
        if logging {
            let mut pl = shared_pl.lock().expect("Error in taking mut pl");
            pl.done();
            eprintln!("{}", pl.to_string());
        }
    }
    */
    /*
    pub fn compute_with_par_iter(&mut self, chunk_size: usize) {
        self.results = vec![
            GeometricCentralityResult {
                closeness: -1f64,
                harmonic: -1f64,
                lin: -1f64,
                exponential: -1f64,
                reachable: 0,
            };
            self.graph.num_nodes()
        ];

        let logging = self.logging;
        if logging {
            self.pl.start(&format!(
                "Computing geometric centralities with max threads and {chunk_size} chunks..."
            ));
        }
        let shared_pl = Mutex::new(&mut self.pl);

        self.results
            .par_iter_mut()
            .enumerate()
            .chunks(chunk_size)
            .for_each(|mut vec_chunk| {
                thread_local!(static DISTANCES: RefCell<Vec<i64>> = RefCell::new(Vec::new()));
                thread_local!(static QUEUE: RefCell<VecDeque<usize>> = RefCell::new(VecDeque::new()));

                vec_chunk.iter_mut().for_each(|(node, ref mut item)| {
                    DISTANCES.with_borrow_mut(|distances| {
                        QUEUE.with_borrow_mut(|queue| {
                            **item = Self::single_visit(self.graph, *node, distances, queue);
                        })
                    });
                });
                if logging {
                    let mut pl = shared_pl.lock().expect("Error in taking mut pl");
                    pl.update_with_count(chunk_size);
                }
            });

        if logging {
            let mut pl = shared_pl.lock().expect("Error in taking mut pl");
            pl.done();
            eprintln!("{}", pl.to_string());
        }
    }
    */

    fn single_visit(
        start: usize,
        bfs: &mut SingleThreadedBreadthFirstVisit<(), &G>,
    ) -> GeometricCentralityResult {
        let mut closeness = 0f64;
        let mut harmonic = 0f64;
        let lin;
        let mut exponential = 0f64;
        let mut reachable: usize = 0;

        let base = DEFAULT_ALPHA;

        bfs.reset();
        bfs.visit(
            start,
            |args| {
                let d = args.distance;

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
            },
            &mut ProgressLogger::default(),
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
}

#[cfg(test)]
mod tests {
    use crate::geometric_centralities::GeometricCentralities;
    use webgraph::labels::Left;
    use webgraph::prelude::VecGraph;

    fn transpose_arc_list(
        arcs: impl IntoIterator<Item = (usize, usize)>,
    ) -> impl IntoIterator<Item = (usize, usize)> {
        arcs.into_iter().map(|(a, b)| (b, a))
    }

    #[test]
    fn test_geom_atom_out_chan() {
        let g = VecGraph::from_arc_list(transpose_arc_list([(0, 1), (1, 2)]));
        let l = &Left(g);
        let mut centralities = GeometricCentralities::new(&l, 0, true);
        centralities.compute_with_atomic_counter_out_channel();

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

    /*
    #[test]
    fn test_geom_2_chan() {
        let g = VecGraph::from_arc_list(transpose_arc_list([(0, 1), (1, 2)]));
        let l = &Left(g);
        let mut centralities = GeometricCentralities::new(&l, 0, true);
        centralities.compute_with_2_channels();
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
    fn test_geom_par_iter() {
        let g = VecGraph::from_arc_list(transpose_arc_list([(0, 1), (1, 2)]));
        let l = &Left(g);
        let mut centralities = GeometricCentralities::new(&l, 0, true);
        centralities.compute_with_par_iter(2);
        assert_eq!(0f64, centralities.results[0].closeness);
        assert_eq!(1f64, centralities.results[1].closeness);
        assert_eq!(1f64 / 3f64, centralities.results[2].closeness);

        assert_eq!(1f64, centralities.results[0].lin);
        assert_eq!(4f64, centralities.results[1].lin);
        assert_eq!(3f64, centralities.results[2].lin);

        assert_eq!(0f64, centralities.results[0].harmonic);
        assert_eq!(1f64, centralities.results[1].harmonic);
        assert_eq!(3f64 / 2f64, centralities.results[2].harmonic);
    }
     */
}
