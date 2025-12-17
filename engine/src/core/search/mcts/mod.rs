use crate::core::Limit;
use crate::core::position::Position;
use crate::core::search::mcts::back::DefaultBackuper;
use crate::core::search::mcts::eval::NNEvaluator;
use crate::core::search::mcts::limiter::DefaultLimiter;
use crate::core::search::mcts::nn::Model;
use crate::core::search::mcts::nn::ModelConfig;
use crate::core::search::mcts::node::Tree;
use crate::core::search::mcts::search::TreeSearcher;
use crate::core::search::mcts::select::PuctSelector;
use crate::core::search::mcts::strategy::MctsStrategy;
use crate::misc::DebugMode;
use crate::uci::sync::CancellationToken;

use std::rc::Rc;
use std::time::Instant;

pub mod back;
pub mod eval;
pub mod limiter;
pub mod nn;
pub mod node;
pub mod search;
pub mod select;
pub mod strategy;
pub mod utils;

pub mod test;

pub fn mcts<S: MctsStrategy>(
    pos: Position,
    state: MctsState,
    limit: Limit,
    _debug: DebugMode,
    ct: CancellationToken,
    mut strategy: S,
) -> (S::Result, MctsState) {
    let limiter = DefaultLimiter::new(limit.clone());

    let time_per_move = limit.time_per_move(&pos);
    let time_limit = Instant::now() + time_per_move;

    let nn = Rc::new(state.nn);
    let dev = Rc::new(state.device);
    let mut tree = state.tree;

    while !ct.is_cancelled() && (!limit.is_active || Instant::now() < time_limit) {
        let evaluator = NNEvaluator::<_, { config::MPV }>::new(nn.clone(), dev.clone());
        let mut searcher = TreeSearcher::<
            { config::MPV },
            _,
            _,
            PuctSelector<{ config::MPV }>,
            DefaultBackuper,
        >::new(&mut tree, pos.clone(), limiter.clone(), evaluator);

        searcher.grow();
        strategy.step(&mut tree);
    }

    (
        strategy.result(&mut tree),
        MctsState::reuse(
            tree,
            Rc::into_inner(nn).expect("Not all references were dropped"),
            Rc::into_inner(dev).expect("Not all references were dropped"),
        ),
    )
}

// todo:
// instead of storing the gametree in between moves, try using bump-allocation to allocate all the
// nodes. Maybe the speed up is better than storing the compute? (We can't do both, since with bump
// allocation we either would have to move the subtree to a new `Bump`, or we would just not be
// able to deallocate the unused nodes. If our search is *that* slow that we aren't even using that
// much memory for the Tree, maybe just risc having a huge memory leak for each `ucinewgame` then
// :3 idk)
//
/// # The search state.
///
/// Either we have ownership of a search-tree, or we have the join handle of the thread that
/// will give us back the ownership of the search-tree.
///
/// (An option because maybe we just started something else like perft or some sht)
#[derive(Debug)]
pub struct MctsState {
    /// The game tree.
    pub tree: Tree,

    /// NN Model
    pub nn: Model<config::Backend>,

    /// Hardware abstraction for the nn model
    pub device: config::Device,
}

impl Default for MctsState {
    fn default() -> Self {
        Self::new(Default::default(), "./weights")
    }
}

impl MctsState {
    pub fn new(tree: Tree, _nn_path: &str) -> Self {
        let device = config::Device::default();

        // todo: read from nn_path
        let nn = ModelConfig::new().init(&device);

        Self { tree, device, nn }
    }

    pub fn reuse(tree: Tree, nn: Model<config::Backend>, device: config::Device) -> Self {
        Self { tree, nn, device }
    }
}

#[cfg(feature = "nn-backend-cuda")]
mod config {
    pub type Backend = burn_cuda::Cuda<f32>;
    pub type Device = <self::Backend as burn::prelude::Backend>::Device;
    pub const MPV: usize = 256;
}

#[cfg(feature = "nn-backend-ndarray")]
mod config {
    type Backend = NdArray;
    pub const MPV: usize = 8;
}
