use burn::prelude::Backend;

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

pub fn mcts<S: MctsStrategy, B: Backend>(
    pos: Position,
    state: &mut MctsState<B>,
    limit: Limit,
    _debug: DebugMode,
    ct: CancellationToken,
    mut strategy: S,
) -> S::Result {
    let limiter = DefaultLimiter::new(limit.clone());

    let time_per_move = limit.time_per_move(&pos);
    let time_limit = Instant::now() + time_per_move;

    let nn = &state.nn;
    let dev = &state.device;
    let tree = &mut state.tree;

    while !ct.is_cancelled() && (!limit.is_active || Instant::now() < time_limit) {
        let evaluator = NNEvaluator::<_, { config::MPV }>::new(nn, dev);
        let mut searcher = TreeSearcher::<
            { config::MPV },
            _,
            _,
            PuctSelector<{ config::MPV }>,
            DefaultBackuper,
        >::new(tree, pos.clone(), limiter.clone(), evaluator);

        searcher.grow();
        strategy.step(tree);
    }

    strategy.result(tree)
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
pub struct MctsState<B: Backend> {
    /// The game tree.
    pub tree: Tree,

    /// NN Model
    pub nn: Model<B>,

    /// Hardware abstraction for the nn model
    pub device: B::Device,
}

impl<B: Backend> Default for MctsState<B> {
    fn default() -> Self {
        Self::from_path(Default::default(), "./weights")
    }
}

impl<B: Backend> MctsState<B> {
    pub fn from_path(tree: Tree, _nn_path: &str) -> Self {
        let device = B::Device::default();

        // todo: read from nn_path
        let nn = ModelConfig::new().init(&device);

        Self::new(tree, nn, device)
    }

    pub fn new(tree: Tree, nn: Model<B>, device: B::Device) -> Self {
        Self { tree, nn, device }
    }
}

#[cfg(feature = "nn-backend-cuda")]
pub mod config {
    pub type Backend = burn_cuda::Cuda<f32>;
    pub type Device = <self::Backend as burn::prelude::Backend>::Device;
    pub const MPV: usize = 64;
}

#[cfg(feature = "nn-backend-ndarray")]
pub mod config {
    type Backend = NdArray;
    pub const MPV: usize = 4;
}
