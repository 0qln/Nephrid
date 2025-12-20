use burn::prelude::Backend;

use crate::core::Limit;
use crate::core::position::Position;
use crate::core::search::mcts::back::DefaultBackuper;
use crate::core::search::mcts::eval::Evaluator;
use crate::core::search::mcts::eval::nn::NNEvaluator;
use crate::core::search::mcts::eval::static_anal::StaticEvaluator;
use crate::core::search::mcts::limiter::DefaultLimiter;
use crate::core::search::mcts::nn::Model;
use crate::core::search::mcts::nn::ModelConfig;
use crate::core::search::mcts::node::Tree;
use crate::core::search::mcts::search::TreeSearcher;
use crate::core::search::mcts::select::PuctSelector;
use crate::core::search::mcts::select::Selector;
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

// todo: add ability to specify custom selector, e.g. for training try to use a selector that
// weighs the actualy game results higher than the value estimations... maybe that will help idk
// though
pub fn mcts<S: MctsStrategy, P: MctsParts, M: MctsState>(
    pos: &Position,
    parts: P,
    state: &mut M,
    limit: Limit,
    _debug: DebugMode,
    ct: CancellationToken,
    mut strategy: S,
) -> S::Result {
    let limiter = DefaultLimiter::new(limit.clone());

    let time_per_move = limit.time_per_move(pos);
    let time_limit = Instant::now() + time_per_move;
    let tree = state.tree();

    while !ct.is_cancelled() && (!limit.is_active || Instant::now() < time_limit) {
        let evaluator = parts.evaluator();
        let selector = parts.selector();

        let mut searcher = TreeSearcher::<{ config::MPV }, _, _, _, DefaultBackuper>::new(
            tree,
            pos.clone(),
            selector,
            limiter.clone(),
            evaluator,
        );

        searcher.grow();
        strategy.step(tree);
    }

    strategy.result(state.tree())
}

pub trait MctsParts<const X: usize = { config::MPV }> {
    type Selector: Selector;
    type Evaluator: Evaluator;

    fn selector(&self) -> Self::Selector;
    fn evaluator(&self) -> Self::Evaluator;
}

pub trait MctsState {
    fn tree(&mut self) -> &mut Tree;
}

// todo:
// instead of storing the gametree in between moves, try using bump-allocation to allocate all the
// nodes. Maybe the speed up is better than storing the compute? (We can't do both, since with bump
// allocation we either would have to move the subtree to a new `Bump`, or we would just not be
// able to deallocate the unused nodes. If our search is *that* slow that we aren't even using that
// much memory for the Tree, maybe just risc having a huge memory leak for each `ucinewgame` then
// :3 idk) or copy all the used nodes to a new bump arena... you'd have to do that node by node
// which could be really slow. <todo:benchmark />
//
/// # The search state.
///
/// Either we have ownership of a search-tree, or we have the join handle of the thread that
/// will give us back the ownership of the search-tree.
///
/// (An option because maybe we just started something else like perft or some sht)
#[derive(Default, Debug)]
pub struct SearchState {
    /// The game tree.
    pub tree: Tree,
}

impl MctsState for SearchState {
    fn tree(&mut self) -> &mut Tree {
        &mut self.tree
    }
}

#[derive(Debug)]
pub struct NNState<B: Backend> {
    /// NN Model
    pub nn: Box<Model<B>>,

    /// Hardware abstraction for the nn model
    pub device: B::Device,
}

impl<'a, B: Backend, const X: usize> MctsParts<X> for &'a NNState<B> {
    type Selector = PuctSelector<X>;

    type Evaluator = NNEvaluator<'a, 'a, B, X>;

    fn selector(&self) -> Self::Selector {
        PuctSelector::<X>::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        NNEvaluator::<_, X>::new(&self.nn, &self.device)
    }
}

impl<B: Backend> Default for NNState<B> {
    fn default() -> Self {
        Self::from_path("./weights")
    }
}

impl<B: Backend> NNState<B> {
    pub fn from_path(_nn_path: &str) -> Self {
        let device = B::Device::default();

        // todo: read from nn_path
        let nn = ModelConfig::new().init(&device);

        Self::new(nn, device)
    }

    pub fn new(nn: Model<B>, device: B::Device) -> Self {
        Self { nn: Box::new(nn), device }
    }
}

#[derive(Debug, Default)]
pub struct StaticAnalState;

impl<const X: usize> MctsParts<X> for &StaticAnalState {
    type Selector = PuctSelector<X>;
    type Evaluator = StaticEvaluator<X>;

    fn selector(&self) -> Self::Selector {
        Default::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        Default::default()
    }
}

#[cfg(feature = "nn-backend-cuda")]
pub mod config {
    pub type Backend = burn_cuda::Cuda<f32>;
    pub type Device = <self::Backend as burn::prelude::Backend>::Device;
    pub const MPV: usize = 32;
}

#[cfg(feature = "nn-backend-ndarray")]
pub mod config {
    type Backend = NdArray;
    pub const MPV: usize = 4;
}
