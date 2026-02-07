use burn::prelude::Backend;
use rand::{SeedableRng, rngs::SmallRng};

use crate::{
    core::{
        Limit,
        position::Position,
        search::mcts::{
            back::{Backpropagater, DefaultBackuper},
            eval::{Evaluator, nn::NNEvaluator, none::NoneEvaluator, r#static::StaticEvaluator},
            limiter::DefaultLimiter,
            nn::{Model, ModelConfig},
            node::Tree,
            noise::{DirichletNoiser, Noiser, NullNoiser},
            search::TreeSearcher,
            select::{PuctSelector, Selector, UcbSelector},
            strategy::MctsStrategy,
        },
    },
    misc::DebugMode,
    uci::sync::CancellationToken,
};

use std::time::Instant;

pub mod back;
pub mod eval;
pub mod limiter;
pub mod nn;
pub mod node;
pub mod noise;
pub mod search;
pub mod select;
pub mod strategy;
pub mod utils;

pub mod test;

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
        let backprop = parts.backprop();
        let noiser = parts.noiser();

        let mut searcher = TreeSearcher::<{ config::MPV }, _, _, _, _, _>::new(
            tree,
            pos.clone(),
            selector,
            limiter.clone(),
            evaluator,
            backprop,
            noiser,
        );

        searcher.grow();
        strategy.step(tree);
    }

    strategy.result(state.tree())
}

pub trait MctsParts<const X: usize = { config::MPV }> {
    type Selector: Selector;
    type Evaluator: Evaluator;
    type Backprop: Backpropagater;
    type Noiser: Noiser;

    fn selector(&self) -> Self::Selector;
    fn evaluator(&self) -> Self::Evaluator;
    fn backprop(&self) -> Self::Backprop;
    fn noiser(&self) -> Self::Noiser;
}

pub trait MctsState {
    fn tree(&mut self) -> &mut Tree;
}

// todo:
// instead of storing the gametree in between moves, try using bump-allocation
// to allocate all the nodes. Maybe the speed up is better than storing the
// compute? (We can't do both, since with bump allocation we either would have
// to move the subtree to a new `Bump`, or we would just not be
// able to deallocate the unused nodes. If our search is *that* slow that we
// aren't even using that much memory for the Tree, maybe just risc having a
// huge memory leak for each `ucinewgame` then :3 idk) or copy all the used
// nodes to a new bump arena... you'd have to do that node by node which could
// be really slow. <todo:benchmark />
//
/// # The search state.
///
/// Either we have ownership of a search-tree, or we have the join handle of the
/// thread that will give us back the ownership of the search-tree.
///
/// (An option because maybe we just started something else like perft or some
/// sht)
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

/// Mcts parts for mcts with puct + nn analysis.
#[derive(Debug)]
pub struct NNParts<B: Backend> {
    /// NN Model
    pub nn: Box<Model<B>>,

    /// Hardware abstraction for the nn model
    pub device: B::Device,
}

impl<'a, B: Backend, const X: usize> MctsParts<X> for &'a NNParts<B> {
    type Selector = PuctSelector<X>;
    type Evaluator = NNEvaluator<'a, 'a, B, X>;
    type Backprop = DefaultBackuper;
    type Noiser = DirichletNoiser;

    fn selector(&self) -> Self::Selector {
        PuctSelector::<X>::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        NNEvaluator::<_, X>::new(&self.nn, &self.device)
    }

    fn backprop(&self) -> Self::Backprop {
        Default::default()
    }

    fn noiser(&self) -> Self::Noiser {
        let rng = SmallRng::from_os_rng();
        DirichletNoiser::new(0.3, 0.25, rng)
    }
}

impl<B: Backend> Default for NNParts<B> {
    fn default() -> Self {
        Self::from_path("./weights")
    }
}

impl<B: Backend> NNParts<B> {
    pub fn from_path(_nn_path: &str) -> Self {
        let device = B::Device::default();
        let nn = ModelConfig::new().init(&device); // todo: read from nn_path
        Self::new(nn, device)
    }

    pub fn new(nn: Model<B>, device: B::Device) -> Self {
        Self { nn: Box::new(nn), device }
    }
}

/// Mcts parts for mcts with puct + static analysis.
#[derive(Debug, Default)]
pub struct StaticParts;

impl<const X: usize> MctsParts<X> for &StaticParts {
    type Selector = PuctSelector<X>;
    type Evaluator = StaticEvaluator<X>;
    type Backprop = DefaultBackuper;
    type Noiser = NullNoiser;

    fn selector(&self) -> Self::Selector {
        Default::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        Default::default()
    }

    fn backprop(&self) -> Self::Backprop {
        Default::default()
    }

    fn noiser(&self) -> Self::Noiser {
        NullNoiser
    }
}

/// Mcts parts for pure mcts.
#[derive(Debug, Default)]
pub struct PureParts;

impl<const X: usize> MctsParts<X> for &PureParts {
    type Selector = UcbSelector<X>;
    type Evaluator = NoneEvaluator<X>;
    type Backprop = DefaultBackuper;
    type Noiser = NullNoiser;

    fn selector(&self) -> Self::Selector {
        Default::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        Default::default()
    }

    fn backprop(&self) -> Self::Backprop {
        Default::default()
    }

    fn noiser(&self) -> Self::Noiser {
        Default::default()
    }
}

pub mod config {
    #[cfg(feature = "mcts-pure")]
    pub const MPV: usize = 1;

    #[cfg(all(feature = "nn-backend-cuda", not(feature = "mcts-pure")))]
    pub const MPV: usize = 32;

    #[cfg(all(feature = "nn-backend-ndarray", not(feature = "mcts-pure")))]
    pub const MPV: usize = 4;

    #[cfg(feature = "nn-backend-cuda")]
    pub mod nn_backend {
        pub type Backend = burn_cuda::Cuda<f32>;
        pub type Device = <self::Backend as burn::prelude::Backend>::Device;
    }

    #[cfg(feature = "nn-backend-ndarray")]
    pub mod nn_backend {
        use burn::backend::NdArray;
        pub type Backend = NdArray;
        pub type Device = <self::Backend as burn::prelude::Backend>::Device;
    }

    #[cfg(feature = "mcts-nn")]
    pub mod mcts {
        use crate::core::search::mcts::NNParts;
        pub type Parts = NNParts<super::nn_backend::Backend>;
    }

    #[cfg(feature = "mcts-sa")]
    pub mod mcts {
        use crate::core::search::mcts::StaticParts;
        pub type Parts = StaticParts;
    }

    #[cfg(feature = "mcts-pure")]
    pub mod mcts {
        use crate::core::search::mcts::PureParts;
        pub type Parts = PureParts;
    }
}
