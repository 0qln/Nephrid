use burn::prelude::Backend;
use rand::{SeedableRng, rngs::SmallRng};
use thiserror::Error;

use crate::{
    core::{
        Limit,
        config::Configuration,
        position::Position,
        search::mcts::{
            back::{Backpropagater, DefaultBackuper},
            eval::{
                Evaluator, nn::NNEvaluator, playout::PlayoutEvaluator, r#static::StaticEvaluator,
            },
            limiter::DefaultLimiter,
            nn::{LoadNNError, Model},
            node::Tree,
            noise::{DirichletNoiser, Noiser, NullNoiser},
            search::TreeSearcher,
            select::{Selector, puct::PuctSelector, ucb::UcbSelector},
            strategy::MctsStrategy,
        },
    },
    misc::DebugMode,
    uci::sync::CancellationToken,
};

use std::{path::PathBuf, time::Instant};

pub mod back;
pub mod eval;
pub mod limiter;
pub mod nn;
pub mod node;
pub mod noise;
pub mod search;
pub mod select;
pub mod strategy;

pub mod test;

pub fn mcts<S: MctsStrategy, P: MctsParts, M: MctsState>(
    pos: &mut Position,
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
    let mut iterations = 0;

    println!("time: {}", time_per_move.as_secs_f32());

    strategy.start(tree);

    let nodes_begin = tree.size() as u64;

    let mut searcher = TreeSearcher::<{ config::MPV }, _, _, _, _, _>::new(
        pos,
        parts.selector(),
        limiter.clone(),
        parts.evaluator(),
        parts.backprop(),
        parts.noiser(),
    );

    searcher.init_root(tree);

    while !(ct.is_cancelled()
        || limit.is_active()
            && limit.is_reached(
                tree.size() as u64 - nodes_begin,
                Instant::now(),
                time_limit,
                iterations,
            ))
    {
        searcher.grow(tree);
        strategy.step(tree);
        iterations += 1;
    }

    strategy.result(state.tree())
}

pub trait MctsParts<const X: usize = { config::MPV }> {
    type Selector: Selector;
    type Evaluator: Evaluator;
    type Backprop: Backpropagater;
    type Noiser: Noiser;
    type Instance: for<'a> TryFrom<&'a Configuration>;

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

    // Noiser
    alpha: f32,
    epsilon: f32,
}

impl<'a, B: Backend, const X: usize> MctsParts<X> for &'a NNParts<B> {
    type Selector = PuctSelector;
    type Evaluator = NNEvaluator<'a, 'a, B>;
    type Backprop = DefaultBackuper;
    type Noiser = DirichletNoiser;
    type Instance = NNParts<B>;

    fn selector(&self) -> Self::Selector {
        PuctSelector::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        NNEvaluator::new(&self.nn, &self.device)
    }

    fn backprop(&self) -> Self::Backprop {
        Default::default()
    }

    fn noiser(&self) -> Self::Noiser {
        let rng = SmallRng::from_os_rng();
        DirichletNoiser::new(self.alpha, self.epsilon, rng)
    }
}

#[derive(Error, Debug)]
pub enum CreateNNPartsError {
    #[error("Error while creating nn-parts: {0}")]
    LoadNNError(LoadNNError),
}

impl<B: Backend> TryFrom<&Configuration> for NNParts<B> {
    type Error = CreateNNPartsError;

    fn try_from(config: &Configuration) -> Result<Self, Self::Error> {
        let alpha = config.dirichlet_alpha();
        let epsilon = config.dirichlet_epsilon();
        let weights = PathBuf::from(config.weights_path());
        let device = B::Device::default();
        let nn = Model::try_from((weights, &device)).map_err(Self::Error::LoadNNError)?;
        Ok(Self::new(nn, device, alpha, epsilon))
    }
}

impl<B: Backend> NNParts<B> {
    pub fn new(nn: Model<B>, device: B::Device, alpha: f32, epsilon: f32) -> Self {
        Self {
            nn: Box::new(nn),
            device,
            alpha,
            epsilon,
        }
    }
}

/// Mcts parts for mcts with puct + static analysis.
#[derive(Debug, Default)]
pub struct StaticParts {
    alpha: f32,
    epsilon: f32,
}

impl<const X: usize> MctsParts<X> for &StaticParts {
    type Selector = PuctSelector;
    type Evaluator = StaticEvaluator;
    type Backprop = DefaultBackuper;
    type Noiser = DirichletNoiser;
    type Instance = StaticParts;

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
        let rng = SmallRng::from_os_rng();
        DirichletNoiser::new(self.alpha, self.epsilon, rng)
    }
}

impl From<&Configuration> for StaticParts {
    fn from(config: &Configuration) -> Self {
        let alpha = config.dirichlet_alpha();
        let epsilon = config.dirichlet_epsilon();
        Self::new(alpha, epsilon)
    }
}

impl StaticParts {
    pub fn new(alpha: f32, epsilon: f32) -> Self {
        Self { alpha, epsilon }
    }
}

/// Mcts parts for pure mcts.
#[derive(Debug, Default)]
pub struct PureParts;

impl<const X: usize> MctsParts<X> for &PureParts {
    type Selector = UcbSelector;
    type Evaluator = PlayoutEvaluator;
    type Backprop = DefaultBackuper;
    type Noiser = NullNoiser;
    type Instance = PureParts;

    fn selector(&self) -> Self::Selector {
        Default::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        let rng = SmallRng::seed_from_u64(0x_dead_beef_u64);
        PlayoutEvaluator::new(rng)
    }

    fn backprop(&self) -> Self::Backprop {
        Default::default()
    }

    fn noiser(&self) -> Self::Noiser {
        Default::default()
    }
}

impl From<&Configuration> for PureParts {
    fn from(_config: &Configuration) -> Self {
        Self {}
    }
}

pub mod config {
    #[cfg(feature = "mcts-pure")]
    pub const MPV: usize = 1;

    #[cfg(all(feature = "nn-backend-cuda", not(feature = "mcts-pure")))]
    pub const MPV: usize = 32;

    #[cfg(all(feature = "nn-backend-ndarray", not(feature = "mcts-pure")))]
    pub const MPV: usize = 1;

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
