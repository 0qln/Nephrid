use burn::prelude::Backend;
use rand::{SeedableRng, rngs::SmallRng};
use thiserror::Error;

use crate::core::{
    config::Configuration,
    r#move::Move,
    position::Position,
    search::mcts::{
        back::{Backpropagater, MctsSolver},
        eval::{Evaluator, hce::HceEvaluator, nn::NNEvaluator, playout::PlayoutEvaluator},
        nn::{LoadNNError, Model},
        node::Tree,
        noise::{DirichletNoiser, Noiser, NullNoiser},
        search::TreeSearcher,
        select::{Selector, puct::PuctSelector, ucb::UcbSelector},
        strategy::MctsStrategy,
    },
};

use std::{path::PathBuf, rc::Rc};

use std::error::Error as StdError;

pub mod back;
pub mod eval;
pub mod nn;
pub mod node;
pub mod noise;
pub mod search;
pub mod select;
pub mod strategy;

pub mod test;

pub fn mcts<const MPV: usize, C: MctsConfig, M: MctsState>(
    pos: &mut Position,
    parts: &C::Parts,
    state: &mut M,
    mut strat: C::Strat,
) -> <C::Strat as MctsStrategy>::Result {
    let tree = state.tree();

    strat.start(tree, pos);

    let mut searcher = TreeSearcher::<{ MPV }, _, _, _, _>::new(
        pos,
        parts.selector(),
        parts.evaluator(),
        parts.backprop(),
        parts.noiser(),
    );

    searcher.init_root(tree);

    while !strat.should_stop(tree) {
        searcher.grow(tree);
        strat.step(tree);
    }

    strat.result(state.tree())
}

pub trait MctsConfig {
    type Parts: MctsParts;
    type Strat: MctsStrategy;
}

pub trait MctsParts: for<'a> TryFrom<&'a Configuration, Error: StdError> {
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
#[derive(Default)]
pub struct SearchState {
    /// The game tree.
    pub tree: Tree,
    pub back_buffer: Tree,
}

impl SearchState {
    pub fn advance_to(&mut self, mov: Move) {
        let root = self.tree.root();
        if self.tree.node(root).state().has_branches()
            && let Some(new_root) = self.tree.branches_rt(root).iter().find(|b| b.mov() == mov)
        {
            self.tree.advance_to(&mut self.back_buffer, new_root.node());
        }
    }
}

impl MctsState for SearchState {
    fn tree(&mut self) -> &mut Tree {
        &mut self.tree
    }
}

/// Mcts parts for mcts with puct + nn analysis.
#[derive(Debug)]
pub struct NNParts<B: Backend> {
    model: Rc<Model<B>>,
    device: Rc<B::Device>,

    // Noiser
    alpha: f32,
    epsilon: f32,
}

impl<B: Backend> MctsParts for NNParts<B> {
    type Selector = PuctSelector;
    type Evaluator = NNEvaluator<B>;
    type Backprop = MctsSolver;
    type Noiser = DirichletNoiser;

    fn selector(&self) -> Self::Selector {
        PuctSelector::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        NNEvaluator::new(self.model.clone(), self.device.clone())
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
    LoadNNError(#[from] LoadNNError),
}

impl<B: Backend> TryFrom<&Configuration> for NNParts<B> {
    type Error = CreateNNPartsError;

    fn try_from(config: &Configuration) -> Result<Self, Self::Error> {
        let alpha = config.dirichlet_alpha();
        let epsilon = config.dirichlet_epsilon();
        let weights = PathBuf::from(config.weights_path());
        let device = B::Device::default();
        let nn = Model::try_from((weights, &device))?;
        // todo: do this
        // nn.warmup(config::MPV, &device);
        Ok(Self::new(nn, device, alpha, epsilon))
    }
}

impl<B: Backend> NNParts<B> {
    pub fn new(nn: Model<B>, device: B::Device, alpha: f32, epsilon: f32) -> Self {
        Self {
            model: Rc::new(nn),
            device: Rc::new(device),
            alpha,
            epsilon,
        }
    }
}

/// Mcts parts for mcts with puct + static analysis.
#[derive(Debug)]
pub struct HceParts {
    alpha: f32,
    epsilon: f32,
}

impl MctsParts for HceParts {
    type Selector = PuctSelector;
    type Evaluator = HceEvaluator;
    type Backprop = MctsSolver;
    type Noiser = DirichletNoiser;

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

impl From<&Configuration> for HceParts {
    fn from(config: &Configuration) -> Self {
        let alpha = config.dirichlet_alpha();
        let epsilon = config.dirichlet_epsilon();
        Self::new(alpha, epsilon)
    }
}

impl Default for HceParts {
    fn default() -> Self {
        Self::new(0.3, 0.25)
    }
}

impl HceParts {
    pub fn new(alpha: f32, epsilon: f32) -> Self {
        Self { alpha, epsilon }
    }
}

/// Mcts parts for pure mcts.
#[derive(Debug, Default)]
pub struct PureParts;

impl MctsParts for PureParts {
    type Selector = UcbSelector;
    type Evaluator = PlayoutEvaluator;
    type Backprop = MctsSolver;
    type Noiser = NullNoiser;

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
