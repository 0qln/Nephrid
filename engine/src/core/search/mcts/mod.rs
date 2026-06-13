use crate::core::{
    params::{IParams, MctsHceParams, MctsNNParams, MctsPureParams},
    search::mcts::select::puct::PuctParams,
};
use burn::prelude::Backend;
use rand::{SeedableRng, rngs::SmallRng};
use thiserror::Error;

use crate::{
    core::{
        config::Configuration,
        r#move::Move,
        position::Position,
        search::mcts::{
            eval::{
                Evaluator, Ratio, hce::HceEvaluator, nn::NNEvaluator, playout::PlayoutEvaluator,
            },
            nn::{CheckModelHealthError, LoadNNError, Model},
            node::Tree,
            noise::{DirichletNoiser, Noiser, NullNoiser},
            search::TreeSearcher,
            select::{Selector, puct::PuctSelector, ucb::UcbSelector},
            strategy::MctsStrategy,
        },
    },
    misc::CheckHealth,
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
    strat: &mut C::Strat,
) -> <C::Strat as MctsStrategy>::Result {
    let tree = state.tree();

    strat.start(tree, pos);

    let mut searcher = TreeSearcher::<{ MPV }, _, _, _>::new(
        pos,
        parts.params(),
        parts.selector(),
        parts.evaluator(),
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
    type Params: IParams;
    type Selector: Selector;
    type Evaluator: Evaluator;
    type Noiser: Noiser;

    fn params(&self) -> <Self::Params as IParams>::Ref;
    fn selector(&self) -> Self::Selector;
    fn evaluator(&self) -> Self::Evaluator;
    fn noiser(&self) -> Self::Noiser;
    fn warmup(&mut self, _batch_size: usize) -> Result<(), String> {
        Ok(())
    }
}

pub trait MctsState {
    fn tree(&mut self) -> &mut Tree;
}

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
    epsilon: Ratio,
}

impl<B: Backend> MctsParts for NNParts<B> {
    type Selector = PuctSelector;
    type Evaluator = NNEvaluator<B>;
    type Noiser = DirichletNoiser;
    type Params = MctsNNParams;

    fn params(&self) -> ParamsRef {
        todo!()
    }

    fn selector(&self) -> Self::Selector {
        PuctSelector::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        NNEvaluator::new(self.model.clone(), self.device.clone())
    }

    fn noiser(&self) -> Self::Noiser {
        let rng = SmallRng::from_os_rng();
        DirichletNoiser::new(self.alpha, self.epsilon, rng)
    }

    fn warmup(&mut self, batch_size: usize) -> Result<(), String> {
        self.model.warmup(batch_size, &self.device);
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum CreateNNPartsError {
    #[error("Error while creating nn-parts: {0}")]
    LoadNNError(#[from] LoadNNError),

    #[error("Unhealthy epsilon: {0}")]
    BadEpsilon(String),

    #[error("Unhealthy nn: {0}")]
    BadNN(CheckModelHealthError),
}

impl<B: Backend> TryFrom<&Configuration> for NNParts<B> {
    type Error = CreateNNPartsError;

    fn try_from(config: &Configuration) -> Result<Self, Self::Error> {
        let alpha = config.dirichlet_alpha();

        let epsilon = Ratio::new(config.dirichlet_epsilon());
        epsilon.check_health().map_err(Self::Error::BadEpsilon)?;

        let weights = PathBuf::from(config.weights_path());

        let device = B::Device::default();

        let nn = Model::try_from((weights, &device))?;
        nn.check_health().map_err(Self::Error::BadNN)?;

        Ok(Self::new(nn, device, alpha, epsilon))
    }
}

impl<B: Backend> NNParts<B> {
    pub fn new(nn: Model<B>, device: B::Device, alpha: f32, epsilon: Ratio) -> Self {
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
pub struct HceParts<P> {
    alpha: f32,
    epsilon: Ratio,
    params: P,
}

impl<P: IParams> MctsParts for HceParts<P::Ref> {
    type Selector = PuctSelector;
    type Evaluator = HceEvaluator;
    type Noiser = DirichletNoiser;

    fn params(&self) -> P::Ref {
        self.params.clone()
    }

    fn selector(&self) -> Self::Selector {
        PuctSelector::new(self.params.select_cpuct())
    }

    fn evaluator(&self) -> Self::Evaluator {
        HceEvaluator::new(self.params.clone())
    }

    fn noiser(&self) -> Self::Noiser {
        let rng = SmallRng::from_os_rng();
        DirichletNoiser::new(self.alpha, self.epsilon, rng)
    }
}

#[derive(Error, Debug)]
pub enum CreateHcePartsError {
    #[error("Unhealthy epsilon: {0}")]
    BadEpsilon(String),

    #[error("Error while creating evaluator params: {0}")]
    EvalParams(#[from] CreateParamsError),
}

impl<P: (for <'a> TryFrom<&'a Configuration>) + IParams> TryFrom<&Configuration> for HceParts<P::Ref> {
    type Error = CreateHcePartsError;

    fn try_from(config: &Configuration) -> Result<Self, Self::Error> {
        let alpha = config.dirichlet_alpha();

        let epsilon = Ratio::new(config.dirichlet_epsilon());
        epsilon.check_health().map_err(Self::Error::BadEpsilon)?;

        let params = P::try_from(config)?.shared();

        Ok(Self::new(alpha, epsilon, params))
    }
}

impl<P> Default for HceParts<P> {
    fn default() -> Self {
        let config = Configuration::builder()
            .qsearch(&MctsHceParams)
            .policy(&MctsHceParams)
            .puct(&MctsHceParams)
            .mcts(&MctsHceParams)
            .build();
        Self::try_from(&config).expect("The default config should be healthy")
    }
}

impl<P> HceParts<P> {
    pub fn new(alpha: f32, epsilon: Ratio, params: P) -> Self {
        Self { alpha, epsilon, params }
    }
}

/// Mcts parts for pure mcts.
#[derive(Debug, Default)]
pub struct PureParts;

impl MctsParts for PureParts {
    type Selector = UcbSelector;
    type Evaluator = PlayoutEvaluator;
    type Noiser = NullNoiser;
    type Params = MctsPureParams;

    fn params(&self) -> {
        todo!()
    }

    fn selector(&self) -> Self::Selector {
        Default::default()
    }

    fn evaluator(&self) -> Self::Evaluator {
        let rng = SmallRng::seed_from_u64(0x_dead_beef_u64);
        PlayoutEvaluator::new(rng)
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
