use crate::core::Limit;
use crate::core::position::Position;
use crate::core::search::mcts::eval::Evaluator;
use crate::core::search::mcts::limiter::DefaultLimiter;
use crate::core::search::mcts::nn::Model;
use crate::core::search::mcts::nn::ModelConfig;
use crate::core::search::mcts::node::Tree;
use crate::core::search::mcts::search::TreeSearcher;
use crate::core::search::mcts::strategy::MctsStrategy;
use crate::misc::DebugMode;
use crate::misc::Somewhere;
use crate::uci::sync::CancellationToken;
use std::error::Error;

use burn::record::CompactRecorder;
use burn_cuda::Cuda;

use std::cell::LazyCell;
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

pub fn mcts<const X: usize, S: MctsStrategy + Default, E: Evaluator<X>>(
    pos: Position,
    tree: &mut Tree,
    model: &mut E,
    limit: Limit,
    debug: DebugMode,
    ct: CancellationToken,
) -> S::Result {
    mcts_inner::<S, E>(pos, model, limit, debug, ct, S::default())
}

fn mcts_inner<S: MctsStrategy>(
    mut pos: Position,
    state: MctsState,
    model: &Model<Backend>,
    limit: Limit,
    _debug: DebugMode,
    ct: CancellationToken,
    mut strategy: S,
) -> (S::Result, MctsState) {
    let limiter = DefaultLimiter::new(limit.clone());

    let time_per_move = limit.time_per_move(&pos);
    let time_limit = Instant::now() + time_per_move;

    let mut tree = state.tree.inner();
    let mut searcher = TreeSearcher::new(&mut tree, model);

    while !ct.is_cancelled() && (!limit.is_active || Instant::now() < time_limit) {
        tree.grow(&mut pos, model, &limiter);
        strategy.step(&mut tree);
    }

    (strategy.result(&mut tree), MctsState::new(tree))
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
    tree: Somewhere<Tree>,

    /// NN Model
    nn: Model<Backend>,
}

impl MctsState {
    pub fn new(tree: Tree) -> Self {
        Self { tree: Somewhere::Here(tree) }
    }

    pub fn load_model(path: &str) -> Result<(), Box<dyn Error>> {
        let record = CompactRecorder::new().load(path.into(), &device)?;

        let model = ModelConfig::new()
            .init::<Backend>(&device)
            .load_record(record);
    }
}

impl Default for MctsState {
    fn default() -> Self {
        Self {
            nn: LazyCell::new(|| ModelConfig::new().load("").init()),
            ..Default::default()
        }
    }
}

#[cfg(feature = "nn-backend-cuda")]
pub type Backend = Cuda;

#[cfg(feature = "nn-backend-cuda")]
pub const MPV: usize = 256;

#[cfg(feature = "nn-backend-ndarray")]
type Backend = NdArray;

#[cfg(feature = "nn-backend-ndarray")]
pub const MPV: usize = 8;
