use crate::core::search::mcts::eval::model::BoardInputTensor;
use crate::core::search::mcts::eval::model::board_input;
use burn::tensor::backend::Backend;
use itertools::Itertools;
use rand::rngs::SmallRng;
use ringbuf::{StaticRb, traits::*};
use std::assert_matches::assert_matches;
use std::fmt;
use std::ops::ControlFlow;
use std::ptr::NonNull;

use crate::core::depth::Depth;
use crate::core::position::CheckState;
use crate::core::search::mcts::eval::model::BOARD_INPUT_HISTORY;
use crate::core::search::mcts::eval::model::POLICY_OUTPUTS;
use crate::core::{color::Color, r#move::Move, move_iter::fold_legal_moves, position::Position};

pub mod eval;

pub mod test;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GameResult {
    Win { relative_to: Color },
    Draw,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Guess {
    relative_to: Color,
    quality: f32,
    policies: Vec<f32>,
}

impl Guess {
    pub fn policies(&self) -> &[f32] {
        &self.policies
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Evaluation {
    // we will go further and have a gues about this game.
    Guess(Guess),
    // we cannot go any further.
    Terminal(GameResult),
    // we don't feel like going any further.
    Nope,
}

impl GameResult {
    /// Returns a number between 0 and 1, where 0 is a loss and 1 is a win.
    const fn to_value(self, turn: Color) -> f32 {
        match self {
            Self::Win { relative_to } => {
                if relative_to.v() == turn.v() {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Draw => 0.5,
        }
    }
}

impl Evaluation {
    /// Returns a number between 0 and 1, where 0 is a loss and 1 is a win.
    fn to_value(&self, turn: Color) -> f32 {
        match self {
            Self::Terminal(result) => result.to_value(turn),
            Self::Nope => GameResult::Draw.to_value(turn),
            Self::Guess(Guess {
                quality,
                relative_to,
                policies: _policies,
            }) => {
                // The quality is between -1 and 1, so we have to convert it to a 0 to 1 range.
                let quality = (quality + 1.0) / 2.0;
                if *relative_to == turn { quality } else { 1.0 - quality }
            }
        }
    }
}

// pub trait Worker {
//     fn process(ct: CancellationToken);

//     fn start(ct: CancellationToken) {
//         while !ct.is_cancelled() {
//             callback(ct.clone());
//         }
//     }
// }

// pub struct SelectionWorker {

// }

// pub struct EvaluationWorker {
//     queue: Vec<>;
// }

// impl Worker for EvaluationWorker {
//     fn process(ct: CancellationToken) {

//     }
// }

// pub trait Searcher {}

pub trait Selector {
    // note: we take the policy as an argument, because if we later convert this
    // tree structure to a graph, we have to consider different policies from different parents.
    // same reason that we have different struct for node and branch.

    /// # Selector::score
    ///
    /// The score that the selector would assign to a branch.
    ///
    /// ## Params
    ///
    /// branch: The branch to be scored.
    /// cap_n_i: The number of times that the parent node has been visited.
    fn score(&self, branch: &Branch, cap_n_i: u32) -> f32;
}

#[derive(Debug)]
pub struct PuctSelector {
    // todo: fine tune c.
    c: f32,
}

impl Default for PuctSelector {
    pub fn new(c: f32) -> Self {
        Self { c }
    }

    pub fn default() -> Self {
        Self { c: f32::sqrt(2.0) };
    }
}

impl Selector for PuctSelector {
    fn puct(&self, branch: &Branch, cap_n_i: u32) -> f32 {
        let n_i = branch.visits() as f32;

        // The quality is updated incrementally as the tree is explored.
        // Because of this, we have to divide by the number of playouts
        // to get the average quality of this node.
        // If this node has not yet been visited, we set the quality to 0.
        let exploitation = if n_i == 0.0 { 0.0 } else { branch.value() / n_i };

        let exploration = self.c * branch.policy() * (cap_n_i as f32).sqrt() / (1f32 + n_i);

        exploitation + exploration
    }
}

// #[derive(Debug)]
// pub struct PuctWithTempSelector {
//     rng: SmallRng,
//     puct: PuctSelector
// }

// impl PuctWithTempSelector {
//     pub fn new(seed: u64) -> Self {
//         let rng = SmallRng::seed_from_u64(seed);
//         let puct =
//         Slef { rng, puct: }
//     }
// }

/// # Tree searcher
///
/// This statemachine should hold all the info that is required to search.
///
/// This is the implementation for the nn-backed eval model searcher. (e.g. lc0, a0)
///
/// ## Multi pv.
///
/// Use multi pv lines with batched evaluation, if we have access to GPU and can parallelize. If we
/// don't have access to hardware accell, resort to using a backend like WebGPU, or NdArray, and just do
/// TreeSearcher<MPV=1>.
///
/// The pv lines are the top few lines sorted by puct. (e.g.:
/// ```rust
/// self.multi_select(|b| b.puct(self))
/// ```
/// )
///
/// Idea: Maybe we can have lines that are a lot more explorative next to the main puct line?
pub struct TreeSearcher<'a, E: Evaluator, L: Limiter, const MPV: usize> {
    /// RNG
    rng: SmallRng,

    // todo: be careful when we dereference the selection, there might be collisions in the tree.
    /// Stack of nodes that were selected during the selection phase, for each principal line.
    selection: [(Weak<RefCell<Node>>, Rc<RefCell<EvalInfoNode>>); MPV],

    // todo: implement this thingy first, then see if we can refactor this into the Evaluator.
    /// Eval info
    eval_info_root: EvalInfoNode,

    /// Evaluations of the selected lines
    evaluations: [Evaluation; MPV],

    /// The position that will be edited during the selection and backpropagatation.
    position: Position,

    /// Evaluator to evaluate non-terminating nodes.
    evaluater: E,

    /// Limiter to dicide whether to keep searching non-terminating nodes.
    limiter: L,

    /// The tree to be searched.
    tree: &'a mut Tree,
}

struct EvalInfoNode {
    board_info: BoardInputFloats,
    state_info: StateInputFloats,

    children: Vec<Rc<RefCell<EvalInfoNode>>>,
    parent: Rc<RefCell<EvalInfoNode>>,
}

// struct SelectionNode {
//     node: Weak<RefCell<Node>>,
//     parent: Rc<RefCell<SelectionNode>>,
//     children: Vec<Rc<RefCell<SelectionNode>>>,
//     board: BoardHistoryBuffer,
//     state: StatsInputFloats
// }

impl<'a, const MPV: usize, E: Evaluator, L: Limiter> TreeSearcher<'a, MPV, E, L> {
    fn grow(&mut self) -> () {
        self.prepare_grow();
        self.select_lines();
        self.eval_leafes_();
        self.backup_evals();
    }

    fn prepare_grow(&mut self) -> () {}

    //
    // # Select the leafes.
    //
    fn select_lines(&mut self) -> () {
        //
        // todo: convert to iterative approach -- or not if the stack frame is
        // small with this one ._.
        //
        self.select_branches(MPV, 0, Depth::MIN, &self.tree.root());
    }

    fn select_branches(
        &mut self,
        budget: usize,
        line_index: usize,
        depth: Depth,
        current: Rc<RefCell<Node>>,
    ) {
        // 0. Make sure that we do nothing when we have no budget.
        if budget == 0 {
            return;
        }

        // 1. Just follow the line if we only got a budget of 1 left.
        if budget == 1 {
            self.follow_line(1, line_index, depth, current);
            return;
        }

        // 2. If we have more budget, split it up between this and the subsequent best nodes.
        let root_visits = self.tree.root().borrow().visits();
        let selector = PuctSelector::default();
        self.tree
            .root()
            .sort_by(|b| -selector.score(b, root_visits));

        let mut budget = budget;
        let mut line_index = line_index;
        let mut branch_index = 0;
        while budget >= 1 {
            // note: if current is root, then root has to be expanded by now.
            // todo: that is currently not ensured.
            if let Some(branch) = current.get_branch(branch_index) {
                // todo: maybe make this relative to the branch's puct score.
                let current_budget = (budget as f32 * 0.3) as usize;

                // follow the line and decide what to do depending on the fucking state.
                self.follow_line(current_budget, line_index, depth, branch);

                budget -= current_budget;
                line_index += current_budget;
                branch_index += 1;
            } else {
                // in this case there are no more branches to distribute the budget to.
                // todo:
                // this currently wastes some remaining budget, fix that.
                // or later if we make the selection and eval parallel, just go select as many
                // lines as possible, until we have a full (X==MPV) batch.
                break;
            }
        }
    }

    fn push_eval_info(&mut self) -> () {
        // todo: push the current board and state
    }

    /// Follows a branch and decides what to do depending on the current state of the branch's
    /// node.
    fn follow_line(&mut self, budget: usize, line_index: usize, depth: Depth, branch: &Branch) {
        // i don't know if this is the right place to do this
        self.push_eval_info();

        let node = branch.node();
        match node.borrow().state() {
            // Only if the node is already expanded we want to follow the branch.
            NodeState::Expanded => {
                // make the move of the current branch, such that we can follow the line.
                self.pos.make_move(branch.mov());

                // here we want to select multiple branches again, iff the budget allows to (that
                // will be decided in `select_branches`).
                self.select_branches(budget, line_index, depth + 1, node);

                // undo the move again
                self.pos.unmake_move(branch.mov())
            }
            NodeState::Leaf => {
                // expand the node's branches for future mcts iterations.
                //
                // this is fine to do now, since we split above and in the fn `select_branches` we
                // immidieatly up the branch_index and thus during a single selection phase, this
                // match block is only reached once. Altough we should probably unit test this
                // somehow. (todo)
                node.borrow_mut()
                    .expand(self.pos, self.evaluator, self.limiter, depth);

                // select the node.
                self.selection[line_index] = (node.downgrade(), self.current_eval_info_node());
            }
            NodeState::Terminal => {
                // select the node.
                self.selection[line_index] = (node.downgrade(), self.current_eval_info_node());
            }
        }
    }

    fn eval_leafes_(&mut self) -> () {
        let mut batch: Vec<usize> = vec![];

        // 1. Loop over the selection and
        // 1.a: Node is terminal => safe the evaluation
        // 1.b: Node is on-going => safe the nodes index into a batch buffer.
        for (leaf, i) in self.selection.enumerate() {
            let evaluation = &mut self.evaluation[i];
            if let Some(leaf) = leaf.upgrade() {
                // Check if the board has a terminal evaluation
                if let Some(terminal_eval) = E::eval_terminal(leaf.borrow(), self.pos, depth) {
                    *evaluation = terminal_eval;
                }
                // If not, note down that the need to guess this node's evaluation.
                else {
                    to_be_guessed.push((i,));
                }
            }
        }

        // 2. For all nodes in the batch buffer, build the input tensors and evaluate them using
        //    the eval model.
        //
        // todo: this belongs in the evaluator. refactor later.
        // self.evaluator.evaluate()
        //
        let board_input = self.build_board_batch();
        let state_input = self.build_state_batch();
        let evals = self.evaluator.evaluate(..);
        for (eval, i) in evals.into_iter().enumerate() {
            // safe the evaluation
            self.evaluation[batch[i]] = eval;
        }
    }

    fn build_board_batch(&self, selection_indeces: &[usize]) -> Tensor<B, 4> {
        // concatenate the board inputs along the batch dimension.
        let board_batch = Tensor::cat(
            batch
                .map(|d_idx| self.get_node_histor(d_idx))
                // concatenate the board inputs along the channel dimension.
                .map(|history| {
                    // todo: make sure that we generate the history in the proper order.
                    // currently the ringbuffer might have a history where the youngest is at the index
                    // 0 and then there are boards that are older. like this:
                    //
                    // (the number represents age:)
                    //
                    // 7 6 5 4 3 2 1 0 (would generate correct)
                    //
                    // then if we write once more:
                    // (all elements age by one and the new one overwrite the oldest.)
                    // 0 7 6 5 4 3 2 1
                    //
                    // this is the correct order, but we need to remember to read from index 1 now
                    // instead of index 0...
                    //
                    let history_tensor = todo!();

                    // pad missing history info with zeroes.
                    let padding_len = BOARD_INPUT_HISTORY - history.len();
                    let padding_tensor = BoardInputTensor::<B>::zeros(
                        [
                            X,
                            BOARD_INPUT_CHANNELS,
                            ranks::N_VARIANTS,
                            files::N_VARIANTS,
                        ],
                        device,
                    );

                    // concat padding with history
                    Tensor::cat(vec![padding_tensor, history], 1)
                })
                .collect_vec(),
            0,
        );
    }

    /// return the history where:
    /// the oldest board state is the first index
    /// the youngest board state is the last index
    fn get_node_history(&self, selected_node_index: usize) -> BoardHistoryBuffer {
        // todo:
        // 1. traverse the eval_info in reverse
        // 2. and select all the
        todo!("")
    }

    fn build_state_batch(&self, selection_indeces: &[usize]) -> Tensor<B, 2> {
        // concatenate the state inputs along the batch dimension.
        let state_batch = Tensor::cat(batch.map(|d_idx| self.data[d_idx].2).collect_vec(), 0);
    }

    fn backup_evals(&mut self) -> () {
        for (selected, i) in self.selection.enumerate() {
            let eval = self.evaluation[i];
            // 1. Traverse the selected node in reverse, updating the parents along the way.

            // for branch in selected.iter_mut().rev().map(|n| n.as_mut()) {
            //     pos.unmake_move(branch.mov());
            //     let value = eval.to_value(pos.get_turn());
            //     branch.update_node(value);
            // }

            // 2. If the eval was a guess make sure to also set the policies of the selected leaf.
            // todo
            // if let Evaluation::Guess(Guess { policies, .. }) = evaluation {
            //     current.set_policies(&policies);
            // }
        }
    }
}

// type BoardHistoryBuffer = StaticRb<BoardInputFloats, BOARD_INPUT_HISTORY>;

/// X: batch size
pub struct EvalModel<B: Backend, const X: usize> {
    // data: [(Node, BoardHistoryBuffer, StatsInputFloats); X],
    // model: Model<B>,
}

// impl<B: Backend, const X: usize> EvalModel<B, X> {
//     pub fn new(model: Model<B>, device: &B::Device) -> Self {
//         Self {
//             board_history: vec![ (); BOARD_INPUT_HISTORY ],
//             model,
//         }
//     }
// }

impl<B: Backend, const X: usize> mcts::Evaluator<X> for EvalModel<B, X> {
    // fn push(&mut self, pos: &Position, x: usize) {
    //     let board_input = [board_input(pos)].into();
    //     let state_input = [state_input(pos)].into();

    //     self.board_history[x].push_overwrite(board_input);
    //     self.state[x] = state_input;
    // }

    // fn pop(&mut self, x: usize) {
    //     assert!(
    //         self.board_history[x].len() > BOARD_INPUT_HISTORY,
    //         "Cannot pop the padding."
    //     );
    //     self.board_history[x].pop();
    // }

    fn eval_terminal(node: &Node) -> Option<Evaluation> {
        // First check if the position is a normal game ending.
        if node.branches.is_empty() {
            Some(if pos.get_check_state() != CheckState::None {
                // If in check and no moves, it's a loss for the current player
                Evaluation::Terminal(GameResult::Win { relative_to: !pos.get_turn() })
            } else {
                // Stalemate
                Evaluation::Terminal(GameResult::Draw)
            })
        }
        // Then check if the position has reached some of the extra-rule endings.
        else if pos.has_threefold_repetition()
            || pos.fifty_move_rule()
            || pos.is_insufficient_material()
        {
            Some(Evaluation::Terminal(GameResult::Draw))
        }
        // Then check if we are even interested in searching this line any further.
        else if limiter.should_stop(pos, depth) {
            Some(Evaluation::Nope)
        }
        // Otherwise not a terminal evaluation.
        else {
            None
        }
    }

    fn evaluate(&self) -> [Evaluation; X] {
        let mut x = 0; // current index in the batch

        for (&mut datum, i) in self.data.iter_mut().enumerate() {
            let node = &datum.0;

            result[i] = Either::Left(x);
            batch.push(i);

            continue;
        }

        // Otherwise guess a score.
        {
            let (quality, raw_policy) = evaluator.evaluate();

            let mut policies = Vec::<f32>::new();
            for branch in &node.branches {
                policies.push(raw_policy[usize::from(branch.node.mov)]);
            }

            // Renormalize the policy to a sum of 1, since not all of the probabilities
            // were assigned to moves that are actually playable in this position:

            let policy_sum = {
                let sum = policies.iter().sum();
                if sum == 0.0 {
                    // Fallback to uniform distribution
                    policies.len() as f32
                } else {
                    sum
                }
            };
            for policy in &mut policies {
                *policy /= policy_sum;
            }

            // Evaluator should return a probability distribution.
            let f32_eq = |a: f32, b: f32, e: f32| f32::abs(a - b) < e;
            debug_assert!(f32_eq(policies.iter().sum::<f32>(), 1., 0.001));

            Evaluation::Guess(Guess {
                relative_to: pos.get_turn(),
                quality,
                policies,
            })
        }

        let b_in_idx = self.board_history.len() - BOARD_INPUT_HISTORY;
        let b_in = Tensor::cat(
            self.board_history[b_in_idx..]
                .iter()
                .map(|x| x.1.clone())
                .collect_vec(),
            1,
        );
        let s_in = self.state.clone();
        let (quality, policy) = self.model.forward(b_in, s_in);

        let quality = quality
            .to_data()
            .to_vec::<f32>()
            .expect("Quality could not be converted to vec.");

        let policy = TryInto::<Box<[f32; POLICY_OUTPUTS]>>::try_into(
            policy
                .to_data()
                .to_vec::<f32>()
                .expect("Policy could not be converted to vec.")
                .into_boxed_slice(),
        );

        (quality[0], *policy.unwrap())
    }
}

#[derive(Default, Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: Rc<RefCell<Node>>,
}

impl Tree {
    pub fn new<E: Evaluator, L: Limiter>() -> Self {
        Self {
            root: Rc::new(RefCell::new(Node::leaf())),
            ..Default::default()
        }
    }

    pub fn advance_best(self) -> Option<Tree> {
        let best = self.root.take_best()?;
        Some(Tree { root: best.node.node })
    }

    /// Returns None if there are no moves.
    pub fn best_move(&self) -> Option<Move> {
        let best = self.root.select_best()?;
        Some(best.mov())
    }

    /// Returns the current principal variation.
    pub fn principal_variation(&self) -> Vec<&Branch> {
        let mut buf = Vec::new();
        let mut current = &self.root;
        loop {
            match current.state() {
                NodeState::Expanded => {
                    debug_assert!(
                        !current.branches.is_empty(),
                        "Contradiction: NodeState == Expanded, but there are no branches."
                    );

                    // SAFETY: This branch is only reached when NodeState == Expanded
                    let branch = unsafe { current.select_best().unwrap_unchecked() };
                    buf.push(branch);
                    current = branch.traverse();
                }
                NodeState::Leaf | NodeState::Terminal => {
                    break;
                }
            }
        }
        buf
    }

    pub fn grow<E: Evaluator, L: Limiter>(&mut self, pos: &mut Position, eval: &mut E, l: &L) {
        let evaluation = self.select_leaf_mut(pos, eval, l);
        eval.clear();
        self.backpropagate(pos, &evaluation);
    }

    /// Walk down the tree and select the branches with the highest score, until we find a leaf
    /// node. sono-ato, expand the leaf and return the evaluation of that leaf.
    pub fn select_leaf_mut<E: Evaluator, L: Limiter>(
        &mut self,
        pos: &mut Position,
        model: &mut E,
        l: &L,
    ) -> Evaluation {
        self.selection_buffer.clear();
        model.push(pos);
        let mut current = unsafe { NonNull::from_ref(&self.root).as_mut() };
        let mut depth = Depth::MIN;
        loop {
            match current.state() {
                NodeState::Expanded => {
                    debug_assert!(
                        !current.branches.is_empty(),
                        "Contradiction: NodeState == Expanded, but there are no branches."
                    );

                    // SAFETY: This branch is only reached when NodeState == Expanded
                    let branch = unsafe { current.select_puct_mut().unwrap_unchecked() };
                    pos.make_move(branch.mov());
                    model.push(pos);
                    self.selection_buffer.push(NonNull::from_ref(branch));
                    current = branch.traverse_mut();
                }
                NodeState::Leaf => {
                    return current.expand_and_eval(pos, model, l, depth);
                }
                NodeState::Terminal => {
                    return current.eval(pos, model, l, depth);
                }
            }
            depth += 1;
        }
    }

    pub fn get_root(&self) -> Rc<RefCell<Node>> {
        self.root
    }
}

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
pub enum NodeState {
    /// A leaf is an untouched node.
    #[default]
    Leaf,
    /// An expanded node is a node which has been analized and to be found to have children.
    Expanded,
    /// A terminal node is a node which has been analized and to be found to have no children.
    Terminal,
}

#[derive(Clone, Default, Debug, PartialEq)]
pub struct Branch {
    /// The node that this branch leads to.
    node: Rc<RefCell<Node>>,

    /// The policy of picking this branch.
    policy: f32,

    /// The move that lead to this node.
    mov: Move,
}

impl Branch {
    pub fn puct(&self, cap_n_i: u32) -> f32 {
        self.node.puct(cap_n_i, self.policy)
    }

    pub fn new(m: Move, policy: f32) -> Self {
        Self {
            node: Node::leaf(),
            policy,
            mov: m,
        }
    }

    pub fn visits(&self) -> u32 {
        self.node.visits()
    }

    pub fn mov(&self) -> Move {
        self.node.mov()
    }

    pub fn update_node(&mut self, value: f32) {
        self.node.update(value)
    }

    pub fn traverse(&self) -> &Node {
        &self.node.node
    }

    pub fn traverse_mut(&mut self) -> &mut Node {
        &mut self.node.node
    }

    pub fn node_state(&self) -> NodeState {
        self.traverse().state()
    }

    pub fn node(&self) -> Rc<RefCell<Node>> {
        self.node.clone()
    }
}

// todo: storing Rc<RefCell<Branch/Node>> everywhere is super expensive, but i don't have a better
// solution for that right now :(

#[derive(Clone, Default, PartialEq)]
pub struct Node {
    /// The number of times this node was visited.
    visits: u32,

    /// The value of this node. (~sums all the values of it's children)
    value: f32,

    /// The current state of this node.
    state: NodeState,

    /// All the branches from this node.
    branches: Vec<Branch>,

    // todo: this is only really needed for a simple backup() and select() implementation
    // in the mcts... we don't really need to waste a wide pointer on this...
    /// The parent node.
    parent: Rc<RefCell<Node>>,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("value", &self.value())
            .field("visits", &self.visits())
            .field("state", &self.state())
            .field(
                "branches",
                &self
                    .branches
                    .iter()
                    .filter(|c| c.visits() != 0)
                    .collect_vec(),
            )
            .finish()
    }
}

pub trait Evaluator<const X: usize> {
    /// prepare the newest position that needs to be evaluated.
    fn push(&mut self, pos: &Position, x: usize) -> ();

    /// pop the latest position that was prepared.
    fn pop(&mut self, x: usize) -> ();

    /// clear all prepaered
    // fn clear(&mut self) -> ();

    /// Returns: (quality [-1;1], policy [over ALL moves])
    fn evaluate(&self) -> [(f32, [f32; POLICY_OUTPUTS]); X];
}

pub trait Limiter {
    /// Returns: Whether to stop searching.
    fn should_stop(&self, pos: &Position, depth: Depth) -> bool;
}

#[derive(Default, Debug)]
pub struct NoopLimiter;

impl Limiter for NoopLimiter {
    fn should_stop(&self, _pos: &Position, _depth: Depth) -> bool {
        false
    }
}

impl Node {
    // Sort the branches.
    pub fn sort_by(&mut self, f: Fn(&Branch) -> f32) {
        // todo: the sorting can be done a lot more efficiently:
        // The puct score does not change very often later on, only as we start the search.
        // Also we might only need the first few branches if MPV is low.
        self.branches.sort_by_key(f);
    }

    pub fn get_branch(&self, index: usize) -> Option<&Branch> {
        self.branches.get(index)
    }

    /// Create a new root node.
    // pub fn root<E: Evaluator, L: Limiter>(pos: &Position, eval: &E, l: &L) -> Self {
    //     let mut result = Self::leaf();
    //     // result.expand_and_eval(pos, eval, l, Depth::MIN); todo
    //     result
    // }

    /// Create a new leaf node.
    pub fn leaf() -> Self {
        debug_assert_eq!(
            Self::default(),
            Self {
                state: NodeState::Leaf,
                branches: Vec::new(),
                visits: 0,
                value: 0.0
            }
        );

        Self {
            state: NodeState::Leaf,
            branches: Vec::new(),
            visits: 0,
            value: 0.0,
        }
    }

    /// Update the node with the result of an evaluation.
    pub fn update(&mut self, value: f32) {
        self.visits += 1;
        self.value += value;
    }

    /// Select the branch with the highest PUCT score.
    /// Returns None if there are no branches.
    pub fn select_puct_mut(&mut self) -> Option<&mut Branch> {
        let visits = self.visits();
        self.select_mut(|b| b.puct(visits))
    }

    /// Select the branch with the most visits.
    /// Returns None if there are no branches.
    pub fn select_best(&self) -> Option<&Branch> {
        self.select(|b| b.visits())
    }

    pub fn take_best(self) -> Option<Branch> {
        self.take(|b| b.visits())
    }

    /// Returns None if there are no branches.
    pub fn select<F, T>(&self, transform: F) -> Option<&Branch>
    where
        F: Fn(&Branch) -> T,
        T: PartialOrd,
    {
        self.branches.iter().max_by(|a, b| {
            let a = transform(a);
            let b = transform(b);
            a.partial_cmp(&b).expect("Node comparison failed!")
        })
    }

    /// Returns None if there are no branches.
    pub fn select_mut<F, T>(&mut self, transform: F) -> Option<&mut Branch>
    where
        F: Fn(&Branch) -> T,
        T: PartialOrd,
    {
        self.branches.iter_mut().max_by(|a, b| {
            let a = transform(a);
            let b = transform(b);
            a.partial_cmp(&b).expect("Node comparison failed!")
        })
    }

    /// Returns None if there are no branches.
    pub fn take<F, T>(self, transform: F) -> Option<Branch>
    where
        F: Fn(Branch) -> T,
        T: PartialOrd,
    {
        self.branches.into_iter().max_by(|a, b| {
            let a = transform(a);
            let b = transform(b);
            a.partial_cmp(&b).expect("Node comparison failed!")
        })
    }

    /// Expand the node.
    fn expand(&mut self, pos: &Position) {
        assert_matches!(self.state(), NodeState::Leaf);

        _ = fold_legal_moves(pos, &mut self.branches, |acc, m| {
            ControlFlow::Continue::<(), _>({
                acc.push(Branch::new(m, 0.0));
                acc
            })
        });

        self.state = if self.branches.is_empty() {
            NodeState::Terminal
        } else {
            NodeState::Expanded
        };
    }

    /// Sets the policies of the branches.
    fn set_policies(&mut self, policies: &[f32]) {
        assert!(
            self.branches.len() == policies.len(),
            "There has to be exactly one policy for each branch."
        );

        for (i, branch) in self.branches.iter_mut().enumerate() {
            branch.policy = policies[i];
        }
    }

    fn visits(&self) -> u32 {
        self.visits
    }

    fn value(&self) -> f32 {
        self.value
    }

    fn state(&self) -> NodeState {
        self.state
    }
}
