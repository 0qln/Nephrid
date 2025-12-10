use crate::core::Move;
use crate::core::Position;
use crate::core::search::ControlFlow;
use crate::core::search::fold_legal_moves;
use crate::core::turn::Turn;
use std::assert_matches::assert_matches;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

#[derive(Default, Debug, Clone)]
pub struct Tree {
    /// Root of the tree.
    root: Rc<RefCell<Node>>,
}

impl Tree {
    pub fn new() -> Self {
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

    /// The turn of the current player
    turn: Turn,

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

    fn turn(&self) -> Turn {
        self.turn
    }
}
