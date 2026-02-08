use std::cmp::max;

use crate::core::{color::colors, piece::piece_type};

use super::*;

#[derive(Debug, PartialEq, Default)]
pub struct QualityInput {
    w_q: u32,
    b_q: u32,
}

impl QualityInput {
    fn material(pos: &Position, color: Color) -> u32 {
        const PIECE_VALUES: [u32; piece_type::N_VARIANTS] = [0, 1, 3, 3, 5, 8, 0];
        (piece_type::PAWN..piece_type::KING)
            .map(|p| pos.get_bitboard(p, color).pop_cnt() * PIECE_VALUES[p.v() as usize])
            .sum()
    }

    fn psqt(_pos: &Position, _color: Color) -> u32 {
        // todo
        0
    }

    fn value(pos: &Position, color: Color) -> u32 {
        Self::material(pos, color) + Self::psqt(pos, color)
    }

    fn new(pos: &Position) -> Self {
        Self {
            w_q: Self::value(pos, colors::WHITE),
            b_q: Self::value(pos, colors::BLACK),
        }
    }
}

#[derive(Debug, PartialEq, Default)]
pub struct PolicyInput {
    p: RawPolicy,
}

#[derive(PartialEq, Debug)]
pub struct EvalInfo {
    /// The node that this eval info is for.
    node: Rc<RefCell<Node>>,

    /// Quality info for static evaluation
    q_input: QualityInput,

    /// Policy for static evaluation
    p_input: PolicyInput,

    /// Turn of the current player.
    turn: Turn,
}

impl EvalInfo {
    pub fn new(node: Rc<RefCell<Node>>, pos: &Position) -> Self {
        Self {
            node,
            turn: pos.get_turn(),
            q_input: QualityInput::new(pos),
            p_input: PolicyInput::default(),
        }
    }
}

pub type EvalInfoNode = DoubleLinkedNode<Option<EvalInfo>>;

/// X: batch size
#[derive(Default)]
pub struct StaticEvaluator<const X: usize> {
    /// Eval infos
    eval_infos: EvaluationInfos<X, Rc<RefCell<EvalInfoNode>>>,
}

impl<const X: usize> StaticEvaluator<X> {
    pub fn new() -> Self {
        Self {
            eval_infos: EvaluationInfos {
                evals: [const { EvalInfo::None }; X],
                root: None,
            },
        }
    }

    fn iter_batch(&self) -> impl Iterator<Item = Rc<RefCell<EvalInfoNode>>> {
        self.eval_infos.evals.iter().filter_map(|x| {
            if let EvalInfo::OnBatch(eval_info) = x {
                Some(eval_info.clone())
            }
            else {
                None
            }
        })
    }
}

impl<const X: usize> Evaluator for StaticEvaluator<X> {
    type Node = EvalInfoNode;
    type NodeRef = Rc<RefCell<Self::Node>>;

    fn init(&mut self, node: Rc<RefCell<Node>>, pos: &Position) -> Rc<RefCell<Self::Node>> {
        let data = EvalInfo::new(node, pos);
        let root = Rc::new(RefCell::new(Self::Node::new_root(Some(data))));
        self.eval_infos.root = Some(root.clone());
        root
    }

    fn create_data(
        &mut self,
        parent: &mut Rc<RefCell<Self::Node>>,
        node: Rc<RefCell<Node>>,
        pos: &Position,
    ) -> Rc<RefCell<Self::Node>> {
        let data = EvalInfo::new(node, pos);
        Self::Node::append(parent, Some(data))
    }

    fn get_eval(&self, index: usize) -> Option<&Evaluation> {
        if let Some(x) = self.eval_infos.evals.get(index)
            && let EvalInfo::Evaluated(eval) = x
        {
            Some(eval)
        }
        else {
            None
        }
    }

    fn set_eval(&mut self, index: usize, eval: Evaluation) {
        if let Some(x) = self.eval_infos.evals.get_mut(index) {
            *x = EvalInfo::Evaluated(eval);
        }
    }

    fn batch_eval(&mut self, index: usize, eval_node: Rc<RefCell<Self::Node>>) {
        if let Some(x) = self.eval_infos.evals.get_mut(index) {
            *x = EvalInfo::OnBatch(eval_node);
        }
        else {
            panic!("Out of range");
        }
    }

    fn eval_guesses(&mut self) {
        let batch_size = self.iter_batch().count();
        if batch_size == 0 {
            return;
        }

        // Enumerate all eval_infos, which have not yet been assigned and assign them a
        // their guess.
        for (index, eval_info) in self
            .eval_infos
            .evals
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if let EvalInfo::OnBatch(eval_info) = x {
                    Some((i, eval_info.clone()))
                }
                else {
                    None
                }
            })
            // ---
            // todo: we allocate here bc if we borrow above, we cannot borrow mutably in the
            // for loop to set the eval_infos...
            // find a better solution
            .collect_vec()
            .into_iter()
        // ---
        {
            let eval_info = eval_info.borrow();
            let eval_info = eval_info
                .data()
                .as_ref()
                .expect("This should be a leaf and leafes should have data.");

            let eval = Evaluation::Guess(Box::new(Guess {
                relative_to: eval_info.turn,
                quality: {
                    // squish into a range from -1 to +1
                    let w_q = eval_info.q_input.w_q;
                    let b_q = eval_info.q_input.b_q;
                    let d = w_q as i32 - b_q as i32;
                    let m = max(w_q, b_q);
                    let q = if m == 0 { 0. } else { d as f32 / m as f32 };
                    if eval_info.turn == colors::WHITE { q } else { -q }
                },
                policy: eval_info.p_input.p.to_owned(),
            }));

            self.eval_infos.evals[index] = EvalInfo::Evaluated(eval);
        }
    }

    fn iter(&self) -> impl Iterator<Item = &EvalInfo<Self::NodeRef>> {
        self.eval_infos.evals.iter()
    }
}
