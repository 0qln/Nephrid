pub trait Evaluator<const X: usize> {
    /// prepare the newest position that needs to be evaluated.
    // fn push(&mut self, pos: &Position, x: usize) -> ();

    /// pop the latest position that was prepared.
    // fn pop(&mut self, x: usize) -> ();

    /// clear all prepaered
    // fn clear(&mut self) -> ();

    /// Returns: (quality [-1;1], policy [over ALL moves])
    // fn evaluate(&self) -> [(f32, [f32; POLICY_OUTPUTS]); X];
}

pub struct InputFloats(BoardInputFloats, StateInputFloats);

pub struct EvalInfo {
    /// input floats for the eval model
    inputs: InputFloats,

    /// whose turn it is at the node
    turn: Turn,
}

pub type EvalInfoNode = DoubleLinkedNode<Option<EvalInfo>>;

// type BoardHistoryBuffer = StaticRb<BoardInputFloats, BOARD_INPUT_HISTORY>;

/// X: batch size
pub struct NNEvaluator<B: Backend, const X: usize> {
    /// Eval info
    eval_info_root: EvalInfoNode,

    // nn model
    model: Model<B>,

    /// shut up the compiler, i will probably use X later on...
    _x: [PhatomData; X]
}

// impl<B: Backend, const X: usize> EvalModel<B, X> {
//     pub fn new(model: Model<B>, device: &B::Device) -> Self {
//         Self {
//             board_history: vec![ (); BOARD_INPUT_HISTORY ],
//             model,
//         }
//     }
// }

impl<B: Backend, const X: usize> Evaluator<X> for EvalModel<B, X> {
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
