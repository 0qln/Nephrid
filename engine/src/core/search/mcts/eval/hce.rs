use std::{convert::Infallible, ops::Deref};

use crate::{
    core::{
        color::{Perspective, perspectives},
        config::Configuration,
        coordinates::EpTargetSquare,
        depth::Depth,
        eval::{
            self,
            hce::{self, TaperValue, bishop_pair, hygge_king, king_safety, material, mobility, passed_pawns},
        },
        r#move::MAX_LEGAL_MOVES,
        params::MctsHceParamsRef,
        position::{CheckState, Position},
        search::{
            id,
            mcts::{
                eval::{Evaluator, Guess, Logits, Policy, Quality},
                node::{
                    NodeId, Tree,
                    node_state::{HasBranches, Valid},
                },
                search::{BatchItem, Selection},
            },
            ordering::{self},
            quiesce::QSearcher,
            score::{AnyScore, Cp, Score, scores},
            tree::node_types,
        },
        turn::Turn,
    },
    misc::List,
};

use crate::core::{
    color::colors,
    r#move::Move,
    position::{PieceInfo, StateInfo},
    search::mcts::node::node_state::Branching,
};

struct StaticEvaluator;

impl eval::StaticEvaluator for StaticEvaluator {
    fn eval<P: Perspective>(&mut self, pos: &PieceInfo, turn: Turn, ep_sq: EpTargetSquare, phase: TaperValue) -> Score<P> {
        fn static_value<P: Perspective>(pos: &PieceInfo, ep_sq: EpTargetSquare, phase: TaperValue, turn: Turn) -> Score<P> {
            material::<P>(pos)
                + mobility::<P>(pos, phase)
                + hce::psqt::<P>(pos, phase)
                + bishop_pair::<P>(pos)
                + king_safety::<P>(pos, ep_sq, turn, phase)
                + passed_pawns::<P>(pos, ep_sq, turn)
                + hygge_king::<P>(pos, phase)
        }

        let (ep_w, ep_b) = if P::COLOR == colors::WHITE {
            (ep_sq, EpTargetSquare::none())
        }
        else {
            (EpTargetSquare::none(), ep_sq)
        };
        let w_q = static_value::<P>(pos, ep_w, phase, turn);
        let b_q = static_value::<P::Opponent>(pos, ep_b, phase, turn);
        w_q + !b_q
    }

    fn try_from_config<C: Deref<Target = Configuration>>(_: C) -> Result<Self, Infallible> { Ok(Self) }
}

#[derive(Debug, PartialEq, Default)]
pub struct PolicyInput;

impl PolicyInput {
    pub fn meta(_pos: &PieceInfo, _mov: Move, _state: &StateInfo) -> i32 {
        // todo: give bonus for promotions etc.
        0
    }

    pub fn check_bonus(phase: TaperValue, pos: &PieceInfo, turn: Turn, mov: Move) -> AnyScore {
        let check = pos.does_check(turn, mov);
        let score = match check {
            CheckState::None => AnyScore::new(0),
            CheckState::Single => AnyScore::new(50),
            CheckState::Double => AnyScore::new(100),
        };
        phase.weighted_eval(scores::ZERO, score)
    }
}

pub struct EvalInfo<Moves: AsRef<[Move]>> {
    /// The to-be-evaluated that this eval info is for.
    moves: Moves,

    /// Turn of the current player.
    turn: Turn,

    /// The current game phase taper value.
    phase: TaperValue,

    /// Piece infos of the position.
    pos: PieceInfo,

    /// Current position state info.
    state: StateInfo,

    /// State info of the position after quieting it.
    quality: Cp,

    // tunables
    params: MctsHceParamsRef,
}

impl<Moves: AsRef<[Move]>> EvalInfo<Moves> {
    pub fn new(moves: Moves, pos: &mut Position, params: MctsHceParamsRef) -> Self {
        let phase = TaperValue::from_position(pos.piece_info());

        // todo: store tt and ss somewhere
        let mut tt = id::TT::new(1);

        let mut ss = id::SS::from(vec![id::SearchEntry { phase, ..Default::default() }]);

        let mut qsearcher = QSearcher::new(pos, &mut tt, &mut ss, pos.ply());

        let quality: Cp = match pos.get_turn().v() {
            colors::WHITE_C => qsearcher
                .go::<perspectives::White, node_types::Normal>(
                    pos,
                    Score::NEG_INF,
                    Score::POS_INF,
                    MctsHceParamsRef::clone(&params),
                    &mut StaticEvaluator,
                    Depth::new(30),
                )
                .into(),
            colors::BLACK_C => qsearcher
                .go::<perspectives::Black, node_types::Normal>(
                    pos,
                    Score::NEG_INF,
                    Score::POS_INF,
                    MctsHceParamsRef::clone(&params),
                    &mut StaticEvaluator,
                    Depth::new(30),
                )
                .into(),
            _ => unreachable!(),
        };
        Self {
            quality,
            pos: pos.piece_info().clone(),
            state: pos.state_info().clone(),
            phase,
            moves,
            turn: pos.get_turn(),
            params,
        }
    }

    /// Convert QualityInput into Quality, where the Quality is relative to
    /// white.
    pub fn quality(&self) -> Quality { Quality::from(self.quality) }

    pub fn policy(&self, buf: &mut List<218 /* inlined MAX_LEGAL_MOVES */, f32>) -> Policy {
        let pos = &self.pos;
        let phase = self.phase;
        let state = &self.state;
        let color = self.turn;

        let mut logits = List::new();

        for &mov in self.moves.as_ref().iter() {
            // policy for each move in the position is the difference of the psqt score from
            // the previous position and the psqt score of the next position that is
            // achieved by the move.
            let from = mov.get_from();
            let to = mov.get_to();
            let piece = pos.get_piece(from).piece_type();
            let score = AnyScore::from(ordering::psqt(phase, piece, from, to, mov.get_flag(), color))
                + ordering::see(pos, mov, color)
                + PolicyInput::check_bonus(phase, pos, color, mov)
                + PolicyInput::meta(pos, mov, state);

            logits.push(score.v() as f32);
        }

        Policy::from_logits(Logits(logits), self.params.policy_temperature(), buf)
    }
}

pub const trait PolicyParams {
    fn policy_temperature(&self) -> f32;
}

#[derive(Debug, Clone)]
pub struct HceEvaluator {
    policy_buf: Box<List<{ MAX_LEGAL_MOVES }, f32>>,
    params: MctsHceParamsRef,
}

impl HceEvaluator {
    pub fn new(params: MctsHceParamsRef) -> Self {
        Self {
            policy_buf: Box::new(List::new()),
            params,
        }
    }
}

impl Evaluator for HceEvaluator {
    type TraceData = Option<EvalInfo<Vec<Move>>>;

    fn trace<S: const Valid + HasBranches>(&self, node: NodeId<S>, tree: &Tree, pos: &mut Position) -> Self::TraceData {
        node.try_into::<Branching>().map(|node| {
            EvalInfo::new(
                tree.branches(node).iter().map(|b| b.mov()).collect(),
                pos,
                MctsHceParamsRef::clone(&self.params),
            )
        })
    }

    fn eval_batch(
        &mut self,
        _tree: &Tree,
        _selection: &Selection<Self::TraceData>,
        leafs: &[&BatchItem<Self::TraceData>],
    ) -> impl Iterator<Item = Guess> {
        leafs.iter().filter_map(|&leaf| {
            let eval_info = leaf.trace.as_ref()?;
            Some(Guess {
                relative_to: colors::WHITE,
                quality: eval_info.quality(),
                policy: eval_info.policy(&mut self.policy_buf),
            })
        })
    }
}
