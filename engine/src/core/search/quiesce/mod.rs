use crate::core::{
    color::{Color, Perspective},
    depth::Depth,
    eval::{
        StaticEvaluator,
        hce::{TaperValue, piece_score},
    },
    r#move::Move,
    move_iter::{
        self,
        opt::{self},
    },
    piece::{PromoPieceType, piece_type},
    position::{CheckState, PieceInfo, Position},
    search::{
        ordering::{self, MovePicker},
        score::Score,
    },
};

pub trait QSearchParams {
    fn futility_margin(&self) -> i32;
    fn delta_pruning_threshold(&self) -> TaperValue;
}

/// # Q-Search
///
/// Make the position quiet.
///
/// [q-search](https://www.chessprogramming.org/Quiescence_Search)
pub fn qsearch<S: StaticEvaluator, P: Perspective, X: QSearchParams + Clone>(
    pos: &mut Position,
    mut alpha: Score<P>,
    beta: Score<P>,
    params: X,
    static_evaluator: &S,
    depth: Depth,
) -> Score<P> {
    let in_check = pos.get_check_state() != CheckState::None;

    let mut best_value = Score::NEG_INF;

    let stm = P::COLOR;
    let piece_info = pos.piece_info();
    let phase = TaperValue::from_position(piece_info);
    let static_eval = || static_evaluator.eval(piece_info, stm, pos.get_ep_target_square(), phase);

    if depth == Depth::new(0) {
        return static_eval();
    }

    // stand pad if not in check
    if !in_check {
        best_value = static_eval();

        if best_value >= beta {
            return best_value;
        }
        if best_value > alpha {
            alpha = best_value;
        }
    }

    // move gen
    struct MoveScorer<'a> {
        pieces: &'a PieceInfo,
        color: Color,
        phase: TaperValue,
    }
    impl ordering::MoveScorer for MoveScorer<'_> {
        fn score(&self, mov: Move) -> i32 {
            let (from, to, _) = mov.into();
            let piece = self.pieces.get_piece(from);

            ordering::see(self.pieces, mov, self.color)
                + ordering::psqt(self.phase, piece.piece_type(), from, to, self.color)
        }
    }

    let move_scorer = MoveScorer {
        pieces: pos.piece_info(),
        color: P::COLOR,
        phase,
    };

    // if in check, we need to generate all moves, otherwise we can skip quiets and
    // promos
    let mut move_picker = if in_check {
        MovePicker::from_position::<opt::All, _>(pos, move_scorer)
    }
    else {
        struct MoveGenOpt;
        impl const move_iter::Options for MoveGenOpt {
            #[inline(always)]
            fn gen_quiets() -> bool {
                false
            }

            #[inline(always)]
            fn gen_promos() -> bool {
                false
            }

            #[inline(always)]
            fn legal() -> bool {
                true
            }
        }

        MovePicker::from_position::<MoveGenOpt, _>(pos, move_scorer)
    };

    // recurse
    while let Some((m, _)) = move_picker.next() {
        // if !pos.is_legal_for::<P>(m) {
        //     continue;
        // }

        // delta pruning
        if !in_check && phase < params.delta_pruning_threshold() {
            let value_bonus = PromoPieceType::try_from(m.get_flag())
                .ok()
                .map(|promo| piece_score(promo.into()) - piece_score(piece_type::PAWN))
                .unwrap_or(0);

            let captured_value = m
                .get_capture_sq()
                .map(|capt_sq| piece_score(pos.get_piece(capt_sq).piece_type()))
                .unwrap_or(0);

            let futility_margin = params.futility_margin();
            let futility_score = captured_value + value_bonus + futility_margin;

            if best_value + Score::new(futility_score) < alpha {
                continue;
            }
        }

        pos.make_move_for::<P>(m);

        let score = !qsearch(
            pos,
            !beta,
            !alpha,
            params.clone(),
            static_evaluator,
            depth - 1,
        );

        pos.unmake_move_for::<P>(m);

        if score >= beta {
            return score;
        }
        if score > best_value {
            best_value = score;
        }
        if score > alpha {
            alpha = score;
        }
    }

    best_value
}
