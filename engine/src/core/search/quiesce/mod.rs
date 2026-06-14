use crate::core::{
    color::{Color, Perspective},
    depth::Depth,
    eval::{
        StaticEvaluator,
        hce::{TaperValue, piece_score},
    },
    r#move::Move,
    piece::{PromoPieceType, piece_type},
    position::{CheckState, Position},
    search::{
        id::RbSet,
        ordering::{self, MovePicker, MoveScore, RtStage, Stage},
        score::Score,
    },
};

pub const trait QSearchParams {
    fn futility_margin(&self) -> i32;
    fn delta_pruning_threshold(&self) -> TaperValue;
}

/// # Q-Search
///
/// Make the position quiet.
///
/// [q-search](https://www.chessprogramming.org/Quiescence_Search)
pub fn qsearch<P: Perspective>(
    pos: &mut Position,
    mut alpha: Score<P>,
    beta: Score<P>,
    params: impl QSearchParams + Clone,
    static_evaluator: &impl StaticEvaluator,
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
    let scorer = MoveScorer { color: P::COLOR, phase };
    let mut move_picker = MovePicker::new_with_max_stage(
        Move::null(),
        RbSet::<Move, 2>::default(),
        if in_check {
            RtStage::Done
        }
        else {
            RtStage::YieldBadCaptures
        },
    );

    // recurse
    while let Some(m) = move_picker.next_for::<P>(pos, &scorer) {
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

struct MoveScorer {
    color: Color,
    phase: TaperValue,
}
impl ordering::MoveScorer for MoveScorer {
    fn score<S: Stage>(&self, pos: &Position, mov: Move) -> MoveScore {
        match S::stage() {
            ordering::RtStage::YieldHashMove => {
                todo!("we don't yet have a hashmove in qsearch")
            }
            ordering::RtStage::GenerateCapturesAndPromos
            | ordering::RtStage::YieldGoodCapturesAndPromos
            | ordering::RtStage::YieldBadCaptures => {
                let pieces = pos.piece_info();

                let (from, to, _) = mov.into();
                let piece = pieces.get_piece(from);

                ordering::see(pieces, mov, self.color)
                    + ordering::psqt(
                        self.phase,
                        piece.piece_type(),
                        from,
                        to,
                        mov.get_flag(),
                        self.color,
                    )
            }
            ordering::RtStage::YieldKillers => todo!("we don't yet have killers in qsearch"),
            ordering::RtStage::GenerateQuiets | ordering::RtStage::YieldQuiets => {
                debug_assert!(
                    pos.get_check_state() != CheckState::None,
                    "we should never be generating quiets in qsearch if we're not in check"
                );
                let pieces = pos.piece_info();
                let (from, to, _) = mov.into();
                let piece = pieces.get_piece(from);
                ordering::psqt(
                    self.phase,
                    piece.piece_type(),
                    from,
                    to,
                    mov.get_flag(),
                    self.color,
                )
            }
            ordering::RtStage::Done => todo!("why are we scoring Done??"),
        }
    }
}
