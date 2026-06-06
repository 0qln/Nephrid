use std::{cmp::Reverse, ops::ControlFlow};

use crate::{
    core::{
        color::Perspective,
        eval::{
            StaticEvaluator,
            hce::{TaperValue, piece_score},
        },
        r#move::{MAX_LEGAL_MOVES, Move},
        move_iter::{self, fold_legal_moves, fold_legals},
        piece::{PromoPieceType, piece_type},
        position::{CheckState, Position},
        search::{ordering, score::Score},
    },
    misc::List,
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
) -> Score<P> {
    let in_check = pos.get_check_state() != CheckState::None;

    let mut best_value = Score::NEG_INF;

    let stm = P::COLOR;
    let piece_info = pos.piece_info();
    let phase = TaperValue::from_position(piece_info);

    // stand pad if not in check
    if !in_check {
        let static_eval = static_evaluator.eval(piece_info, stm, pos.get_ep_target_square(), phase);

        best_value = static_eval;

        if best_value >= beta {
            return best_value;
        }
        if best_value > alpha {
            alpha = best_value;
        }
    }

    // move gen
    let mut move_list = List::<{ MAX_LEGAL_MOVES }, (Move, i32)>::new();
    if in_check {
        _ = fold_legal_moves::<_, _, _>(pos, (), |_, m| {
            move_list.push((m, 0));
            ControlFlow::Continue::<(), ()>(())
        });
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
        }

        _ = fold_legals::<MoveGenOpt, _, _, _>(pos, (), |_, m| {
            move_list.push((m, 0));
            ControlFlow::Continue::<(), ()>(())
        });
    };

    /*\                                             /*\
    |*|---------------------------------------------|*|
    |*| generate the see score outside of the move  |*|
    |*| generation and the sorting, such that it    |*|
    |*| isn't computed for each comparison and when |*|
    |*| don't break cache locality.                 |*|
    |*|---------------------------------------------|*|
    \*/                                             \*/
    for &mut (m, ref mut score) in move_list.as_mut_slice() {
        let (from, to, _) = m.into();
        let piece = pos.get_piece(from);

        // todo: use a dedicated MovePicker instead of this ...
        *score = ordering::see(pos.piece_info(), m, P::COLOR)
            + ordering::psqt(phase, piece.piece_type(), from, to, stm);
    }

    // move ordering
    move_list
        .as_mut_slice()
        .sort_unstable_by_key(|&(_, score)| Reverse(score));

    // recurse
    for &(m, _) in move_list.iter() {
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

        let score = !qsearch(pos, !beta, !alpha, params.clone(), static_evaluator);

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
