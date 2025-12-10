use crate::core::search::mcts::eval::model::BoardInputTensor;
use crate::core::search::mcts::eval::model::board_input;
use burn::tensor::backend::Backend;
use itertools::Itertools;
use rand::rngs::SmallRng;
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
