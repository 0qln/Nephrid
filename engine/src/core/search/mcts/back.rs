use super::utils::*;
use crate::core::turn::Turn;

pub trait Backpropagater {}

pub struct BackupInfo {
    /// whose turn it is at the node
    turn: Turn,
}

pub type BackupNode = DoubleLinkedNode<BackupInfo>;

pub struct DefaultBackuper {}
