pub trait Backpropagater {}

pub struct BackupInfo {
    /// whose turn it is at the node
    turn: Turn,
}

pub type BackupNode = DoubleLinkedNode<BackupInfo>;
