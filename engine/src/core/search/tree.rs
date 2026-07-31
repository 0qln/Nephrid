/// ref: https://www.chessprogramming.org/Node_Types
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Root,
    Pv,
    Cut,
    All,
}

pub const trait NodeType {
    type Next: NodeType;
    const KIND: NodeKind;
}

pub mod node_types {
    use super::{NodeKind, NodeType};

    pub struct Root;
    const impl NodeType for Root {
        type Next = Pv;
        const KIND: NodeKind = NodeKind::Root;
    }

    pub struct Pv;
    const impl NodeType for Pv {
        type Next = Cut;
        const KIND: NodeKind = NodeKind::Pv;
    }

    pub struct Cut;
    const impl NodeType for Cut {
        type Next = All;
        const KIND: NodeKind = NodeKind::Cut;
    }

    pub struct All;
    const impl NodeType for All {
        type Next = Cut;
        const KIND: NodeKind = NodeKind::All;
    }
}
