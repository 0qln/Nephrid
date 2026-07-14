#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Root,
    Pv,
    Cut,
    All,
}

pub const trait NodeType {
    const KIND: NodeKind;
}

pub mod node_types {
    use super::{NodeKind, NodeType};

    pub struct Root;
    impl const NodeType for Root {
        const KIND: NodeKind = NodeKind::Root;
    }

    pub struct Pv;
    impl const NodeType for Pv {
        const KIND: NodeKind = NodeKind::Pv;
    }

    pub struct Cut;
    impl const NodeType for Cut {
        const KIND: NodeKind = NodeKind::Cut;
    }

    pub struct All;
    impl const NodeType for All {
        const KIND: NodeKind = NodeKind::All;
    }
}
