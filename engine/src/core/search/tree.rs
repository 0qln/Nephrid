#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Root,
    Normal,
    Cut,
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

    pub struct Normal;
    impl const NodeType for Normal {
        const KIND: NodeKind = NodeKind::Normal;
    }

    pub struct Cut;
    impl const NodeType for Cut {
        const KIND: NodeKind = NodeKind::Cut;
    }
}
