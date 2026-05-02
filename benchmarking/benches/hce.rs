use criterion::{Criterion, criterion_group, criterion_main};
use engine::{core::{
    depth::Depth, r#move::MAX_LEGAL_MOVES, position::Position, search::mcts::{
        eval::hce::EvalInfo,
        node::{
            Tree,
            node_state::{Branching, Leaf},
        },
    }
}, misc::List};

pub fn policy(c: &mut Criterion) {
    let mut pos = Position::start_position();
    let mut tree = Tree::new();
    let root = tree.root();

    let root = tree
        .expand_node(
            tree.node_switch(root).get::<Leaf>().unwrap(),
            &pos,
            Depth::ROOT,
        )
        .get::<Branching>()
        .unwrap();

    let eval = EvalInfo::new(root, &tree, &mut pos);

    let mut buf = List::<{ MAX_LEGAL_MOVES }, f32>::new();

    c.bench_function("policy", |b| b.iter(|| eval.policy(&mut buf)));
}

criterion_group!(benches, policy);
criterion_main!(benches);
