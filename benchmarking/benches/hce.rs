use engine::core::{move_iter::sliding_piece::magics, search::mcts::eval::Ratio, zobrist};
use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use engine::{
    core::{
        depth::Depth,
        r#move::MAX_LEGAL_MOVES,
        position::Position,
        search::mcts::{
            HceParts, MctsParts,
            eval::{Probability, hce::EvalInfo},
            node::{
                Height, Tree, VisitCount,
                node_state::{Branching, Evaluated, Leaf},
            },
            search::TreeSearcher,
            select::{Selector, puct::PuctSelector},
        },
    },
    misc::List,
};

pub fn policy(c: &mut Criterion) {
    magics::init();
    zobrist::init();

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

pub fn puct(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    // very small function so make sure we get a lot of samples
    let mut group = c.benchmark_group("puct");
    group.measurement_time(Duration::from_secs(20));

    let mut pos = Position::start_position();
    let mut tree = Tree::new();
    let parts = HceParts::default();
    let mut searcher = TreeSearcher::<1, _, _, _>::new(
        &mut pos,
        PuctSelector::default(),
        parts.evaluator(),
        parts.noiser(),
    );

    // initialize root node so we can run puct on the children
    searcher.init_root(&mut tree);

    let root_id = tree.try_node::<Evaluated>(tree.root()).unwrap().id();
    let branch = tree.branch_ids(root_id).next().unwrap();
    let sel = PuctSelector::default();

    group.bench_function("puct::zero_visits", |b| {
        b.iter(|| sel.score(black_box(&tree), black_box(branch), black_box(root_id)))
    });

    // make sure root node and the layer below is expanded
    while tree.compute_minheight() <= Height(2) {
        searcher.grow(&mut tree);
    }
    assert!(tree.node(tree.branch(branch).node()).visits() > VisitCount(0));

    group.bench_function("puct::some_visits", |b| {
        b.iter(|| sel.score(black_box(&tree), black_box(branch), black_box(root_id)))
    });
}

pub fn prob_mix(c: &mut Criterion) {
    magics::init();
    zobrist::init();

    let mut group = c.benchmark_group("puct");
    group.measurement_time(Duration::from_secs(20));

    let p1 = Probability::new(0.71_f32);
    let p2 = Probability::new(0.39_f32);
    let ratio = Ratio::new(0.4_f32);
    group.bench_function("prob_mix", |b| {
        b.iter(|| Probability::mix(black_box(&mut p1.clone()), black_box(p2), black_box(ratio)))
    });
}

criterion_group!(benches, policy, puct, prob_mix);
criterion_main!(benches);
