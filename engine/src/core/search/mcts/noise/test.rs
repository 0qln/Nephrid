use approx::assert_relative_eq;
use rand::{RngCore, SeedableRng};

use crate::core::{
    depth::Depth,
    position::Position,
    search::mcts::{
        eval::Policy,
        node::node_state::{Branching, Leaf},
    },
};

use super::*;

#[test]
fn test_dirichlet_noise_basic() {
    let mut rng = SmallRng::seed_from_u64(42);

    let mut tree = Tree::new();

    let pos = Position::start_position();
    let _node = tree.expand_node(
        tree.node_switch(tree.root()).get::<Leaf>().unwrap(),
        &pos,
        Depth::ROOT,
    );
    let node = tree.node_switch(tree.root()).get::<Branching>().unwrap();

    let policy = {
        let branches = tree.branches(node);
        Policy::from_logits(branches.iter().map(|_| rng.next_u32() as f32).collect_vec())
    };
    let node = tree.set_policy(node.clone(), &policy);

    let mut noiser = DirichletNoiser::new(0.03, 0.25, rng);
    let result = noiser.apply_noise(node.clone(), &mut tree);

    assert!(result.is_ok());

    // Policies should be modified
    let new_policy = tree.branches(node).iter().map(|b| b.policy()).collect_vec();
    assert_ne!(new_policy, policy.iter().collect_vec());

    // Policies should still sum to approximately 1
    let sum: f32 = new_policy.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

    // All policies should be non-negative
    assert!(new_policy.iter().all(|&p| (0. ..1.).contains(&p)));
}
