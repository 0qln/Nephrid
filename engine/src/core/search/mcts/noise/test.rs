use approx::assert_relative_eq;
use rand::{RngCore, SeedableRng};

use crate::core::{
    depth::Depth,
    position::Position,
    search::mcts::{
        eval::Policy,
        node::{Node, RtNodeRef},
    },
};

use super::*;

#[test]
fn test_dirichlet_noise_basic() {
    let mut rng = SmallRng::seed_from_u64(42);

    let node = CtNodeRef::new(Node::new_leaf());
    let mut tree = Tree::new(RtNodeRef::from_ct(node.clone()));

    let pos = Position::start_position();
    let node = tree.expand_node(node.clone(), &pos, Depth::ROOT);
    let node = node.branching().expect("startpos should have branches");

    let policy = {
        let node = node.borrow();
        let branches = node.branches();
        Policy::new(branches.iter().map(|_| rng.next_u32() as f32).collect_vec())
    };
    let node = tree.set_policy(node.clone(), &policy);

    let mut noiser = DirichletNoiser::new(0.03, 0.25, rng);
    let result = noiser.apply_noise(node.clone(), &mut tree);

    assert!(result.is_ok());

    // Policies should be modified
    let new_policy = node
        .borrow()
        .branches()
        .iter()
        .map(|b| b.policy())
        .collect_vec();
    assert_ne!(new_policy, policy.iter().collect_vec());

    // Policies should still sum to approximately 1
    let sum: f32 = new_policy.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

    // All policies should be non-negative
    assert!(new_policy.iter().all(|&p| p >= 0. && p < 1.));
}
