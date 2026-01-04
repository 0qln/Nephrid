use approx::assert_relative_eq;
use rand::SeedableRng;

use crate::core::{
    r#move::Move,
    search::mcts::{eval::Policy, node::Branch},
};

use super::*;

#[test]
fn test_dirichlet_noise_basic() {
    let rng = SmallRng::seed_from_u64(42);
    let mut noiser = DirichletNoiser::new(0.03, 0.25, rng);

    let policy = Policy::new(vec![0.7, 0.2, 0.08, 0.02]);
    let mut node = Node::leaf();
    node.set_branches(
        policy
            .iter()
            .map(|p| Branch::new(Move::null(), p))
            .collect_vec(),
    );

    let result = noiser.apply_noise(&mut node);
    assert!(result.is_ok());

    // Policies should be modified
    let new_policy = node.iter_branches().map(|b| b.policy()).collect_vec();
    assert_ne!(new_policy, policy.iter().collect_vec());

    // Policies should still sum to approximately 1
    let sum: f32 = new_policy.iter().sum();
    assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

    // All policies should be non-negative
    assert!(new_policy.iter().all(|&p| p >= 0. && p < 1.));
}
