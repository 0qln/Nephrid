use super::*;

pub struct UcbSelector {
    c: f32,
}

impl UcbSelector {
    pub fn new(c: f32) -> Self {
        Self { c }
    }
}

impl Default for UcbSelector {
    fn default() -> Self {
        Self { c: f32::sqrt(2.0) }
    }
}

impl Selector for UcbSelector {
    type Score = Score;

    fn score(&self, branch: &Branch, cap_n_i: u32) -> Score {
        match branch.visits() {
            0 => Score(f32::INFINITY),
            n_i => {
                let w_i = branch.value();
                let n_i = n_i as f32;
                let exploitation = w_i / n_i;
                let exploration = self.c * f32::sqrt((cap_n_i as f32).ln() / n_i);
                Score(exploitation + exploration)
            }
        }
    }
}

#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Score(pub f32);

impl_op!(-|x: Score| -> Score { Score(-x.0) });

impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("This shouldn't happen for ucb scores.")
    }
}
