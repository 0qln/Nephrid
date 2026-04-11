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
    type Score = super::Score;

    fn score(&self, node: &NodeData, _branch: &Branch, cap_n_i: u32) -> Score {
        match node.visits() {
            0 => Score(f32::INFINITY),
            n_i => {
                let w_i = node.value();
                let n_i = n_i as f32;
                let exploitation = w_i / n_i;
                let exploration = self.c * f32::sqrt((cap_n_i as f32).ln() / n_i);
                Score(exploitation + exploration)
            }
        }
    }

    fn min_score(&self) -> Self::Score {
        Score(f32::NEG_INFINITY)
    }
}
