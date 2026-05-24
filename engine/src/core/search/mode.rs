#[derive(Debug, Default, Clone)]
pub enum Mode {
    #[default]
    Normal,
    Ponder,
    Perft {
        captures_only: bool,
    },
}
