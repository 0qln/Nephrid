use engine::{
    core::{Engine, execute_uci, move_iter::sliding_piece::magics, zobrist},
    misc::CancellationToken,
};
use std::io::stdin;

mod search {
    use engine::core::search::{
        MctsWorker,
        mcts::{self, MctsConfig, strategy::MctsUci},
    };

    pub struct Config;

    // todo: mcts/id (iterative deepening) feature gate
    pub type Worker = MctsWorker<MPV, Config>;

    #[cfg(feature = "mcts-hce")]
    impl MctsConfig for Config {
        type Parts = mcts::HceParts;
        type Strat = MctsUci;
    }

    // todo: this was supposed to be inside the MctsConfig trait, but we get some
    // kind of evaluation overflow error :(
    #[cfg(feature = "mcts-hce")]
    const MPV: usize = 1; //todo tune

    #[cfg(feature = "mcts-pure")]
    impl MctsConfig for Config {
        type Parts = mcts::PureParts;
        type Strat = MctsUci;
    }

    #[cfg(feature = "mcts-pure")]
    const MPV: usize = 1; //todo tune

    #[cfg(all(feature = "mcts-nn", feature = "nn-backend-cuda"))]
    impl MctsConfig for Config {
        const MPV: usize = 64;
        type Parts = mcts::NNParts<burn_cuda::Cuda<f32>>;
        type Strat = MctsUci;
    }

    #[cfg(all(feature = "mcts-nn", feature = "nn-backend-cuda"))]
    const MPV: usize = 64; //todo tune

    #[cfg(all(feature = "mcts-nn", feature = "nn-backend-ndarray"))]
    impl MctsConfig for Config {
        type Parts = mcts::NNParts<burn::backend::NdArray>;
        type Strat = MctsUci;
    }

    #[cfg(all(feature = "mcts-nn", feature = "nn-backend-ndarray"))]
    const MPV: usize = 1; //todo tune
}

fn main() {
    magics::init();
    zobrist::init();

    let input_stream = stdin();
    let mut engine = Engine::new::<search::Worker>();
    let mut cmd_cancellation = CancellationToken::new();

    execute_uci(
        &mut engine,
        "ucinewgame".to_string(),
        cmd_cancellation.clone(),
    )
    .unwrap();

    execute_uci(
        &mut engine,
        "position startpos".to_string(),
        cmd_cancellation.clone(),
    )
    .unwrap();

    loop {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => {
                if let Err(e) = execute_uci(&mut engine, input, cmd_cancellation.clone()) {
                    println!("{e}");
                }
            }
            Err(err) => {
                println!("Error: {err}");
            }
        }
        // Replace the cancellation token if it's burned.
        if cmd_cancellation.is_cancelled() {
            cmd_cancellation = Default::default();
        }
    }
}
