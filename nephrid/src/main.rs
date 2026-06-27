use engine::{
    core::{Engine, config::Configuration, execute_uci, move_iter::sliding_piece::magics, params::*, zobrist},
    misc::CancellationToken,
};
use std::io::stdin;

mod search {
    use engine::core::{
        params::{self},
        search::{self},
    };

    pub type Worker = cfg_select! {
        any(feature = "mcts-hce", feature = "mcts-pure", feature = "mcts-nn") => {
            search::MctsWorker<MPV, Config, Params>
        },
        feature = "id-hce" => {
            search::IdWorker
        },
    };

    #[allow(dead_code)]
    pub type Params = cfg_select! {
        feature = "mcts-hce"  => params::MctsHceParams,
        feature = "mcts-nn"   => params::MctsNNParams,
        feature = "mcts-pure" => params::MctsPureParams,
        feature = "id-hce"    => params::IdHceParams,
    };

    #[cfg(any(feature = "mcts-hce", feature = "mcts-pure", feature = "mcts-nn"))]
    pub struct Config;

    #[cfg(feature = "mcts-hce")]
    impl search::mcts::MctsConfig for Config {
        type Parts = search::mcts::HceParts;
        type Strat = search::mcts::strategy::MctsUci;
    }

    // todo: this was supposed to be inside the MctsConfig trait, but we get some
    // kind of evaluation overflow error :(
    // todo: tune
    const MPV: usize = cfg_select! {
        feature = "mcts-hce"  => 1,
        feature = "mcts-pure" => 1,
        feature = "id-hce"    => 1,
        all(feature = "mcts-nn", feature = "nn-backend-cuda") => 64,
        all(feature = "mcts-nn", feature = "nn-backend-ndarray") => 1,
    };

    #[cfg(feature = "mcts-pure")]
    impl search::mcts::MctsConfig for Config {
        type Parts = search::mcts::PureParts;
        type Strat = search::mcts::strategy::MctsUci;
    }

    #[cfg(all(feature = "mcts-nn", feature = "nn-backend-cuda"))]
    impl search::mcts::MctsConfig for Config {
        type Parts = search::mcts::NNParts<burn_cuda::Cuda<f32>>;
        type Strat = search::mcts::strategy::MctsUci;
    }

    #[cfg(all(feature = "mcts-nn", feature = "nn-backend-ndarray"))]
    impl search::mcts::MctsConfig for Config {
        type Parts = search::mcts::NNParts<burn::backend::NdArray>;
        type Strat = search::mcts::strategy::MctsUci;
    }
}

fn config() -> Configuration {
    let mut builder = Configuration::builder();

    builder = cfg_select! {
        feature = "mcts-hce" => {{
            mcts_hce_params_default().build_config(builder)
        }},
        feature = "mcts-nn" => {{
            mcts_nn_params_default().build_config(builder)
        }},
        feature = "mcts-pure" => {{
            mcts_pure_params_default().build_config(builder)
        }},
        feature = "id-hce" => {{
            id_hce_params_default().build_config(builder)
        }},
    };

    builder.build()
}

fn main() {
    use search::*;

    magics::init();
    zobrist::init();

    let input_stream = stdin();
    let config = config();
    let mut engine = Engine::new::<Worker>(config);
    let mut ct = CancellationToken::new();

    execute_uci(&mut engine, "ucinewgame", ct.clone()).unwrap();
    execute_uci(&mut engine, "position startpos", ct.clone()).unwrap();

    loop {
        let mut input = String::new();
        match input_stream.read_line(&mut input) {
            Ok(_) => {
                if let Err(e) = execute_uci(&mut engine, input, ct.clone()) {
                    println!("{e}");
                }
            }
            Err(err) => {
                println!("Error: {err}");
            }
        }
        // Replace the cancellation token if it's burned.
        if ct.is_cancelled() {
            ct = Default::default();
        }
    }
}
