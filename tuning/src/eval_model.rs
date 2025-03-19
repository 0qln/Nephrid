use engine::{core::{execute_uci, Engine}, uci::sync::CancellationToken};
use tensorflow::{Code, Graph, SavedModelBundle, SessionOptions, Status};

extern crate tensorflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Tensorflow version = {}", tensorflow::version().unwrap());
    println!("{}", tensorflow::Device)

    // Load the model
    let mut graph = Graph::new();
    let file_name = "engine/src/core/search/mcts/eval/model.pb";
    let bundle = SavedModelBundle::load(
        &SessionOptions::new(), 
        &["serve"], 
        &mut graph, 
        file_name
    )?;
    let session = &bundle.session;
    let train = &bundle.meta_graph_def().get_signature("train")?;
    let evalutate = &bundle.meta_graph_def().get_signature("evaluate")?;
    
    let cmd_cancellation = CancellationToken::new();
    let mut engine = Engine::default();
    
    execute_uci(&mut engine, "ucinewgame".to_string(), cmd_cancellation.clone())?;

    Ok(())
}
