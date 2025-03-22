use engine::{core::{execute_uci, Engine}, uci::sync::CancellationToken};
use itertools::Itertools;
use tensorflow::{Code, Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Status, Tensor};

extern crate tensorflow;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Tensorflow version = {}", tensorflow::version().unwrap());

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
    // let train = bundle.meta_graph_def().get_signature("train")?;
    let evaluate = bundle.meta_graph_def().get_signature("evaluate")?;
    
    let board_input_info = evaluate.get_input("board_input")?;
    let state_input_info = evaluate.get_input("state_input")?;
    let value_output_info = evaluate.get_output("value_output")?;
    let policy_output_info = evaluate.get_output("policy_output")?;
    
    println!("{:?}", board_input_info);
    println!("{:?}", state_input_info);
    println!("{:?}", value_output_info);
    println!("{:?}", policy_output_info);
    
    let board_input_shape = Option::<Vec<Option<i64>>>::from(board_input_info.shape().clone())
        .expect("Could not get board input shape")
        .iter().map(|o| o.unwrap_or(1) as u64)
        .collect_vec();
    let state_input_shape = Option::<Vec<Option<i64>>>::from(state_input_info.shape().clone())
        .expect("Could not get state input shape")
        .iter().map(|o| o.unwrap_or(1) as u64)
        .collect_vec();
    
    let board_input_tensor = Tensor::<f32>::new(&board_input_shape);
    let state_input_tensor = Tensor::<f32>::new(&state_input_shape);
    
    let board_input_name = board_input_info.name().name.clone();
    let state_input_name = state_input_info.name().name.clone();
    let value_output_name = value_output_info.name().name.clone();
    let policy_output_name = policy_output_info.name().name.clone();
    
    let board_input_op = graph.operation_by_name_required(&board_input_name)?;
    let state_input_op = graph.operation_by_name_required(&state_input_name)?;
    let value_output_op = graph.operation_by_name_required(&value_output_name)?;
    let policy_output_op = graph.operation_by_name_required(&policy_output_name)?;
    
    let mut args = SessionRunArgs::new();
    args.add_feed(&board_input_op, board_input_info.name().index, &board_input_tensor);
    args.add_feed(&state_input_op, state_input_info.name().index, &state_input_tensor);
    
    let value_output_fetch = args.request_fetch(&value_output_op, value_output_info.name().index);
    let policy_output_fetch = args.request_fetch(&policy_output_op, policy_output_info.name().index);
    
    session.run(&mut args)?;
    
    let value_output_result = args.fetch::<f32>(value_output_fetch)?[0];
    let policy_output_result = args.fetch::<f32>(policy_output_fetch)?[0];
    
    println!("Value output: {}", value_output_result);
    println!("Policy output: {}", policy_output_result);
    
    let cmd_cancellation = CancellationToken::new();
    let mut engine = Engine::default();
    
    execute_uci(&mut engine, "ucinewgame".to_string(), cmd_cancellation.clone())?;
    execute_uci(&mut engine, "position startpos".to_string(), cmd_cancellation.clone())?;

    Ok(())
}
