use core::error;
use std::{
    fmt::Debug, io::{
        stdout, 
        Write,
    }, sync::{
        atomic::{
            AtomicBool, 
            Ordering}, 
        Arc,
    }
};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum UciError {
    #[error("Expected value for {0}")]
    MissingArgument(&'static str),
    
    #[error("Value {0} is out of range: [{1}; {2}]")]
    InputOutOfRange(String, String, String),
    
    #[error("Invalid value {0}, expected one of: {1:?}")]
    InvalidValue(String, Vec<String>),
    
    #[error("Invalid command: {0}")]
    InvalidCommand(String),

    #[error("Unknown option")]
    UnknownOption,
}

pub fn out(msg: &str) {
    let stdout = stdout();
    let _ = writeln!(&mut stdout.lock(), "{}", msg);
}


#[derive(Default, Clone)]
pub struct CancellationToken { v: Arc<AtomicBool> }

impl CancellationToken {
    pub fn new() -> Self {
        Self { v: Arc::new(AtomicBool::new(false)) }
    }

    pub fn cancel(&self) {
        self.v.store(true, Ordering::Relaxed)
    }    
    
    pub fn is_cancelled(&self) -> bool {
        self.v.load(Ordering::Relaxed)
    }
}
