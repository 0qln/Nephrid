use std::{
    sync::{
        atomic::{
            AtomicBool, 
            Ordering}, 
        Arc,
    },
    io::{
        stdout, 
        Write,
    }
};

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
