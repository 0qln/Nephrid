// pub trait Worker {
//     fn process(ct: CancellationToken);

//     fn start(ct: CancellationToken) {
//         while !ct.is_cancelled() {
//             callback(ct.clone());
//         }
//     }
// }

// pub struct SelectionWorker {

// }

// pub struct EvaluationWorker {
//     queue: Vec<>;
// }

// impl Worker for EvaluationWorker {
//     fn process(ct: CancellationToken) {

//     }
// }
