extern crate tensorflow;

use std::path::Path;

use tensorflow::{Code, Status};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "nephrid/model.pb";

    if !Path::new(file_name).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python addition.py' to generate {} \
                     and try again.",
                    file_name
                ),
            )
            .unwrap(),
        ));
    }
    
    Ok(())
}