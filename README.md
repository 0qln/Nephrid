# Nephrid
 
An experimental UCI-Compatible MCTS chess engine. 


## 

### MCTS

#### HCE (Hand-crafted Evaluation)

HCE version can be compiled with 
```
cargo build --release --bin nephrid --features "mcts-hce" --no-default-features
```

#### NN (Neural network Evaluation)

NN evaluation can be compiled with
```
cargo build --release --bin nephrid --features "mcts-nn" --no-default-features
```
