# Nephrid
 
An experimental UCI-Compatible MCTS chess engine. 

live on lichess: https://lichess.org/@/nephrid


## Search

### MCTS

#### HCE (Hand-crafted Evaluation)

HCE version:
```
cargo build --release --bin nephrid --features "mcts-hce" --no-default-features
```

#### NN (Neural network Evaluation)

NN evaluation:
```
cargo build --release --bin nephrid --features "mcts-nn" --no-default-features
```

#### Pure

Pure MCTS evaluation, via rollouts at leaf nodes:
```
cargo build --release --bin nephrid --features "mcts-pure" --no-default-features
```

### ID (Iterative deepening)

#### HCE (Hand-crafted Evaluation)

HCE version:
```
cargo build --release --bin nephrid --features "id-hce" --no-default-features
```
