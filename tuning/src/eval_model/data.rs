use std::{
    env::var,
    fs,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};

use burn::{
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::{Dataset, InMemDataset},
    },
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use engine::{
    core::{
        r#move::Move,
        position::{FenExport, FenImport, Position},
        search::mcts::nn::{BoardInputFloats, StateInputFloats},
    },
    uci::tokens::Tokenizer,
};
use itertools::Itertools;

use crate::TrainingConfig;

#[derive(Clone, Debug)]
pub struct BoardInput(pub Vec<BoardInputFloats>);

#[derive(Clone, Debug)]
pub struct StateInput(pub StateInputFloats);

#[derive(Clone, Default, Debug)]
pub struct FenItemRaw {
    /// The position string that is to be played out
    pub fen: String,
}

impl FenItemRaw {
    pub fn new(fen: String) -> Self {
        Self { fen }
    }
}

pub struct FenDataset {
    dataset: InMemDataset<FenItemRaw>,
}

impl Dataset<FenItemRaw> for FenDataset {
    fn get(&self, index: usize) -> Option<FenItemRaw> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl FenDataset {
    pub fn from_epd(path: &str, split: &str, num_fens_total: usize) -> Self {
        let root = FenDataset::load_path(path, split);
        let fens = FenDataset::read_epd(&root, split, 0.9, num_fens_total);

        let items: Vec<_> = fens
            .into_iter()
            .map(|s| FenItemRaw::new(s.to_owned()))
            .collect();

        let dataset = InMemDataset::new(items);

        Self { dataset }
    }

    pub fn load_path(path: &str, _split: &str) -> PathBuf {
        let mut buf = PathBuf::new();
        buf.push(var("PROJECT_ROOT").expect("Set the $PROJECT_ROOT variable"));
        buf.push("resources/datasets");
        buf.push(path);
        buf
    }

    /// num_fens_total: the number of fens in |train + test|
    pub fn read_epd<P: AsRef<Path>>(
        root: &P,
        split: &str,
        split_ratio: f32,
        num_fens_total: usize,
    ) -> Vec<String> {
        println!("Reading EPD from path: {:?}", root.as_ref());
        let epd = fs::read_to_string(root).expect("Couldn't read path");
        let lines = epd
            .lines()
            .filter_map(|l| {
                let mut tok = Tokenizer::new(l);
                let mut pos = Position::try_from(FenImport(&mut tok))
                    .map_err(|err| {
                        log::error!(target: "data", "Error while parsing FEN from EPD line {l}: {err}");
                        err
                    })
                    .ok()?;

                // if its a epd it could have some op codes with moves that lead to the actual
                // position.
                if let Some("fm" | "sm") = tok.next_token() {
                    let tok = tok.next_token().unwrap();
                    let mov = Move::from_lan(tok, &pos)
                        .map_err(|err| {
                            log::error!(target: "data", "Error while parsing LAN from EPD line {l}: {err}");
                            err
                        })
                        .ok()?;
                    pos.make_move(mov);
                }

                // output the fen for the line
                let fen = FenExport(&pos).to_string();
                Some(fen)
            })
            .take(num_fens_total)
            .collect_vec();
        let split_idx = (lines.len() as f32 * split_ratio) as usize;
        match split {
            "train" => lines[..=split_idx].to_vec(),
            "test" => lines[split_idx + 1..].to_vec(),
            _ => panic!("invalid split"),
        }
    }
}

#[derive(Clone, Default)]
pub struct IdentityBatcher<I> {
    item: PhantomData<I>,
}

impl<B: Backend, I: Send + Sync> Batcher<B, I, Vec<I>> for IdentityBatcher<I> {
    fn batch(&self, items: Vec<I>, _device: &B::Device) -> Vec<I> {
        items
    }
}

pub fn build_dataloader<B: AutodiffBackend>(
    config: &TrainingConfig, split: &str,
) -> Arc<dyn DataLoader<B, Vec<FenItemRaw>>> {
    DataLoaderBuilder::<B, _, _>::new(IdentityBatcher::<FenItemRaw>::default())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(0)
        .build(FenDataset::from_epd(
            &config.epd_dataset_path,
            split,
            config.epd_dataset_fens_total,
        ))
}
