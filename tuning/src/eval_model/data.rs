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
use engine::core::search::mcts::nn::{BoardInputFloats, StateInputFloats};
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
    pub fn new(path: &str, split: &str, num_fens_total: usize) -> Self {
        let root = FenDataset::load_path(path, split);
        let fens = FenDataset::read_edp(&root, split, 0.9, num_fens_total);

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
        buf.push("resources/datasets/edp");
        buf.push(path);
        buf
    }

    /// num_fens_total: the number of fens in |train + test|
    pub fn read_edp<P: AsRef<Path>>(
        root: &P,
        split: &str,
        split_ratio: f32,
        num_fens_total: usize,
    ) -> Vec<String> {
        println!("Reading EDP from path: {:?}", root.as_ref());
        let edp = fs::read_to_string(root).expect("Couldn't read path");
        let lines = edp
            .lines()
            .map(|l| l.to_owned())
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
    config: &TrainingConfig,
) -> Arc<dyn DataLoader<B, Vec<FenItemRaw>>> {
    DataLoaderBuilder::<B, _, _>::new(IdentityBatcher::<FenItemRaw>::default())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(0)
        .build(FenDataset::new(
            &config.edp_dataset_path,
            "train",
            config.edp_dataset_fens_total,
        ))
}
