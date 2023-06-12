use tch::{Device, Tensor, nn::{self, VarStore, ModuleT, seq, Sequential}};

// https://arxiv.org/pdf/1602.07261v2.pdf
// https://ai.googleblog.com/2016/08/improving-inception-and-image.html?m=1

#[derive(Debug)]
pub struct InceptionResnetV2 {
    vs: VarStore,

    stem: Stem,
    multi_blocks_a: BlocksA,
    reduction_a: ReductionA,
    multi_blocks_b: BlocksB,
    reduction_b: ReductionB,
    multi_blocks_c: BlocksC,
}

impl InceptionResnetV2 {
    pub fn new(device: Device, total_classes: i64) -> Self {
        let vs = VarStore::new(device);

        Self {
            vs,
        }
    }
}

impl ModuleT for InceptionResnetV2 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
    }
}