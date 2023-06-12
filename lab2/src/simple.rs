use tch::{Device, Kind, nn::{self, VarStore, ModuleT, Conv2D, Linear}};

// simple resnet model, just to verify that everything works

#[derive(Debug)]
pub struct SimpleModel {
    vs: VarStore,

    conv1: Conv2D,
    conv2: Conv2D,
    linear1: Linear,
}

impl SimpleModel {
    pub fn new(device: Device, total_classes: i64) -> Self {
        let vs = VarStore::new(device);
        let root = vs.root();

        let conv1 = nn::conv2d(&root, 3, 10, 5, Default::default());
        let conv2 = nn::conv2d(&root, 10, 8, 3, Default::default());
        
        let linear1 = nn::linear(&root, 8 * 218 * 218, total_classes, Default::default());

        Self {
            vs,

            conv1,
            conv2,
            linear1,
        }
    }

    pub fn var_store(&self) -> &VarStore {
        &self.vs
    }
}

impl ModuleT for SimpleModel {
    fn forward_t(&self, xs: &tch::Tensor, _train: bool) -> tch::Tensor {
        xs
            .apply(&self.conv1)
            .relu()
            .apply(&self.conv2)
            .relu()
            .view([-1, 8 * 218 * 218])
            .apply(&self.linear1)
            .log_softmax(-1, Kind::Float)
    }
}
