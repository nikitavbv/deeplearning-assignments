use tch::{
    Device, 
    Tensor, 
    nn::{
        self, 
        VarStore, 
        ModuleT, 
        Conv2D, 
        BatchNorm, 
        ConvConfig, 
        ConvConfigND,
        Path, 
        BatchNormConfig,
        SequentialT,
        seq_t,
        Linear,
    },
};

// https://arxiv.org/pdf/1602.07261v2.pdf
// https://ai.googleblog.com/2016/08/improving-inception-and-image.html?m=1

#[derive(Debug)]
pub struct InceptionResnetV2 {
    vs: VarStore,

    stem: Stem,
    transition_a: TransitionA,
    inception_block_a: SequentialT,
    transition_b: TransitionB,
    inception_block_b: SequentialT,
    transition_c: TransitionC,
    inception_block_c: SequentialT,
    inception_block_c_last: InceptionBlockC,
    conv1: ConvLayer,
    linear1: Linear,
}

impl InceptionResnetV2 {
    pub fn new(device: Device, total_classes: i64) -> Self {
        let vs = VarStore::new(device);
        let path = vs.root();

        Self {
            stem: Stem::new(&path),
            transition_a: TransitionA::new(&path),
            inception_block_a: seq_t()
                .add(InceptionBlockA::new(&path, 0.17))
                .add(InceptionBlockA::new(&path, 0.17))
                .add(InceptionBlockA::new(&path, 0.17))
                .add(InceptionBlockA::new(&path, 0.17))
                .add(InceptionBlockA::new(&path, 0.17))
                .add(InceptionBlockA::new(&path, 0.17))
                .add(InceptionBlockA::new(&path, 0.17))
                .add(InceptionBlockA::new(&path, 0.17))
                .add(InceptionBlockA::new(&path, 0.17))
                .add(InceptionBlockA::new(&path, 0.17)),
            transition_b: TransitionB::new(&path),
            inception_block_b: seq_t()
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10))
                .add(InceptionBlockB::new(&path, 0.10)),
            transition_c: TransitionC::new(&path),
            inception_block_c: seq_t()
                .add(InceptionBlockC::new(&path, 0.20, true))
                .add(InceptionBlockC::new(&path, 0.20, true))
                .add(InceptionBlockC::new(&path, 0.20, true))
                .add(InceptionBlockC::new(&path, 0.20, true))
                .add(InceptionBlockC::new(&path, 0.20, true))
                .add(InceptionBlockC::new(&path, 0.20, true))
                .add(InceptionBlockC::new(&path, 0.20, true))
                .add(InceptionBlockC::new(&path, 0.20, true))
                .add(InceptionBlockC::new(&path, 0.20, true)),
            inception_block_c_last: InceptionBlockC::new(&path, 1.0, false),
            conv1: ConvLayer::new(&path, 2080, 1536, 1, 1, 0),
            linear1: nn::linear(&path, 1536, total_classes, Default::default()),

            vs
        }
    }

    pub fn var_store(&self) -> &VarStore {
        &self.vs
    }
}

impl ModuleT for InceptionResnetV2 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x = xs
            .apply_t(&self.stem, train)
            .apply_t(&self.transition_a, train)
            .apply_t(&self.inception_block_a, train)
            .apply_t(&self.transition_b, train)
            .apply_t(&self.inception_block_b, train)
            .apply_t(&self.transition_c, train)
            .apply_t(&self.inception_block_c, train)
            .apply_t(&self.inception_block_c_last, train)
            .apply_t(&self.conv1, train)
            .avg_pool2d([5, 5], [5, 5], [0, 0], false, false, None);
        
        x.view([x.size()[0], -1]).apply(&self.linear1)
    }
}

#[derive(Debug)]
pub struct Stem {
    conv1: ConvLayer,
    conv2: ConvLayer,
    conv3: ConvLayer,

    conv4: ConvLayer,
    conv5: ConvLayer,
}

impl Stem {
    pub fn new(path: &Path<'_>) -> Self {
        Self {
            conv1: ConvLayer::new(path, 3, 32, 3, 2, 0),
            conv2: ConvLayer::new(path, 32, 32, 3, 1, 0),
            conv3: ConvLayer::new(path, 32, 64, 3, 1, 1),

            conv4: ConvLayer::new(path, 64, 80, 1, 1, 0),
            conv5: ConvLayer::new(path, 80, 192, 3, 1, 0),
        }
    }
}

impl ModuleT for Stem {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs
            .apply_t(&self.conv1, train)
            .apply_t(&self.conv2, train)
            .apply_t(&self.conv3, train)
            .max_pool2d([3, 3], [2, 2], [0, 0], [1, 1], false)
            .apply_t(&self.conv4, train)
            .apply_t(&self.conv5, train)
            .max_pool2d([3, 3], [2, 2], [0, 0], [1, 1], false)
    }
}

#[derive(Debug)]
pub struct ConvLayer {
    conv: Conv2D,
    batch_norm: BatchNorm,
}

impl ConvLayer {
    pub fn new(path: &Path<'_>, input: i64, output: i64, kernel_size: i64, stride: i64, padding: i64) -> Self {
        Self {
            conv: nn::conv2d(path, input, output, kernel_size, ConvConfig {
                padding,
                bias: false,
                stride,
                ..Default::default()
            }),
            batch_norm: nn::batch_norm2d(path, output, BatchNormConfig {
                eps: 0.001,
                momentum: 0.1,
                affine: true,
                ..Default::default()
            }),
        }
    }

    pub fn new_2d(path: &Path<'_>, input: i64, output: i64, kernel_size: (i64, i64), stride: (i64, i64), padding: (i64, i64)) -> Self {
        Self {
            conv: nn::conv(path, input, output, [kernel_size.0, kernel_size.1], ConvConfigND {
                padding: [padding.0, padding.1],
                bias: false,
                stride: [stride.0, stride.1],
                ..Default::default()
            }),
            batch_norm: nn::batch_norm2d(path, output, BatchNormConfig {
                eps: 0.001,
                momentum: 0.1,
                affine: true,
                ..Default::default()
            })
        }
    }
}

impl ModuleT for ConvLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs
            .apply(&self.conv)
            .apply_t(&self.batch_norm, train)
            .relu()
    }
}

#[derive(Debug)]
pub struct TransitionA {
    branch0: ConvLayer,
    branch1: SequentialT,
    branch2: SequentialT,
    branch3: ConvLayer,
}

impl TransitionA {
    pub fn new(path: &Path<'_>) -> Self {
        Self {
            branch0: ConvLayer::new(path, 192, 96, 1, 1, 0),
            branch1: seq_t()
                .add(ConvLayer::new(path, 192, 48, 1, 1, 0))
                .add(ConvLayer::new(path, 48, 64, 5, 1, 2)),
            branch2: seq_t()
                .add(ConvLayer::new(path, 192, 64, 1, 1, 0))
                .add(ConvLayer::new(path, 64, 96, 3, 1, 1))
                .add(ConvLayer::new(path, 96, 96, 3, 1, 1)),
            branch3: ConvLayer::new(path, 192, 64, 1, 1, 0),
        }
    }
}

impl ModuleT for TransitionA {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x0 = xs.apply_t(&self.branch0, train);
        let x1 = xs.apply_t(&self.branch1, train);
        let x2 = xs.apply_t(&self.branch2, train);
        let x3 = xs.avg_pool2d([3, 3], [1, 1], [1, 1], false, false, None)
            .apply_t(&self.branch3, true);
        Tensor::cat(&[x0, x1, x2, x3], 1)
    }
}

#[derive(Debug)]
pub struct InceptionBlockA {
    scale: f32,
    
    branch0: ConvLayer,
    branch1: SequentialT,
    branch2: SequentialT,

    conv1: Conv2D,
}

impl InceptionBlockA {
    pub fn new(path: &Path<'_>, scale: f32) -> Self {
        Self {
            scale,

            branch0: ConvLayer::new(path, 320, 32, 1, 1, 0),
            branch1: seq_t()
                .add(ConvLayer::new(path, 320, 32, 1, 1, 0))
                .add(ConvLayer::new(path, 32, 32, 3, 1, 1)),
            branch2: seq_t()
                .add(ConvLayer::new(path, 320, 32, 1, 1, 0))
                .add(ConvLayer::new(path, 32, 48, 3, 1, 1))
                .add(ConvLayer::new(path, 48, 64, 3, 1, 1)),
            
            conv1: nn::conv2d(path, 128, 320, 1, ConvConfig {
                stride: 1,
                ..Default::default()
            }),
        }
    }
}

impl ModuleT for InceptionBlockA {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x0 = xs.apply_t(&self.branch0, train);
        let x1 = xs.apply_t(&self.branch1, train);
        let x2 = xs.apply_t(&self.branch2, train);
        let out = Tensor::cat(&[x0, x1, x2], 1)
            .apply_t(&self.conv1, train);
        let out = self.scale * out + xs;
        out.relu()
    }
}

#[derive(Debug)]
struct TransitionB {
    branch0: ConvLayer,
    branch1: SequentialT,
}

impl TransitionB {
    pub fn new(path: &Path<'_>) -> Self {
        Self {
            branch0: ConvLayer::new(path, 320, 384, 3, 2, 0),
            branch1: seq_t()
                .add(ConvLayer::new(path, 320, 256, 1, 1, 0))
                .add(ConvLayer::new(path, 256, 256, 3, 1, 1))
                .add(ConvLayer::new(path, 256, 384, 3, 2, 0)),
        }
    }
}

impl ModuleT for TransitionB {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x0 = xs.apply_t(&self.branch0, train);
        let x1 = xs.apply_t(&self.branch1, train);
        let x2 = xs.max_pool2d([3, 3], [2, 2], [0, 0], [1, 1], false);
        Tensor::cat(&[x0, x1, x2], 1)
    }
}

#[derive(Debug)]
struct InceptionBlockB {
    scale: f32,

    branch0: ConvLayer,
    branch1: SequentialT,

    conv1: Conv2D,
}

impl InceptionBlockB {
    pub fn new(path: &Path<'_>, scale: f32) -> Self {
        Self {
            scale,

            branch0: ConvLayer::new(path, 1088, 192, 1, 1, 0),
            branch1: seq_t()
                .add(ConvLayer::new(path, 1088, 128, 1, 1, 0))
                .add(ConvLayer::new_2d(path, 128, 160, (1, 7), (1, 1), (0, 3)))
                .add(ConvLayer::new_2d(path, 160, 192, (7, 1), (1, 1), (3, 0))),

            conv1: nn::conv2d(path, 384, 1088, 1, ConvConfig {
                stride: 1,
                ..Default::default()
            }),
        }
    }
}

impl ModuleT for InceptionBlockB {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x0 = xs.apply_t(&self.branch0, train);
        let x1 = xs.apply_t(&self.branch1, train);

        let out = Tensor::cat(&[x0, x1], 1).apply_t(&self.conv1, train);
        let out = self.scale * out + xs;
        out.relu()
    }
}

#[derive(Debug)]
struct TransitionC {
    branch0: SequentialT,
    branch1: SequentialT,
    branch2: SequentialT,
}

impl TransitionC {
    pub fn new(path: &Path<'_>) -> Self {
        Self {
            branch0: seq_t()
                .add(ConvLayer::new(path, 1088, 256, 1, 1, 0))
                .add(ConvLayer::new(path, 256, 384, 3, 2, 0)),
            branch1: seq_t()
                .add(ConvLayer::new(path, 1088, 256, 1, 1, 0))
                .add(ConvLayer::new(path, 256, 288, 3, 2, 0)),
            branch2: seq_t()
                .add(ConvLayer::new(path, 1088, 256, 1, 1, 0))
                .add(ConvLayer::new(path, 256, 288, 3, 1, 1))
                .add(ConvLayer::new(path, 288, 320, 3, 2, 0)),
        }
    }
}

impl ModuleT for TransitionC {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x0 = xs.apply_t(&self.branch0, train);
        let x1 = xs.apply_t(&self.branch1, train);
        let x2 = xs.apply_t(&self.branch2, train);
        let x3 = xs.max_pool2d([3, 3], [2, 2], [0, 0], [1, 1], false);
        Tensor::cat(&[x0, x1, x2, x3], 1)
    }
}

#[derive(Debug)]
struct InceptionBlockC {
    scale: f32,
    apply_relu: bool,

    branch0: ConvLayer,
    branch1: SequentialT,

    conv1: Conv2D,
}

impl InceptionBlockC {
    pub fn new(path: &Path<'_>, scale: f32, apply_relu: bool) -> Self {
        Self {
            scale,
            apply_relu,

            branch0: ConvLayer::new(path, 2080, 192, 1, 1, 0),
            branch1: seq_t()
                .add(ConvLayer::new(path, 2080, 192, 1, 1, 0))
                .add(ConvLayer::new_2d(path, 192, 224, (1, 3), (1, 1), (0, 1)))
                .add(ConvLayer::new_2d(path, 224, 256, (3, 1), (1, 1), (1, 0))),
            
            conv1: nn::conv2d(path, 448, 2080, 1, ConvConfig {
                stride: 1,
                ..Default::default()
            }),
        }
    }
}

impl ModuleT for InceptionBlockC {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x0 = xs.apply_t(&self.branch0, train);
        let x1 = xs.apply_t(&self.branch1, train);
        let out = Tensor::cat(&[x0, x1], 1).apply(&self.conv1);
        let out = self.scale * out + xs;
        if self.apply_relu {
            out.relu()
        } else {
            out
        }
    }
}