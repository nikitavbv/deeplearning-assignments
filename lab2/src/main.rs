use {
    tch::Device,
    crate::{
        config::Config,
        metrics::Metrics,
        data::{prepare_datasets, Dataset},
    },
};

pub mod config;
pub mod data;
pub mod metrics;

fn main() {
    println!("let's train Inception-ResNet-v2 on ImageNet");

    let config = Config::load();
    let metrics = Metrics::new(config.metrics_endpoint.clone(), config.metrics_password.clone(), &config.run_id);

    tch::maybe_init_cuda();
    let device = Device::cuda_if_available();

    println!("using device: {:?}", device);

    // prepare_datasets();
    train(device);
}

fn train(device: Device) {
    loop {
        let mut train = Dataset::new("train", 128);
        while train.has_more_chunks() {
            let mut chunk = train.next_chunk();
            for (xs, ys) in chunk.to_device(device) {
                println!("data: {:?} {:?}", xs, ys);
            }
        }
        panic!("done");
    }
}