use crate::{
    config::Config,
    metrics::Metrics,
};

pub mod config;
pub mod metrics;

fn main() {
    println!("let's train Inception-ResNet-v2 on ImageNet");

    let config = Config::load();
    let mut metrics = Metrics::new(config.metrics_endpoint.clone(), config.metrics_password.clone(), &config.run_id);

}
