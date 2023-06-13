use {
    tch::{Device, Tensor, Reduction, Kind, no_grad, nn::{Adam, OptimizerConfig, ModuleT}, vision},
    crate::{
        config::Config,
        metrics::{Metrics, EpochMetrics},
        data::{prepare_datasets, Dataset},
        simple::SimpleModel,
        inception_resnet_v2::InceptionResnetV2,
    },
};

pub mod config;
pub mod data;
pub mod inception_resnet_v2;
pub mod metrics;
pub mod simple;

fn main() {
    println!("let's train Inception-ResNet-v2 on ImageNet");

    let config = Config::load();
    let metrics = Metrics::new(config.metrics_endpoint.clone(), config.metrics_password.clone(), &config.run_id);

    tch::maybe_init_cuda();
    let device = Device::cuda_if_available();

    println!("using device: {:?}", device);

    // prepare_datasets();
    train(device, config, metrics);
}

fn train(device: Device, config: Config, mut metrics: Metrics) {
    let total_classes = 2;
    let net = InceptionResnetV2::new(device, total_classes);

    let mut opt = Adam::default().build(&net.var_store(), 1e-4).unwrap();
    let mut epoch = 0;

    let mut best_test_accuracy = 0.0;

    loop {
        println!("running epoch: {}", epoch);
        metrics.set_epoch(epoch);

        let mut train = Dataset::new("train", 128);
        let mut train_metrics = EpochMetrics::new();
        while train.has_more_chunks() {
            let mut chunk = train.next_chunk();
            for (xs, ys) in chunk.to_device(device) {
                // let xs = vision::dataset::augmentation(&xs, true, 4, 8);

                opt.zero_grad();

                train_metrics.total_samples += xs.size()[0];

                let outputs = net.forward_t(&xs, true);
                let loss = outputs.cross_entropy_for_logits(&ys.argmax(1, false));
                train_metrics.total_loss += loss.double_value(&[]);
                train_metrics.total_correct += count_correct(&outputs, &ys);

                opt.backward_step(&loss);
            }
        }
        metrics.set_train_metrics(&train_metrics);

        let mut test = Dataset::new("val", 128);
        let mut test_metrics = EpochMetrics::new();
        no_grad(|| {
            while test.has_more_chunks() {
                let mut chunk = test.next_chunk();
                for (xs, ys) in chunk.to_device(device) {
                    test_metrics.total_samples += xs.size()[0];

                    let outputs = net.forward_t(&xs, false);
                    let loss = outputs.cross_entropy_for_logits(&ys.argmax(1, false));
                    test_metrics.total_loss += loss.double_value(&[]);
                    test_metrics.total_correct += count_correct(&outputs, &ys);
                }
            }
        });
        metrics.set_test_metrics(&test_metrics);

        metrics.report();
        
        let acc = metrics.test_accuracy();
        if acc > best_test_accuracy {
            println!("best epoch so far, saving weights");
            net.var_store().save(format!("weights/{}.pt", &config.run_id)).unwrap();
            best_test_accuracy = acc;
        }

        epoch += 1;
    }
}

fn count_correct(output: &Tensor, ys: &Tensor) -> i64 {
    let (_, predictions) = output.max_dim(-1, false);
    let (_, expected) = ys.max_dim(-1, false);

    let correct = expected.eq_tensor(&predictions);
    correct.sum(Kind::Float).int64_value(&[])
}