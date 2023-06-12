use {
    std::{collections::HashMap, time::Instant},
    prometheus::{
        Registry,
        TextEncoder,
        IntGauge,
        GaugeVec,
        register_int_gauge_with_registry,
        register_gauge_vec_with_registry,
    },
};

// I know that using something like MLFlow would be better, but this is faster to integrate and gets the job don de.

pub struct Metrics {
    pushed_at: Option<Instant>,

    registry: Registry,
    encoder: TextEncoder,
    client: reqwest::blocking::Client,
    endpoint: String,
    password: String,

    epoch: IntGauge,
    loss: GaugeVec,
    accuracy: GaugeVec,
}

impl Metrics {
    pub fn new(endpoint: String, password: String, run_id: &str) -> Self {
        let mut labels = HashMap::new();
        labels.insert("run_id".to_owned(), run_id.to_owned());

        let registry = Registry::new_custom(
            Some("deeplearning_lab_training".to_owned()),
            Some(labels)
        ).unwrap();

        Self {
            epoch: register_int_gauge_with_registry!("epoch", "training epoch", registry).unwrap(),
            loss: register_gauge_vec_with_registry!("loss", "loss", &["phase"], registry).unwrap(),
            accuracy: register_gauge_vec_with_registry!("accuracy", "accuracy", &["phase"], registry).unwrap(),

            registry,
            encoder: TextEncoder::new(),
            client: reqwest::blocking::Client::new(),
            endpoint,
            password,

            pushed_at: None,
        }
    }

    pub fn set_epoch(&self, epoch: i64) {
        self.epoch.set(epoch);
    }

    pub fn set_train_metrics(&self, metrics: &EpochMetrics) {
        self.set_metrics("train", metrics);
    }

    pub fn set_test_metrics(&self, metrics: &EpochMetrics) {
        self.set_metrics("test", metrics);
    }

    fn set_metrics(&self, phase: &str, metrics: &EpochMetrics) {
        self.loss.with_label_values(&[phase]).set(metrics.total_loss / (metrics.total_samples as f64));
        self.accuracy.with_label_values(&[phase]).set((metrics.total_correct as f64) / (metrics.total_samples as f64));
    }

    pub fn report(&mut self) {
        self.push_to_stdout();
        self.push_to_remote();
    }

    pub fn push_to_stdout(&self) {
        println!("train loss = {}, acc = {}", self.loss.get_metric_with_label_values(&["train"]).unwrap().get(), self.accuracy.get_metric_with_label_values(&["train"]).unwrap().get());
        println!("test loss = {}, acc = {}", self.loss.get_metric_with_label_values(&["test"]).unwrap().get(), self.accuracy.get_metric_with_label_values(&["test"]).unwrap().get());
    }

    pub fn push_to_remote(&mut self) {
        let now = Instant::now();
        if self.pushed_at.map(|v| (now - v).as_secs_f32() < 1.0).unwrap_or(false) {
            // do not push metrics more frequently than once every second
            return;
        }

        let metrics = self.registry.gather();
        let encoded = self.encoder.encode_to_string(&metrics).unwrap();

        let res = self.client.post(&self.endpoint)
            .basic_auth("vmuser", Some(self.password.clone()))
            .body(encoded)
            .send()
            .unwrap();

        if !res.status().is_success() {
            eprintln!("failed to push metrics, status = {:?}", res.status());
        }

        self.pushed_at = Some(now);
    }
}

pub struct EpochMetrics {
    pub total_samples: i64,
    pub total_loss: f64,
    pub total_correct: i64,
}

impl EpochMetrics {
    pub fn new() -> Self {
        Self {
            total_samples: 0,
            total_loss: 0.0,
            total_correct: 0,
        }        
    }
}