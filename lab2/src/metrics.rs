use {
    std::{collections::HashMap, time::Instant},
    prometheus::{
        Registry,
        TextEncoder,
        IntGauge,
        register_int_gauge_with_registry,
    },
};

// I know that using something like MLFlow would be better, but this is faster to integrate and gets the job donie.

pub struct Metrics {
    pushed_at: Option<Instant>,

    registry: Registry,
    encoder: TextEncoder,
    client: reqwest::blocking::Client,
    endpoint: String,
    password: String,

    pub epoch: IntGauge,
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

            registry,
            encoder: TextEncoder::new(),
            client: reqwest::blocking::Client::new(),
            endpoint,
            password,

            pushed_at: None,
        }
    }

    pub fn push(&mut self) {
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