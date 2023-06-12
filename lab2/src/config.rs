use {
    std::fs::read_to_string,
    serde::Deserialize,
};

#[derive(Deserialize)]
pub struct Config {
    pub run_id: String,

    pub metrics_endpoint: String,
    pub metrics_password: String,
}

impl Config {
    pub fn load() -> Self {
        toml::from_str(&read_to_string("./config.toml").unwrap()).unwrap()
    }
}