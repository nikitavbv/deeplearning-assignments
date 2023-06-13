use {
    std::{fs, path::Path, collections::HashMap},
    tch::{Device, Tensor},
};

pub mod model;

fn main() {
    println!("Let's train RNN to classify some text");

    let device = Device::cuda_if_available();

    println!("loading dataset...");
    let (xs, ys) = load_dataset();
    println!("xs: {:?}", xs);
    println!("ys: {:?}", ys);
}

fn load_dataset() -> (Tensor, Tensor) {
    if Path::new("./dataset/xs.pt").exists() {
        return (Tensor::load("./dataset/xs.pt").unwrap(), Tensor::load("./dataset/ys.pt").unwrap());
    }

    println!("generating dataset");
    let (xs, ys) = create_dataset();
    println!("saving dataset");
    xs.save("./dataset/xs.pt").unwrap();
    ys.save("./dataset/ys.pt").unwrap();

    (xs, ys)
}

fn create_dataset() -> (Tensor, Tensor) {
    let mut reader = csv::Reader::from_path("./dataset/data.csv").unwrap();

    let mut label_for_char = HashMap::<char, u8>::new();

    let mut data = Vec::new();
    let mut labels = Vec::new();
    let mut longest_text = 0;

    for record in reader.records() {
        let record = match record {
            Ok(v) => v,
            Err(_err) => {
                // there are some utf8 errors
                continue;
            }
        };

        let label: u8 = record.get(0).unwrap().parse().unwrap();
        let text = record.get(5).unwrap();

        let mut text_encoded = Vec::new();
        for c in text.chars().into_iter() {
            let len = label_for_char.len() as u8;
            let label = *label_for_char.entry(c).or_insert_with(|| len);
            text_encoded.push(label);
        }
        longest_text = longest_text.max(text_encoded.len());

        data.push(text_encoded);
        labels.push(Tensor::from_slice(&[label]));

        println!("generating dataset: {}", data.len());
    }

    println!("padding tensors");
    let data_tensors: Vec<Tensor> = data.into_iter()
        .map(|v| {
            let data = Tensor::from_slice(&v);
            let zeros = Tensor::zeros(&[(longest_text - v.len()) as i64], (data.kind(), data.device()));
            Tensor::cat(&[zeros, data], 0)
        })
        .collect();

    println!("cat tensors");
    (Tensor::cat(&data_tensors, 0), Tensor::cat(&labels, 0))
}