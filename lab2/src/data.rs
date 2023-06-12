use {
    std::{fs::{self, read_dir, read_to_string}, collections::HashMap},
    rand::prelude::*,
    tch::{Tensor, data::Iter2, vision::imagenet::load_image_and_resize224},
};

// Dataset downloaded from https://huggingface.co/datasets/imagenet-1k/tree/main/data

pub fn prepare_datasets() {
    println!("preparing datasets");

    let classes_to_use: HashMap<String, usize> = vec![
        "n07873807", // pizza
        "n07718472", // cucumber
    ].iter().enumerate().map(|v| (v.1.to_string(), v.0)).collect();
    
    prepare_dataset(&classes_to_use, "train");
    prepare_dataset(&classes_to_use, "val");
}

fn prepare_dataset(classes_to_use: &HashMap<String, usize>, dataset: &str) {
    println!("preparing dataset: {}", dataset);

    let total_classes = classes_to_use.len();

    let mut images = Vec::new();
    let mut labels = Vec::new();
    let mut chunk_index = 0;

    let max_chunk_size = 5000;

    let mut total = 0;

    for file in read_dir(format!("imagenet/raw/{}", dataset)).unwrap() {
        let file = file.unwrap();
        let name = file.file_name().to_string_lossy().to_string().replace(".JPEG", "");

        let class = name.split("_").last().unwrap();
        let class_index = classes_to_use.get(class);

        if let Some(class_index) = class_index {
            let image = load_image_and_resize224(file.path()).unwrap().unsqueeze(0);
            images.push(image);

            let mut label = vec![0.0; total_classes];
            label[*class_index] =  1.0;
            let label = Tensor::from_slice(&label).unsqueeze(0);
            labels.push(label);
            
            total += 1;
        
            if images.len() > max_chunk_size {
                save_chunk(dataset, chunk_index, &images, &labels);
                chunk_index += 1;

                images.clear();
                labels.clear();
            }
        }
    }

    if images.len() > 0 {
        save_chunk(dataset, chunk_index, &images, &labels);
        fs::write(format!("./imagenet/dataset/{}/total_chunks", dataset), (chunk_index + 1).to_string()).unwrap();
    }

    println!("done, total entries: {}", total);
}

fn save_chunk(dataset: &str, chunk_index: i64, images: &[Tensor], labels: &[Tensor]) {
    let images = Tensor::cat(&images, 0);
    let labels = Tensor::cat(&labels, 0);

    images.save(format!("./imagenet/dataset/{}/images_{}.pt", dataset, chunk_index)).unwrap();
    labels.save(format!("./imagenet/dataset/{}/labels_{}.pt", dataset, chunk_index)).unwrap();
}

pub struct Dataset {
    name: String,
    chunks: Vec<u64>,
    batch_size: i64,
}

impl Dataset {
    pub fn new(name: &str, batch_size: i64) -> Self {
        let total_chunks: u64 = read_to_string(format!("./imagenet/dataset/{}/total_chunks", name)).unwrap().parse().unwrap();
        let mut chunks: Vec<u64> = (0..total_chunks).collect();
        chunks.shuffle(&mut rand::thread_rng());

        Self {
            name: name.to_owned(),
            chunks,
            batch_size,
        }
    }

    pub fn has_more_chunks(&self) -> bool {
        !self.chunks.is_empty()
    }

    pub fn next_chunk(&mut self) -> Iter2 {
        let chunk_index = self.chunks.pop().unwrap();
        let xs = Tensor::load(format!("./imagenet/dataset/{}/images_{}.pt", &self.name, chunk_index)).unwrap();
        let ys = Tensor::load(format!("./imagenet/dataset/{}/labels_{}.pt", &self.name, chunk_index)).unwrap();
        let mut iter = Iter2::new(&xs, &ys, self.batch_size);
        iter.return_smaller_last_batch();
        iter.shuffle();
        iter
    }
}
