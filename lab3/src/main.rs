use tch::Device;

pub mod model;

fn main() {
    println!("Let's train RNN to classify some text");

    let device = Device::cuda_if_available();

    
}
