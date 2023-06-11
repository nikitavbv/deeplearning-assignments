use tch::{Tensor, vision};

fn main() {
    // [channel, height, width] - 0,0 is top left, channels are rgb
    let rgb = vision::image::load("./demo.jpg").unwrap();

    // Провести лінійну та експоненційну корекцію яскравості зображення через простір HSV 
    // (для цього нормалізуйте V канал). Не забудьте, що колірний простір, у якому cv2.imread 
    // віддає зображення це BGR.
    let hsv = rgb_to_hsv(&rgb);
}

fn rgb_to_hsv(rgb: &Tensor) -> Tensor {
    Tensor::from_slice(&[0.0, 0.0, 0.0]).reshape([3, 1, 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_hsv() {
        let rgb = Tensor::from_slice(&[20.0, 40.0, 60.0]).reshape([3, 1, 1]);
        let hsv = rgb_to_hsv(&rgb);
        panic!("f: {:?}", vec_f32_from(&hsv));

        // assert!(hsv.allclose(&Tensor::from_slice(&[0.0, 0.0, 0.0]).reshape([3, 1, 1]), 1e-5, 1e-8, false));
    }

    fn from<'a, T>(t: &'a Tensor) -> T
    where
        <T as TryFrom<&'a tch::Tensor>>::Error: std::fmt::Debug,
        T: TryFrom<&'a Tensor>,
    {
        T::try_from(t).unwrap()
    }

    fn vec_f32_from(t: &Tensor) -> Vec<f32> {
        from::<Vec<f32>>(&t.reshape(-1))
    }    
}