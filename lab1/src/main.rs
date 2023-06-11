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
    let rgb_norm = rgb / 255.0;

    let r = rgb_norm.select(0, 0);
    let g = rgb_norm.select(0, 1);
    let b = rgb_norm.select(0, 2);

    let c_max = rgb_norm.amax(&[0], true);
    let c_min = rgb_norm.amin(&[0], true);
    let delta = &c_max - &c_min;

    let h = Tensor::zeros_like(&r)
        .f_where_self(&c_max.eq_tensor(&r).bitwise_not(), &((&g - &b) / &delta * 60.0 + 360.0).remainder(360.0)).unwrap()
        .f_where_self(&c_max.eq_tensor(&g).bitwise_not(), &((&b - &r) / &delta * 60.0 + 120.0)).unwrap()
        .f_where_self(&c_max.eq_tensor(&b).bitwise_not(), &((&r - &g) / &delta * 60.0 + 240.0)).unwrap()
        .masked_fill(&c_max.eq_tensor(&c_min), 0.0);

    let s = Tensor::zeros_like(&r)
        .f_where_self(&c_max.eq(0.0), &(&delta / &c_max))
        .unwrap();

    let v = c_max;

    Tensor::stack(&[h, s, v], 1).squeeze_dim(0)
}

fn hsv_to_rgb(hsv: &Tensor) -> Tensor {
    let h = hsv.select(0, 0);
    let s = hsv.select(0, 1);
    let v = hsv.select(0, 2);

    let c = &v * s;
    let x = &c * (1 / ((&h / 60).remainder(2) - 1).abs());
    let m = v - &c;

    let r = Tensor::zeros_like(&h);
    let g = Tensor::zeros_like(&h);
    let b = Tensor::zeros_like(&h);

    let condition = h.lt(60.0).bitwise_not();
    let r = r.where_self(&condition, &c);
    let g = g.where_self(&condition, &x);

    let condition = (&h.ge(60.0) * &h.lt(120.0)).bitwise_not();
    let r = r.where_self(&condition, &x);
    let g = g.where_self(&condition, &c);

    let condition = (&h.ge(120.0) * &h.lt(180.0)).bitwise_not();
    let g = g.where_self(&condition, &c);
    let b = b.where_self(&condition, &x);

    let condition = (&h.ge(180.0) * &h.lt(240.0)).bitwise_not();
    let g = g.where_self(&condition, &x);
    let b = b.where_self(&condition, &c);

    let condition = (&h.ge(240.0) * &h.lt(300.0)).bitwise_not();
    let r = r.where_self(&condition, &x);
    let b = b.where_self(&condition, &c);

    let condition = h.ge(300.0).bitwise_not();
    let r = r.where_self(&condition, &c);
    let b = b.where_self(&condition, &x);

    Tensor::stack(&[&r + &m, &g + &m, &b + &m], 1).squeeze_dim(0) * 255.0
}

#[allow(dead_code)]
fn vec_f32_from(t: &Tensor) -> Vec<f32> {
    from::<Vec<f32>>(&t.reshape(-1))
}

#[allow(dead_code)]
fn from<'a, T>(t: &'a Tensor) -> T
where
    <T as TryFrom<&'a tch::Tensor>>::Error: std::fmt::Debug,
    T: TryFrom<&'a Tensor>,
{
    T::try_from(t).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_hsv() {
        let rgb = Tensor::from_slice(&[20.0, 40.0, 60.0]).reshape([3, 1, 1]);
        let hsv = rgb_to_hsv(&rgb);
        let expected = Tensor::from_slice(&[210.0, 0.6666667, 0.23529412]).reshape([3, 1, 1]);
        assert_tensors_close(&expected, &hsv);
    }

    #[test]
    fn test_hsv_to_rgb() {
        let hsv = Tensor::from_slice(&[210.0, 0.6666667, 0.23529412]).reshape([3, 1, 1]);
        let rgb = hsv_to_rgb(&hsv);
        let expected = Tensor::from_slice(&[20.0, 40.0, 60.0]).reshape([3, 1, 1]);
        assert_tensors_close(&expected, &rgb);
    }

    fn assert_tensors_close(expected: &Tensor, actual: &Tensor) {
        let is_close = actual.allclose(&expected, 1e-2, 1e-3, false);
        assert!(
            is_close, 
            "tensors do not match, expected: {:?} ({:?}), actual: {:?} ({:?})", 
            vec_f32_from(expected), 
            expected.size(), 
            vec_f32_from(actual), 
            actual.size()
        );
    }
}