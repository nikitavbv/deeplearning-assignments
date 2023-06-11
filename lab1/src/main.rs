use tch::{Tensor, Kind, Device, vision, nn::{self, Module}, no_grad};

fn main() {
    // [channel, height, width] - 0,0 is top left, channels are rgb
    let rgb = vision::image::load("./demo.jpg").unwrap();

    // Провести лінійну та експоненційну корекцію яскравості зображення через простір HSV 
    // (для цього нормалізуйте V канал). Не забудьте, що колірний простір, у якому cv2.imread 
    // віддає зображення це BGR.
    let hsv = rgb_to_hsv(&rgb);
    
    let hsv_linear = normalize_brightness_linearly(&hsv);
    let hsv_exp = normalize_brightness_exp(&hsv);
    
    let rgb_linear = hsv_to_rgb(&hsv_linear);
    let rgb_exp = hsv_to_rgb(&hsv_exp);
    
    vision::image::save(&rgb_linear, "./report/images/brightness-linear.png").unwrap();
    vision::image::save(&rgb_exp, "./report/images/brightness-exp.png").unwrap();

    // Застосувати до зображення фільтр Гауса
    let image_gaussian_filter = apply_gaussian_filter(&rgb, 5, 3.0);
    vision::image::save(&image_gaussian_filter, "./report/images/gaussian.png").unwrap();

    // box-фільтр
    let image_box_filter = apply_box_filter(&rgb, 5, 0.04);
    vision::image::save(&image_box_filter, "./report/images/box.png").unwrap();

    // unsharp masking
    let image_unsharp_masking = apply_unsharp_masking(&apply_gaussian_filter(&rgb, 5, 10.0));
    vision::image::save(&image_unsharp_masking, "./report/images/unsharp.png").unwrap();

    // Провести детекцію границь оператором Собеля.
    let image_edges = apply_edge_detection(&apply_gaussian_filter(&rgb, 5, 10.0));
    vision::image::save(&image_edges, "./report/images/edges.png").unwrap();
}

fn apply_edge_detection(rgb: &Tensor) -> Tensor {
    let kernel = Tensor::from_slice(&[
         0.0, -1.0,  0.0,
        -1.0,  4.0, -1.0,
         0.0, -1.0,  0.0,
    ]).to_kind(Kind::Float).reshape([3, 3]);

    apply_kernel(rgb, &kernel, 3)
}

fn apply_unsharp_masking(rgb: &Tensor) -> Tensor {
    let a = Tensor::from_slice(&[
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 0.0,
    ]).to_kind(Kind::Float).reshape([3, 3]);

    let b = Tensor::from_slice(&[
        0.0, -1.0, 0.0,
        -1.0, 4.0, -1.0,
        0.0, -1.0, 0.0,
    ]).to_kind(Kind::Float).reshape([3, 3]);

    let kernel: Tensor = a + 2 * b;

    apply_kernel(rgb, &kernel, 3)
}

fn apply_box_filter(rgb: &Tensor, size: i64, value: f32) -> Tensor {
    apply_kernel(rgb, &box_kernel(size, value), size)
}

fn box_kernel(size: i64, value: f32) -> Tensor {
    Tensor::from(value).repeat([size, size])
}

fn apply_gaussian_filter(rgb: &Tensor, size: i64, std: f32) -> Tensor {
    apply_kernel(rgb, &gaussian_kernel(rgb.device(), size, std), size)
}

fn gaussian_kernel(device: Device, size: i64, std: f32) -> Tensor {
    // formula: https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
    let one_d: Tensor = Tensor::arange(size, (Kind::Float, device)) - (size / 2);
    let x = one_d.view([size, 1]);
    let y = one_d.view([1, size]);

    let filter = (-(&x * &x + &y * &y) / Tensor::from(2.0f32 * std * std)).exp() / Tensor::from(2.0 * std::f32::consts::PI * std * std);
    &filter / filter.sum(Kind::Float)
}

fn apply_kernel(rgb: &Tensor, kernel: &Tensor, size: i64) -> Tensor {
    let channels = 3;
    let vs = nn::VarStore::new(rgb.device());
    let mut conv = nn::conv2d(vs.root(), channels, channels, size, nn::ConvConfig {
        groups: channels,
        ..Default::default()
    });
    
    conv.ws = kernel
        .reshape([1, 1, size, size])
        .repeat([channels, 1, 1, 1]);

    no_grad(|| conv.forward(&rgb.to_kind(Kind::Float)))
}

fn normalize_brightness_exp(hsv: &Tensor) -> Tensor {
    let h = hsv.select(0, 0);
    let s = hsv.select(0, 1);
    let v = hsv.select(0, 2);

    let v = (v.exp() * 0.4).clamp_max(1.0);

    Tensor::stack(&[&h, &s, &v], 0)
}

fn normalize_brightness_linearly(hsv: &Tensor) -> Tensor {
    let h = hsv.select(0, 0);
    let s = hsv.select(0, 1);
    let v = hsv.select(0, 2);
    
    let v_min = f64::try_from(v.min()).unwrap();
    let v_max = f64::try_from(v.max()).unwrap();

    let v_delta = &v_max - &v_min;

    let v = (v - v_min) / v_delta;
    
    Tensor::stack(&[&h, &s, &v], 0)
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
    let x = &c * (1.0 - ((&h / 60).remainder(2) - 1).abs());
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

    Tensor::stack(&[&r + &m, &g + &m, &b + &m], 0) * 255.0
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

    #[test]
    fn test_rgb_hsv_rgb_large() {
        let source = vision::image::load("./demo.jpg").unwrap().to_kind(Kind::Float);
        let hsv = rgb_to_hsv(&source);
        let rgb = hsv_to_rgb(&hsv);
        assert_tensors_close(&source, &rgb);
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