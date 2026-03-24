use image::GrayImage;
use std::sync::LazyLock;

const TARGET_SIZE: u32 = 256;
const CELLS_X: usize = 8;
const CELLS_Y: usize = 8;
const CELL_W: usize = (TARGET_SIZE / CELLS_X as u32) as usize;
const CELL_H: usize = (TARGET_SIZE / CELLS_Y as u32) as usize;
const ORIENT_BINS: usize = 8;
const HASH_BYTES: usize = CELLS_X * CELLS_Y * (ORIENT_BINS / 8); // 16

/// 方向查找表：根据 (dx, dy) 量化后的符号组合得到方向索引
static ORIENT_LUT: LazyLock<[[u8; 9]; 9]> = LazyLock::new(|| {
    let mut lut = [[0u8; 9]; 9];
    for dx in -4i8..=4 {
        for dy in -4i8..=4 {
            let idx_x = (dx + 4) as usize;
            let idx_y = (dy + 4) as usize;
            if dx == 0 && dy == 0 {
                lut[idx_x][idx_y] = 0;
                continue;
            }
            let angle = (dy as f32).atan2(dx as f32);
            let angle = if angle < 0.0 { angle + std::f32::consts::PI } else { angle };
            let bin = (angle / std::f32::consts::PI * ORIENT_BINS as f32) as usize;
            let bin = bin.min(ORIENT_BINS - 1);
            lut[idx_x][idx_y] = bin as u8;
        }
    }
    lut
});

/// 缩放图像至固定尺寸（最近邻）
fn resize_to_fixed(src: &GrayImage, dst_size: u32) -> GrayImage {
    let (src_w, src_h) = src.dimensions();
    let src_data = src.as_raw();
    let mut dst = GrayImage::new(dst_size, dst_size);
    let dst_data = dst.as_mut();  // 使用 as_mut() 代替 as_mut_raw()

    for y in 0..dst_size {
        let src_y = (y * src_h / dst_size) as usize;
        let src_row_start = src_y * src_w as usize;
        let dst_row_start = y as usize * dst_size as usize;
        for x in 0..dst_size {
            let src_x = (x * src_w / dst_size) as usize;
            let pixel = src_data[src_row_start + src_x];
            dst_data[dst_row_start + x as usize] = pixel;
        }
    }
    dst
}

/// 计算整数梯度和方向（Sobel 算子）
/// 返回 (mag, orient)，mag 为曼哈顿幅度，orient 为方向 0..7
fn compute_gradient_int(gray: &GrayImage) -> (Vec<u16>, Vec<u8>) {
    let w = gray.width() as usize;
    let h = gray.height() as usize;
    let data = gray.as_raw();
    let mut mag = vec![0u16; w * h];
    let mut orient = vec![0u8; w * h];

    // Sobel 算子系数
    const GX: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    const GY: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    // 内部区域
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut gx = 0i32;
            let mut gy = 0i32;
            for ky in 0..3 {
                let py = y + ky - 1;
                let row_start = py * w;
                for kx in 0..3 {
                    let px = x + kx - 1;
                    let pixel = data[row_start + px] as i32;
                    gx += GX[ky][kx] * pixel;
                    gy += GY[ky][kx] * pixel;
                }
            }
            let mag_val = (gx.abs() + gy.abs()) as u16;
            mag[y * w + x] = mag_val;

            // 方向查表：将 gx, gy 归一化到 [-4,4] 范围
            let idx_x = if gx == 0 {
                4
            } else {
                let ratio = ((gx.abs() * 8) / (gx.abs() + gy.abs())) as i8;
                let s = if gx > 0 { ratio } else { -ratio };
                s.clamp(-4, 4)
            };
            let idx_y = if gy == 0 {
                4
            } else {
                let ratio = ((gy.abs() * 8) / (gx.abs() + gy.abs())) as i8;
                let s = if gy > 0 { ratio } else { -ratio };
                s.clamp(-4, 4)
            };
            orient[y * w + x] = ORIENT_LUT[(idx_x + 4) as usize][(idx_y + 4) as usize];
        }
    }

    // 边界填充：复制最近邻
    for y in 0..h {
        for x in 0..w {
            if y == 0 || y == h - 1 || x == 0 || x == w - 1 {
                let ny = if y == 0 { 1 } else if y == h - 1 { h - 2 } else { y };
                let nx = if x == 0 { 1 } else if x == w - 1 { w - 2 } else { x };
                mag[y * w + x] = mag[ny * w + nx];
                orient[y * w + x] = orient[ny * w + nx];
            }
        }
    }

    (mag, orient)
}

/// 单元格直方图哈希（8位）
fn cell_histogram_hash(
    mag: &[u16],
    orient: &[u8],
    w: usize,
    cell_x: usize,
    cell_y: usize,
) -> u8 {
    let start_x = cell_x * CELL_W;
    let start_y = cell_y * CELL_H;
    let end_x = start_x + CELL_W;
    let end_y = start_y + CELL_H;

    let mut hist = [0usize; ORIENT_BINS];
    let mut total_weight = 0usize;

    for y in start_y..end_y {
        let mag_row = &mag[y * w..];
        let orient_row = &orient[y * w..];
        for x in start_x..end_x {
            let bin = orient_row[x] as usize;
            let m = mag_row[x] as usize;
            hist[bin] += m;
            total_weight += m;
        }
    }

    let avg = total_weight / ORIENT_BINS;
    let mut hash = 0u8;
    for (i, &cnt) in hist.iter().enumerate() {
        if cnt >= avg {
            hash |= 1 << i;
        }
    }
    hash
}

/// 计算局部梯度哈希（返回 16 字节数组）
fn local_gradient_hash(gray: &GrayImage) -> [u8; HASH_BYTES] {
    let (mag, orient) = compute_gradient_int(gray);
    let w = gray.width() as usize;

    let mut hash = [0u8; HASH_BYTES];
    for cy in 0..CELLS_Y {
        for cx in 0..CELLS_X {
            let idx = cy * CELLS_X + cx;
            hash[idx] = cell_histogram_hash(&mag, &orient, w, cx, cy);
        }
    }
    hash
}

/// 汉明距离（字节数组）
fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x ^ y).count_ones() as usize)
        .sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <image1> <image2>", args[0]);
        std::process::exit(1);
    }

    let path1 = std::path::Path::new(&args[1]);
    let path2 = std::path::Path::new(&args[2]);

    // 加载图像并转为灰度
    let img1 = image::open(path1)?.to_luma8();
    let img2 = image::open(path2)?.to_luma8();

    // 缩放
    let resized1 = resize_to_fixed(&img1, TARGET_SIZE);
    let resized2 = resize_to_fixed(&img2, TARGET_SIZE);

    // 哈希
    let hash1 = local_gradient_hash(&resized1);
    let hash2 = local_gradient_hash(&resized2);

    // 汉明距离
    let dist = hamming_distance(&hash1, &hash2);

    // 输出
    println!("Hash1: {}", hex::encode(&hash1));
    println!("Hash2: {}", hex::encode(&hash2));
    println!("Hamming distance: {} / {}", dist, HASH_BYTES * 8);

    Ok(())
}