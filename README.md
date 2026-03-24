# hash
## 这是什么
该项目基于RUST环境开发 旨在通过梯度哈希来判断两张图片是否相似
## 编译
在下载代码后 你还需要在工作目录下执行 
```bash
cargo build --release
```
来完成编译
## 用法
你只需要输入 `./target/release/img_hash 图片1路径 图片2路径` 即可完成计算
## 调试
为获取更高精度 你可以在main.rs中修改CELLS_X和CELLS_Y值并重新编译即可
## 许可证
GPL-3.0