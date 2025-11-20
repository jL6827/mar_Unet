```markdown
# UNet for ocean velocity prediction (example)

说明：
- 假设输入文件 `processed_data_mean.csv` 在当前目录，CSV 必须含下列列：
  time（ISO 日期或时间戳，可解析为 pandas datetime），lon, lat, depth, uo, vo, so, thetao
- 代码把散点插值到水平规则网格 (nx,ny)，并把 depth 分箱（depth_bins）构成样本。
- 运行示例：
  python train_unet.py --csv processed_data_mean.csv --nx 128 --ny 128 --ndepths 6 --epochs 40
- 如需预测模型参数（例如 Ro, M11 等），需要在 CSV 中包含这些字段或用另一个标签源；当前脚本示例将 uo, vo 作为目标，也演示如何扩展输出通道以预测额外参数。

依赖参考见 requirements.txt
```