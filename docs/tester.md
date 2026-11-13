# 委托测试说明（JumpArray 性能对比）

> 目的：在一台**不同于开发机**的设备上跑 `examples/bench.rs`，把输出发回。
> 只跑基准，不需要改代码、不需要网络、不需要任何数据文件。

## 方式 A：直接跑 exe（推荐）

1. 收到 `bench.exe`（约 0.5 MB），放到任意目录，例如 `C:\bench\`。
2. 需要 **Windows x64**。若启动报“缺少 VCRUNTIME140.dll”，装一次
   **Microsoft Visual C++ 2015–2022 Redistributable (x64)** 后重试。
3. 笔记本接电源、电源计划设“高性能”；关闭浏览器/聊天工具/IDE，跑的时候别做别的重活。
4. 打开 PowerShell，执行（每条约 4–6 分钟）：

```powershell
cd C:\bench
.\bench.exe 4000000 | Tee-Object -FilePath bench_4m.txt
.\bench.exe 4000000 --win 8 | Tee-Object -FilePath bench_4m_win8.txt
```

5. 把 `bench_4m.txt`、`bench_4m_win8.txt` 发回，并附上：
   - CPU 型号（`Get-CimInstance Win32_Processor | Select-Object Name`）
   - 内存大小、操作系统版本

> 说明：程序会打印多段结果（主表、阈值扫描、窗口拆解、块宽扫描、小数组阈值扫描）。
> 全部原样发回即可，不需要筛选。

## 方式 B：从源码跑（需要有 Rust 环境）

```powershell
git clone https://github.com/Milk-COCO/jump-array
cd jump-array
cargo run --release --example bench -- 4000000 | Tee-Object -FilePath bench_4m.txt
cargo run --release --example bench -- 4000000 --win 8 | Tee-Object -FilePath bench_4m_win8.txt
```

需要 Rust 稳定版（edition 2024 ⇒ 1.85+）与 MSVC Build Tools（`link.exe`）。

## 输出怎么读（不用分析，原样发回即可）

- `## n = …` 各段的 `vec+二分` 与 `jump` 两行：同一轮交替测量的结果；
- `binary_below` / `scale` 等扫描段用于确认拐点；
- 两次运行（`--win 64` 与 `--win 8`）的差异用于看“窗口大小”对定位成本的影响。

## 注意

- 设备降频会让绝对值整体变慢；同一命令两次运行差 1–2 倍属正常，**不要只跑一次就下结论**，
  如果时间允许，每个命令跑两次、都发回。
- 程序只分配内存（4M 规模约 160 MB），不写磁盘（除你重定向的输出文件）、不联网。
