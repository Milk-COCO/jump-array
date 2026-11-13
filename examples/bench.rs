//! 压力测试：`JumpArray`（块索引 + 连续数组）vs `排序 Vec + 二分`。
//!
//! ```text
//! cargo run --release --example bench            # 默认到 1e5
//! cargo run --release --example bench -- 1000000
//! ```
//!
//! 每个规模：升序构建、点查询（随机/顺序）、整表遍历、64 元素窗口遍历、
//! 随机插入、随机删除；5 轮取最小值。写入类负载按规模封顶（Vec 的搬移本身 O(n)）。

use std::hint::black_box;
use std::time::Instant;

use jump_array::JumpArray;

/// 参与对比的两种实现。
trait Store {
    fn new() -> Self
    where
        Self: Sized;
    fn insert(&mut self, k: i64, v: u32) -> Result<(), u32>;
    fn remove(&mut self, k: &i64) -> Option<u32>;
    fn find_le(&self, k: &i64) -> Option<(&i64, &u32)>;
    fn iter_all<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a i64, &'a u32)> + 'a>;
    fn range<'a>(&'a self, lo: i64, hi: i64) -> Box<dyn Iterator<Item = (&'a i64, &'a u32)> + 'a>;
    fn mem_bytes(&self) -> usize;
    /// 顺序滑窗求和（用 `range` 的无状态实现；有游标的实现覆盖它）。
    fn window_sum(&self, key: &dyn Fn(usize) -> i64, wins: usize, n: usize, win: usize) -> i64 {
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * win);
            let hi = key(((w + 1) * win).min(n - 1));
            for (k, _) in self.range(lo, hi) {
                acc = acc.wrapping_add(*k);
            }
        }
        acc
    }
    /// 批量写完之后调用。
    fn finalize(&mut self) {}
    fn label() -> &'static str;
}

struct VecSorted {
    v: Vec<(i64, u32)>,
}

impl Store for VecSorted {
    fn new() -> Self {
        Self { v: Vec::new() }
    }
    fn insert(&mut self, k: i64, v: u32) -> Result<(), u32> {
        let pos = self.v.partition_point(|(x, _)| *x < k);
        if pos < self.v.len() && self.v[pos].0 == k {
            return Err(v);
        }
        self.v.insert(pos, (k, v));
        Ok(())
    }
    fn remove(&mut self, k: &i64) -> Option<u32> {
        let pos = self.v.partition_point(|(x, _)| x < k);
        if pos < self.v.len() && self.v[pos].0 == *k {
            Some(self.v.remove(pos).1)
        } else {
            None
        }
    }
    fn find_le(&self, k: &i64) -> Option<(&i64, &u32)> {
        let pos = self.v.partition_point(|(x, _)| x <= k);
        if pos == 0 {
            None
        } else {
            let e = &self.v[pos - 1];
            Some((&e.0, &e.1))
        }
    }
    fn iter_all<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a i64, &'a u32)> + 'a> {
        Box::new(self.v.iter().map(|(k, v)| (k, v)))
    }
    fn range<'a>(&'a self, lo: i64, hi: i64) -> Box<dyn Iterator<Item = (&'a i64, &'a u32)> + 'a> {
        let a = self.v.partition_point(|(x, _)| *x < lo);
        let b = self.v.partition_point(|(x, _)| *x <= hi);
        Box::new(self.v[a..b].iter().map(|(k, v)| (k, v)))
    }
    fn mem_bytes(&self) -> usize {
        self.v.capacity() * std::mem::size_of::<(i64, u32)>()
    }
    /// 直接走切片（默认实现会经 `range` 的 `Box<dyn Iterator>`，每窗多一次分配）。
    fn window_sum(&self, key: &dyn Fn(usize) -> i64, wins: usize, n: usize, win: usize) -> i64 {
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * win);
            let hi = key(((w + 1) * win).min(n - 1));
            let a = self.v.partition_point(|(x, _)| *x < lo);
            let b = self.v.partition_point(|(x, _)| *x <= hi);
            for (k, _) in &self.v[a..b] {
                acc = acc.wrapping_add(*k);
            }
        }
        acc
    }
    fn label() -> &'static str {
        "vec+二分"
    }
}

struct Jump(JumpArray<i64, u32>);

impl Store for Jump {
    fn new() -> Self {
        Self(JumpArray::new())
    }
    fn insert(&mut self, k: i64, v: u32) -> Result<(), u32> {
        self.0.insert(k, v)
    }
    fn remove(&mut self, k: &i64) -> Option<u32> {
        self.0.remove(k)
    }
    fn find_le(&self, k: &i64) -> Option<(&i64, &u32)> {
        self.0.find_le(k)
    }
    fn iter_all<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a i64, &'a u32)> + 'a> {
        Box::new(self.0.iter())
    }
    fn range<'a>(&'a self, lo: i64, hi: i64) -> Box<dyn Iterator<Item = (&'a i64, &'a u32)> + 'a> {
        Box::new(self.0.range(lo, hi))
    }
    fn mem_bytes(&self) -> usize {
        let (blocks, len, _) = self.0.stats();
        len * std::mem::size_of::<(i64, u32)>() + blocks * 4
    }
    fn window_sum(&self, key: &dyn Fn(usize) -> i64, wins: usize, n: usize, win: usize) -> i64 {
        let mut cur = self.0.cursor();
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * win);
            let hi = key(((w + 1) * win).min(n - 1));
            for (k, _) in self.0.range_from(&mut cur, lo, hi) {
                acc = acc.wrapping_add(*k);
            }
        }
        acc
    }
    fn finalize(&mut self) {
        self.0.rebuild();
    }
    fn label() -> &'static str {
        "jump"
    }
}

struct JumpTuned<const LM: usize>(JumpArray<i64, u32>);

impl<const LM: usize> Store for JumpTuned<LM> {
    fn new() -> Self {
        Self(JumpArray::new().with_linear_max(LM))
    }
    fn insert(&mut self, k: i64, v: u32) -> Result<(), u32> {
        self.0.insert(k, v)
    }
    fn remove(&mut self, k: &i64) -> Option<u32> {
        self.0.remove(k)
    }
    fn find_le(&self, k: &i64) -> Option<(&i64, &u32)> {
        self.0.find_le(k)
    }
    fn iter_all<'a>(&'a self) -> Box<dyn Iterator<Item = (&'a i64, &'a u32)> + 'a> {
        Box::new(self.0.iter())
    }
    fn range<'a>(&'a self, lo: i64, hi: i64) -> Box<dyn Iterator<Item = (&'a i64, &'a u32)> + 'a> {
        Box::new(self.0.range(lo, hi))
    }
    fn mem_bytes(&self) -> usize {
        let (blocks, len, _) = self.0.stats();
        len * std::mem::size_of::<(i64, u32)>() + blocks * 4
    }
    fn window_sum(&self, key: &dyn Fn(usize) -> i64, wins: usize, n: usize, win: usize) -> i64 {
        let mut cur = self.0.cursor();
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * win);
            let hi = key(((w + 1) * win).min(n - 1));
            for (k, _) in self.0.range_from(&mut cur, lo, hi) {
                acc = acc.wrapping_add(*k);
            }
        }
        acc
    }
    fn finalize(&mut self) {
        self.0.rebuild();
    }
    fn label() -> &'static str {
        match LM {
            1 => "jump-lm1",
            2 => "jump-lm2",
            4 => "jump-lm4",
            8 => "jump-lm8",
            16 => "jump-lm16",
            32 => "jump-lm32",
            64 => "jump-lm64",
            _ => "jump-lm?",
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

const ROUNDS: usize = 5;

/// 固定 CPU 负载耗时（毫秒）：降频/被抢占时会显著变长，用来剔除坏轮次。
fn cpu_probe_ms() -> f64 {
    let t = Instant::now();
    let mut x = 0u64;
    for i in 0..2_000_000u64 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
    }
    black_box(x);
    t.elapsed().as_secs_f64() * 1e3
}

/// 每轮先测一次 probe；只保留 probe ≤ 最优 probe × 1.2 的轮次，取中位数。
///
/// 只取 `min` 会让"偶发一次极快"主导结果；只取中位数又可能被降频轮次拖偏，
/// 所以先按 probe 过滤再取中位数。
fn bench(mut f: impl FnMut()) -> f64 {
    let mut samples: Vec<(f64, f64)> = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let probe = cpu_probe_ms();
        let t = Instant::now();
        f();
        samples.push((probe, t.elapsed().as_secs_f64()));
    }
    let best_probe = samples.iter().map(|(p, _)| *p).fold(f64::MAX, f64::min);
    let mut times: Vec<f64> = samples
        .iter()
        .filter(|(p, _)| *p <= best_probe * 1.2)
        .map(|(_, e)| *e)
        .collect();
    if times.is_empty() {
        times = samples.iter().map(|(_, e)| *e).collect();
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

struct Metrics {
    build_ms: f64,
    mem_mb: f64,
    rand_ns: f64,
    seq_ns: f64,
    iter_ns: f64,
    win_ns: f64,
    wincur_ns: f64,
    ins_ns: f64,
    del_ns: f64,
}

fn measure<S: Store>(n: usize, key: impl Fn(usize) -> i64, win: usize) -> Metrics {
    let build_ms = bench(|| {
        let mut s = S::new();
        for i in 0..n {
            s.insert(key(i), i as u32).ok();
        }
        s.finalize();
        black_box(s.mem_bytes());
    }) * 1e3;

    let mut store = S::new();
    for i in 0..n {
        store.insert(key(i), i as u32).ok();
    }
    store.finalize();
    let mem_mb = store.mem_bytes() as f64 / 1e6;

    let probes = (n * 2).clamp(10_000, 2_000_000);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let qs: Vec<i64> = (0..probes)
        .map(|_| (rng.below((n as u64) * 997 + 2000)) as i64)
        .collect();
    let rand_ns = bench(|| {
        let mut acc = 0i64;
        for q in &qs {
            if let Some((k, _)) = store.find_le(q) {
                acc = acc.wrapping_add(*k);
            }
        }
        black_box(acc);
    }) * 1e9
        / probes as f64;

    let seq_ns = bench(|| {
        let mut acc = 0i64;
        for i in 0..n {
            if let Some((k, _)) = store.find_le(&key(i)) {
                acc = acc.wrapping_add(*k);
            }
        }
        black_box(acc);
    }) * 1e9
        / n as f64;

    let iter_ns = bench(|| {
        let mut acc = 0u64;
        for (k, v) in store.iter_all() {
            acc = acc.wrapping_add(*k as u64).wrapping_add(*v as u64);
        }
        black_box(acc);
    }) * 1e9
        / n as f64;

    let wins = (n / win).max(1);
    let win_ns = bench(|| {
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * win);
            let hi = key(((w + 1) * win).min(n - 1));
            for (k, _) in store.range(lo, hi) {
                acc = acc.wrapping_add(*k);
            }
        }
        black_box(acc);
    }) * 1e9
        / wins as f64;

    // 顺序滑窗（有游标的实现走 `range_from`，其余等价于上面的 range）。
    let wincur_ns = bench(|| {
        black_box(store.window_sum(&key, wins, n, win));
    }) * 1e9
        / wins as f64;

    let add = (n / 10).clamp(100, 2000);
    let mut rng = Rng(0xdead_beef_1234_5678);
    let inserts: Vec<i64> = (0..add)
        .map(|_| (rng.below((n as u64) * 997 * 2)) as i64)
        .collect();
    // 构建放在计时之外：否则 `ins_ns`/`del_ns` 里混着 O(n) 的建表成本，
    // 大 n 时会被误读成"单次插入很慢"。
    let mut best_ins = f64::MAX;
    for _ in 0..ROUNDS {
        let mut s = S::new();
        for i in 0..n {
            s.insert(key(i), i as u32).ok();
        }
        s.finalize();
        let t = Instant::now();
        for (j, k) in inserts.iter().enumerate() {
            s.insert(*k, j as u32).ok();
        }
        black_box(s.mem_bytes());
        best_ins = best_ins.min(t.elapsed().as_secs_f64());
    }
    let ins_ns = best_ins * 1e9 / add as f64;

    let removes: Vec<i64> = inserts.clone();
    let mut best_del = f64::MAX;
    for _ in 0..ROUNDS {
        let mut s = S::new();
        for i in 0..n {
            s.insert(key(i), i as u32).ok();
        }
        let t = Instant::now();
        for k in &removes {
            s.remove(k);
        }
        black_box(s.mem_bytes());
        best_del = best_del.min(t.elapsed().as_secs_f64());
    }
    let del_ns = best_del * 1e9 / removes.len() as f64;

    Metrics {
        build_ms,
        mem_mb,
        rand_ns,
        seq_ns,
        iter_ns,
        win_ns,
        wincur_ns,
        ins_ns,
        del_ns,
    }
}

fn print_row<S: Store, F: Fn(usize) -> i64 + Copy>(n: usize, key: F, win: usize) {
    let m = measure::<S>(n, key, win);
    println!(
        "  {:<10} build {:>8.2}ms | mem {:>5.2}MB | find_le rand {:>6.1}ns seq {:>5.1}ns | iter {:>5.2}ns/e | win {:>6.1}ns | win-cur {:>6.1}ns | ins {:>9.1}ns | del {:>9.1}ns",
        S::label(),
        m.build_ms,
        m.mem_mb,
        m.rand_ns,
        m.seq_ns,
        m.iter_ns,
        m.win_ns,
        m.wincur_ns,
        m.ins_ns,
        m.del_ns
    );
}

fn main() {
    let mut max_n = 100_000usize;
    let mut win = 64usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--win" => {
                win = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .expect("--win 需要数字");
            }
            other => {
                max_n = other.parse().expect("位置参数是最大规模 n");
            }
        }
    }
    win = win.max(1);
    println!("# JumpArray vs Vec+二分（probe 过滤 + 中位数，窗口 {win} 元素）");
    for n in [1_000usize, 10_000, 100_000, 1_000_000] {
        if n > max_n {
            break;
        }
        // ① 稀疏均匀：步长 997（键域远大于元素数）。
        println!("\n## n = {n}  稀疏均匀（step=997）");
        let sparse = |i: usize| (i as i64) * 997;
        print_row::<VecSorted, _>(n, sparse, win);
        print_row::<Jump, _>(n, sparse, win);

        // ② 聚簇：每组 100 个键密集排布（间距 1），组间跳 100_000。
        println!("## n = {n}  聚簇（100 键/组，组间跳 100_000）");
        let clustered = |i: usize| ((i / 100) as i64) * 100_000 + (i % 100) as i64;
        print_row::<VecSorted, _>(n, clustered, win);
        print_row::<Jump, _>(n, clustered, win);
    }

    // ③ 阈值扫描：同一个 JumpArray，只改"块内线性扫上限"，看查询/窗口的拐点。
    let n = max_n;
    let sparse = |i: usize| (i as i64) * 997;
    let clustered = |i: usize| ((i / 100) as i64) * 100_000 + (i % 100) as i64;
    println!("\n## 阈值扫描 n = {n}  稀疏均匀");
    print_row::<JumpTuned<1>, _>(n, sparse, win);
    print_row::<JumpTuned<2>, _>(n, sparse, win);
    print_row::<JumpTuned<4>, _>(n, sparse, win);
    print_row::<JumpTuned<8>, _>(n, sparse, win);
    print_row::<JumpTuned<16>, _>(n, sparse, win);
    print_row::<JumpTuned<32>, _>(n, sparse, win);
    print_row::<JumpTuned<64>, _>(n, sparse, win);
    println!("\n## 阈值扫描 n = {n}  聚簇");
    print_row::<JumpTuned<1>, _>(n, clustered, win);
    print_row::<JumpTuned<2>, _>(n, clustered, win);
    print_row::<JumpTuned<4>, _>(n, clustered, win);
    print_row::<JumpTuned<8>, _>(n, clustered, win);
    print_row::<JumpTuned<16>, _>(n, clustered, win);
    print_row::<JumpTuned<32>, _>(n, clustered, win);
    print_row::<JumpTuned<64>, _>(n, clustered, win);

    // ④ 拆解窗口成本：定位（pos）/ 迭代（iter）/ 合计（full），ns/窗口。
    println!(
        "\n## 窗口拆解 n = {n}（窗口 {win}；ns/窗口：pos=只定位 iter=只迭代 full=合计）"
    );
    window_split(n, "稀疏均匀", sparse, win);
    window_split(n, "聚簇    ", clustered, win);

    // ⑤ 块宽倍数扫描：块表更小（更可能进缓存）vs 块内扫更多。
    println!("\n## 块宽倍数扫描 n = {n}（scale=1 是平均每块一个元素）");
    width_scale_sweep(n, "稀疏均匀", sparse, win);
    width_scale_sweep(n, "聚簇    ", clustered, win);
}

fn width_scale_sweep(n: usize, label: &str, key: impl Fn(usize) -> i64 + Copy, win: usize) {
    let wins = (n / win).max(1);
    let probes = 100_000usize.min(n.saturating_mul(10)).max(1_000);
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let qs: Vec<i64> = (0..probes)
        .map(|_| rng.below((n as u64) * 997 * 2) as i64)
        .collect();
    for scale in [0.5f64, 1.0, 2.0, 4.0, 8.0] {
        let mut a = JumpArray::<i64, u32>::new().with_width_scale(scale);
        for i in 0..n {
            a.insert(key(i), i as u32).ok();
        }
        a.rebuild();
        let (blocks, _, width) = a.stats();
        let rand = bench(|| {
            let mut acc = 0i64;
            for q in &qs {
                if let Some((k, _)) = a.find_le(q) {
                    acc = acc.wrapping_add(*k);
                }
            }
            black_box(acc);
        }) * 1e9
            / probes as f64;
        let wincur = bench(|| {
            let mut cur = a.cursor();
            let mut acc = 0i64;
            for w in 0..wins {
                let lo = key(w * win);
                let hi = key(((w + 1) * win).min(n - 1));
                for (k, _) in a.range_from(&mut cur, lo, hi) {
                    acc = acc.wrapping_add(*k);
                }
            }
            black_box(acc);
        }) * 1e9
            / wins as f64;
        println!(
            "  {label} scale {scale:>4} blocks {blocks:>7} W {width:>12.0} | rand {rand:>6.1}ns | win-cur {wincur:>6.1}ns"
        );
    }
}

/// 把"定位"和"迭代"分开计时：两边都按同样的 `win` 元素窗口序列。
fn window_split(n: usize, label: &str, key: impl Fn(usize) -> i64 + Copy, win: usize) {
    let wins = (n / win).max(1);
    let mut j = JumpArray::<i64, u32>::new();
    let mut v: Vec<(i64, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        let k = key(i);
        j.insert(k, i as u32).ok();
        v.push((k, i as u32));
    }
    j.rebuild();

    // Jump：先预取 spans（用于 iter-only），再分别计时。
    let mut cur = j.cursor();
    let mut spans = Vec::with_capacity(wins);
    for w in 0..wins {
        let lo = key(w * win);
        let hi = key(((w + 1) * win).min(n - 1));
        spans.push(j.range_from(&mut cur, lo, hi).span());
    }

    let pos_j = bench(|| {
        let mut cur = j.cursor();
        let mut acc = 0usize;
        for w in 0..wins {
            let lo = key(w * win);
            let hi = key(((w + 1) * win).min(n - 1));
            let (a, b) = j.range_from(&mut cur, lo, hi).span();
            acc = acc.wrapping_add(a ^ b);
        }
        black_box(acc);
    }) * 1e9
        / wins as f64;
    let iter_j = bench(|| {
        let mut acc = 0i64;
        for &sp in &spans {
            for (k, _) in j.span_iter(sp) {
                acc = acc.wrapping_add(*k);
            }
        }
        black_box(acc);
    }) * 1e9
        / wins as f64;
    let full_j = bench(|| {
        let mut cur = j.cursor();
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * win);
            let hi = key(((w + 1) * win).min(n - 1));
            for (k, _) in j.range_from(&mut cur, lo, hi) {
                acc = acc.wrapping_add(*k);
            }
        }
        black_box(acc);
    }) * 1e9
        / wins as f64;

    // Vec：同样拆两半。
    let mut vspans = Vec::with_capacity(wins);
    for w in 0..wins {
        let lo = key(w * win);
        let hi = key(((w + 1) * win).min(n - 1));
        vspans.push((
            v.partition_point(|(x, _)| *x < lo),
            v.partition_point(|(x, _)| *x <= hi),
        ));
    }
    let pos_v = bench(|| {
        let mut acc = 0usize;
        for w in 0..wins {
            let lo = key(w * win);
            let hi = key(((w + 1) * win).min(n - 1));
            let a = v.partition_point(|(x, _)| *x < lo);
            let b = v.partition_point(|(x, _)| *x <= hi);
            acc = acc.wrapping_add(a ^ b);
        }
        black_box(acc);
    }) * 1e9
        / wins as f64;
    let iter_v = bench(|| {
        let mut acc = 0i64;
        for &(a, b) in &vspans {
            for (k, _) in &v[a..b] {
                acc = acc.wrapping_add(*k);
            }
        }
        black_box(acc);
    }) * 1e9
        / wins as f64;
    let full_v = bench(|| {
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * win);
            let hi = key(((w + 1) * win).min(n - 1));
            let a = v.partition_point(|(x, _)| *x < lo);
            let b = v.partition_point(|(x, _)| *x <= hi);
            for (k, _) in &v[a..b] {
                acc = acc.wrapping_add(*k);
            }
        }
        black_box(acc);
    }) * 1e9
        / wins as f64;

    println!(
        "  {label} | jump pos {:6.1} iter {:6.1} full {:6.1} | vec pos {:6.1} iter {:6.1} full {:6.1}",
        pos_j, iter_j, full_j, pos_v, iter_v, full_v
    );
}
