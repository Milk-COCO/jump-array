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
    fn window_sum(&self, key: &dyn Fn(usize) -> i64, wins: usize, n: usize) -> i64 {
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * 64);
            let hi = key(((w + 1) * 64).min(n - 1));
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
    fn window_sum(&self, key: &dyn Fn(usize) -> i64, wins: usize, n: usize) -> i64 {
        let mut cur = self.0.cursor();
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * 64);
            let hi = key(((w + 1) * 64).min(n - 1));
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

fn bench(mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..ROUNDS {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
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

fn measure<S: Store>(n: usize, key: impl Fn(usize) -> i64) -> Metrics {
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

    let wins = (n / 64).max(1);
    let win_ns = bench(|| {
        let mut acc = 0i64;
        for w in 0..wins {
            let lo = key(w * 64);
            let hi = key(((w + 1) * 64).min(n - 1));
            for (k, _) in store.range(lo, hi) {
                acc = acc.wrapping_add(*k);
            }
        }
        black_box(acc);
    }) * 1e9
        / wins as f64;

    // 顺序滑窗（有游标的实现走 `range_from`，其余等价于上面的 range）。
    let wincur_ns = bench(|| {
        black_box(store.window_sum(&key, wins, n));
    }) * 1e9
        / wins as f64;

    let add = (n / 10).clamp(100, 2000);
    let mut rng = Rng(0xdead_beef_1234_5678);
    let inserts: Vec<i64> = (0..add)
        .map(|_| (rng.below((n as u64) * 997 * 2)) as i64)
        .collect();
    let ins_ns = bench(|| {
        let mut s = S::new();
        for i in 0..n {
            s.insert(key(i), i as u32).ok();
        }
        for (j, k) in inserts.iter().enumerate() {
            s.insert(*k, j as u32).ok();
        }
        s.finalize();
        black_box(s.mem_bytes());
    }) * 1e9
        / add as f64;

    let removes: Vec<i64> = inserts.clone();
    let del_ns = bench(|| {
        let mut s = S::new();
        for i in 0..n {
            s.insert(key(i), i as u32).ok();
        }
        for k in &removes {
            s.remove(k);
        }
        black_box(s.mem_bytes());
    }) * 1e9
        / removes.len() as f64;

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

fn print_row<S: Store, F: Fn(usize) -> i64 + Copy>(n: usize, key: F) {
    let m = measure::<S>(n, key);
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
    let max_n = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100_000);
    println!("# JumpArray vs Vec+二分（5 轮最小值）");
    for n in [1_000usize, 10_000, 100_000, 1_000_000] {
        if n > max_n {
            break;
        }
        // ① 稀疏均匀：步长 997（键域远大于元素数）。
        println!("\n## n = {n}  稀疏均匀（step=997）");
        let sparse = |i: usize| (i as i64) * 997;
        print_row::<VecSorted, _>(n, sparse);
        print_row::<Jump, _>(n, sparse);

        // ② 聚簇：每组 100 个键密集排布（间距 1），组间跳 100_000。
        println!("## n = {n}  聚簇（100 键/组，组间跳 100_000）");
        let clustered = |i: usize| ((i / 100) as i64) * 100_000 + (i % 100) as i64;
        print_row::<VecSorted, _>(n, clustered);
        print_row::<Jump, _>(n, clustered);
    }

    // ③ 阈值扫描：同一个 JumpArray，只改"块内线性扫上限"，看查询/窗口的拐点。
    let n = max_n;
    let sparse = |i: usize| (i as i64) * 997;
    let clustered = |i: usize| ((i / 100) as i64) * 100_000 + (i % 100) as i64;
    println!("\n## 阈值扫描 n = {n}  稀疏均匀");
    print_row::<JumpTuned<1>, _>(n, sparse);
    print_row::<JumpTuned<2>, _>(n, sparse);
    print_row::<JumpTuned<4>, _>(n, sparse);
    print_row::<JumpTuned<8>, _>(n, sparse);
    print_row::<JumpTuned<16>, _>(n, sparse);
    print_row::<JumpTuned<32>, _>(n, sparse);
    print_row::<JumpTuned<64>, _>(n, sparse);
    println!("\n## 阈值扫描 n = {n}  聚簇");
    print_row::<JumpTuned<1>, _>(n, clustered);
    print_row::<JumpTuned<2>, _>(n, clustered);
    print_row::<JumpTuned<4>, _>(n, clustered);
    print_row::<JumpTuned<8>, _>(n, clustered);
    print_row::<JumpTuned<16>, _>(n, clustered);
    print_row::<JumpTuned<32>, _>(n, clustered);
    print_row::<JumpTuned<64>, _>(n, clustered);
}
