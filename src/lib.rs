//! Rusty JumpArray：kipphi 的"块索引 + 短走步"思想，落在**单块连续数组**上。
//!
//! ## 结构
//!
//! ```text
//! entries: Vec<(K, E)>   按键升序、连续存放
//! blocks:  Vec<u32>      块起点 → 第一个 key ≥ 块起点的条目下标
//! ```
//!
//! * **块宽** `W = 2^ceil(log2(span/n))` ⇒ 块数 ≈ 元素数，与键的数值大小无关
//!   （稀疏键不膨胀；这是相对"按数值单位分槽"的关键差异）；
//! * **查询**：`b = floor((key - base)/W)` → `blocks[b]` → 从该下标**向前走几步**
//!   （均匀数据 ~1 项）到精确位置。索引有效时是 **2 次依赖加载**；
//! * **区间遍历**：索引起跳后直接切连续切片，无指针追逐；
//! * **写入**：数组搬移 + 标脏。批量写完之后调一次 [`JumpArray::rebuild`] 重建块表
//!   （O(n)）。这是刻意的取舍：不做逐次插入/删除的增量维护，换取查询与遍历的简单性和
//!   缓存友好——"频繁增删"不是本容器的目标形态；
//! * **脏索引**：`rebuild` 之前查询自动退回二分（`O(log n)`），结果仍然正确。
//!
//! ## 与 kipphi 原版的差异
//!
//! 1. 数据载体是连续数组而不是双向链表：区间遍历与短走步都是顺序读；
//! 2. 索引不是 `T | T[]` union，而是扁平 `u32` 下标，无 per-block 分配；
//! 3. 没有游标/`nextFn`：短走步就是在数组上递增下标；
//! 4. 索引重建是显式的（批量写后一次），不做 `updateRange` 式增量修补。
//!
//! ## 复杂度
//!
//! | 操作 | 复杂度 |
//! |---|---|
//! | 写入 | O(n) 搬移（数组代价）+ 标脏 |
//! | `rebuild` | O(n) |
//! | 点查询（索引有效） | O(1) 起跳 + 块内短扫 |
//! | 点查询（脏） | O(log n) |
//! | 区间 / 遍历 | O(长度)，连续切片 |
//! | 内存 | 载荷 + 4 B/元素 |

use std::cmp::Ordering;

/// 有序键：全序比较 + 数值化（块下标计算用）。
pub trait Key: Copy + std::fmt::Debug {
    fn cmp_key(&self, other: &Self) -> Ordering;
    fn to_f64(self) -> f64;
}

macro_rules! impl_int_key {
    ($($t:ty),*) => {$(
        impl Key for $t {
            #[inline] fn cmp_key(&self, other: &Self) -> Ordering { self.cmp(other) }
            #[inline] fn to_f64(self) -> f64 { self as f64 }
        }
    )*};
}
impl_int_key!(i32, i64, u32);

impl Key for f64 {
    #[inline]
    fn cmp_key(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// 有序容器（key 唯一、按 key 升序）。
pub struct JumpArray<K: Key, E> {
    entries: Vec<(K, E)>,
    /// 块起点 → 第一个 `key >= block_start` 的条目下标。
    blocks: Vec<u32>,
    base: f64,
    width: f64,
    /// 索引过期（写入后置脏，`rebuild` 清除）。
    dirty: bool,
    /// 块内元素数不超过它时线性扫，超过则块内二分（可调）。
    linear_max: usize,
}

/// `linear_max` 默认值。
const DEFAULT_LINEAR_MAX: usize = 8;

impl<K: Key, E> Default for JumpArray<K, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, E> JumpArray<K, E> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            blocks: Vec::new(),
            base: 0.0,
            width: 1.0,
            dirty: true,
            linear_max: DEFAULT_LINEAR_MAX,
        }
    }

    /// 调整"块内线性扫"阈值（块内元素数超过它则块内二分）。
    pub fn with_linear_max(mut self, linear_max: usize) -> Self {
        self.linear_max = linear_max.max(1);
        self
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut s = Self::new();
        s.entries.reserve(capacity);
        s
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 索引诊断：`(块表长度, 元素数, 块宽)`。
    pub fn stats(&self) -> (usize, usize, f64) {
        (self.blocks.len(), self.entries.len(), self.width)
    }

    /// 索引是否被写入弄脏：`true` ⇒ 查询退回二分（仍正确，只是多 log n 次比较），
    /// 需要 [`JumpArray::rebuild`] 恢复"块表起跳 + 短扫"的快路径。
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn first(&self) -> Option<(&K, &E)> {
        self.entries.first().map(|(k, v)| (k, v))
    }

    pub fn last(&self) -> Option<(&K, &E)> {
        self.entries.last().map(|(k, v)| (k, v))
    }

    /// 按 key 升序遍历。
    pub fn iter(&self) -> impl Iterator<Item = (&K, &E)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    /// 重建块表（批量写完之后调一次，恢复 O(1) 查询）。
    ///
    /// 幂等：索引已经是干净的（`!dirty`）时直接返回，不重复 O(n) 扫描。
    pub fn rebuild(&mut self) {
        if !self.dirty {
            return;
        }
        self.blocks.clear();
        if self.entries.is_empty() {
            self.base = 0.0;
            self.width = 1.0;
            self.dirty = false;
            return;
        }
        let first = self.entries[0].0.to_f64();
        let last = self.entries[self.entries.len() - 1].0.to_f64();
        let span = (last - first).max(0.0);
        let width = pow2_ceil(span / self.entries.len().max(1) as f64);
        self.base = first;
        self.width = width;
        let nblocks = ((span / width).ceil() as usize).saturating_add(1).max(1);
        self.blocks.reserve(nblocks);
        let mut i = 0usize;
        for b in 0..nblocks {
            let start = first + b as f64 * width;
            while i < self.entries.len() && self.entries[i].0.to_f64() < start {
                i += 1;
            }
            self.blocks.push(i as u32);
        }
        self.dirty = false;
    }

    #[inline]
    fn block_of(&self, k: &K) -> usize {
        if self.blocks.is_empty() {
            return 0;
        }
        let x = (k.to_f64() - self.base) / self.width;
        if !x.is_finite() || x <= 0.0 {
            0
        } else {
            let b = x.floor();
            if b >= self.blocks.len() as f64 {
                self.blocks.len() - 1
            } else {
                b as usize
            }
        }
    }

    /// 第一个 `key >= k` 的下标（二分；写入与脏索引路径用）。
    fn lower_bound(&self, k: &K) -> usize {
        self.entries
            .partition_point(|(x, _)| x.cmp_key(k) == Ordering::Less)
    }

    fn upper_bound(&self, k: &K) -> usize {
        self.entries
            .partition_point(|(x, _)| x.cmp_key(k) != Ordering::Greater)
    }

    /// 块表给出的 `[start, end)`：该块覆盖的元素下标区间。
    ///
    /// 空块时 `start == end`（两个锚点相同）。
    #[inline]
    fn block_span(&self, b: usize) -> (usize, usize) {
        let start = self.blocks[b] as usize;
        let end = if b + 1 < self.blocks.len() {
            self.blocks[b + 1] as usize
        } else {
            self.entries.len()
        };
        (start, end)
    }

    /// 索引起跳 + 块内短扫（块内元素多则二分）。
    #[inline]
    fn lower_bound_fast(&self, k: &K) -> usize {
        if self.dirty || self.blocks.is_empty() {
            return self.lower_bound(k);
        }
        let (start, end) = self.block_span(self.block_of(k));
        if end - start <= self.linear_max {
            let mut i = start;
            while i < end && self.entries[i].0.cmp_key(k) == Ordering::Less {
                i += 1;
            }
            i
        } else {
            start
                + self.entries[start..end]
                    .partition_point(|(x, _)| x.cmp_key(k) == Ordering::Less)
        }
    }

    /// 同上，取"第一个 `key > k`"。
    #[inline]
    fn upper_bound_fast(&self, k: &K) -> usize {
        if self.dirty || self.blocks.is_empty() {
            return self.upper_bound(k);
        }
        let (start, end) = self.block_span(self.block_of(k));
        if end - start <= self.linear_max {
            let mut i = start;
            while i < end && self.entries[i].0.cmp_key(k) != Ordering::Greater {
                i += 1;
            }
            i
        } else {
            start
                + self.entries[start..end]
                    .partition_point(|(x, _)| x.cmp_key(k) != Ordering::Greater)
        }
    }

    /// 插入；key 已存在时返回 `Err(value)`。写入后需 `rebuild` 才恢复 O(1) 查询。
    pub fn insert(&mut self, key: K, value: E) -> Result<(), E> {
        let pos = self.lower_bound(&key);
        if pos < self.entries.len() && self.entries[pos].0.cmp_key(&key) == Ordering::Equal {
            return Err(value);
        }
        self.entries.insert(pos, (key, value));
        self.dirty = true;
        Ok(())
    }

    /// 删除；不存在返回 `None`。
    pub fn remove(&mut self, key: &K) -> Option<E> {
        let pos = self.lower_bound(key);
        if pos >= self.entries.len() || self.entries[pos].0.cmp_key(key) != Ordering::Equal {
            return None;
        }
        let (_, v) = self.entries.remove(pos);
        self.dirty = true;
        Some(v)
    }

    pub fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    pub fn get(&self, key: &K) -> Option<&E> {
        self.find_le(key)
            .filter(|(k, _)| k.cmp_key(key) == Ordering::Equal)
            .map(|(_, v)| v)
    }

    /// 最后一个 `key <= k`。
    pub fn find_le(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.upper_bound_fast(k);
        if i == 0 {
            None
        } else {
            let (key, v) = &self.entries[i - 1];
            Some((key, v))
        }
    }

    /// 最后一个 `key < k`。
    pub fn find_lt(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.lower_bound_fast(k);
        if i == 0 {
            None
        } else {
            let (key, v) = &self.entries[i - 1];
            Some((key, v))
        }
    }

    /// 第一个 `key >= k`。
    pub fn find_ge(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.lower_bound_fast(k);
        self.entries.get(i).map(|(key, v)| (key, v))
    }

    /// 第一个 `key > k`。
    pub fn find_gt(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.upper_bound_fast(k);
        self.entries.get(i).map(|(key, v)| (key, v))
    }

    /// `[lo, hi]` 区间（含两端、升序、连续切片）。
    ///
    /// 上界优先从 `from` **向前线性探**（窗口小的时候最便宜）；探到扫描上限还没
    /// 越过 `hi` 才回退到 `upper_bound_fast`（块内二分/二分）。
    pub fn range(&self, lo: K, hi: K) -> RangeIter<'_, K, E> {
        if self.is_empty() || lo.cmp_key(&hi) == Ordering::Greater {
            return RangeIter { slice: &[] };
        }
        let from = self.range_lower_bound(&lo);
        let n = self.entries.len();
        let cap = (self.linear_max * 8).max(64);
        let mut to = from;
        while to < n && to - from < cap && self.entries[to].0.cmp_key(&hi) != Ordering::Greater {
            to += 1;
        }
        if to - from >= cap && to < n && self.entries[to].0.cmp_key(&hi) != Ordering::Greater {
            to = self.upper_bound_fast(&hi).max(from);
        }
        RangeIter {
            slice: &self.entries[from..to],
        }
    }

    /// `range` 的下界：先线性探 `PROBE_MAX` 步（窗口起点常常就在块首附近），
    /// 探不动再回退到 [`Self::lower_bound_fast`]。
    #[inline]
    fn range_lower_bound(&self, lo: &K) -> usize {
        const PROBE_MAX: usize = 4;
        if self.dirty || self.blocks.is_empty() {
            return self.lower_bound(lo);
        }
        let (start, end) = self.block_span(self.block_of(lo));
        let mut i = start;
        let mut steps = 0usize;
        while i < end && steps < PROBE_MAX && self.entries[i].0.cmp_key(lo) == Ordering::Less {
            i += 1;
            steps += 1;
        }
        if i == end || steps < PROBE_MAX || self.entries[i].0.cmp_key(lo) != Ordering::Less {
            return i;
        }
        self.lower_bound_fast(lo)
    }
}

fn pow2_ceil(x: f64) -> f64 {
    if !(x > 0.0) || !x.is_finite() {
        return 1.0;
    }
    let e = x.log2().ceil().clamp(-60.0, 40.0);
    2f64.powf(e)
}

/// [`JumpArray::range`] 的迭代器。
pub struct RangeIter<'a, K: Key, E> {
    slice: &'a [(K, E)],
}

impl<'a, K: Key, E> Iterator for RangeIter<'a, K, E> {
    type Item = (&'a K, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let (head, rest) = self.slice.split_first()?;
        self.slice = rest;
        Some((&head.0, &head.1))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.slice.len(), Some(self.slice.len()))
    }
}

impl<'a, K: Key, E> ExactSizeIterator for RangeIter<'a, K, E> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与"排序 Vec + 二分"逐操作对照（覆盖索引有效 / 脏两条路径）。
    #[test]
    fn matches_sorted_vec_semantics() {
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut a = JumpArray::<i64, i64>::new();
        let mut model: Vec<i64> = Vec::new();
        for step in 0..6000 {
            let k = (next() % 40_000) as i64;
            if next() % 3 != 0 {
                let r = a.insert(k, k * 2);
                let dup = model.binary_search(&k).is_ok();
                assert_eq!(r.is_err(), dup, "step {step} insert {k}");
                if !dup {
                    let pos = model.partition_point(|x| *x < k);
                    model.insert(pos, k);
                }
            } else {
                let r = a.remove(&k);
                match model.binary_search(&k) {
                    Ok(pos) => {
                        model.remove(pos);
                        assert_eq!(r, Some(k * 2), "step {step} remove {k}");
                    }
                    Err(_) => assert!(r.is_none(), "step {step} remove {k}"),
                }
            }
            // 周期性重建：覆盖"索引有效"与"脏"两条路径。
            if step % 97 == 0 {
                a.rebuild();
            }
            let q = (next() % 40_000) as i64;
            let le = model.partition_point(|x| *x <= q);
            let lt = model.partition_point(|x| *x < q);
            assert_eq!(
                a.find_le(&q).map(|(k, _)| *k),
                (le > 0).then(|| model[le - 1]),
                "step {step} find_le {q}"
            );
            assert_eq!(
                a.find_lt(&q).map(|(k, _)| *k),
                (lt > 0).then(|| model[lt - 1]),
                "step {step} find_lt {q}"
            );
            assert_eq!(
                a.find_ge(&q).map(|(k, _)| *k),
                model.get(lt).copied(),
                "step {step} find_ge {q}"
            );
            assert_eq!(
                a.find_gt(&q).map(|(k, _)| *k),
                model.get(le).copied(),
                "step {step} find_gt {q}"
            );
            let (lo, hi) = if q < 20_000 { (q, q + 200) } else { (q - 200, q) };
            let want: Vec<i64> = model
                .iter()
                .copied()
                .filter(|x| *x >= lo && *x <= hi)
                .collect();
            let got: Vec<i64> = a.range(lo, hi).map(|(k, _)| *k).collect();
            assert_eq!(got, want, "step {step} range [{lo},{hi}]");
        }
        a.rebuild();
        assert_eq!(a.len(), model.len());
        assert_eq!(a.iter().map(|(k, _)| *k).collect::<Vec<_>>(), model);
    }

    /// 空 / 单元素 / 反向区间。
    #[test]
    fn handles_edges() {
        let mut a = JumpArray::<i64, u8>::new();
        assert!(a.find_le(&0).is_none());
        a.rebuild();
        assert_eq!(a.range(0, 10).count(), 0);
        a.insert(7, 7).unwrap();
        a.rebuild();
        assert_eq!(a.find_le(&7).map(|(_, v)| *v), Some(7));
        assert_eq!(a.find_le(&6), None);
        assert_eq!(a.find_ge(&8), None);
        assert_eq!(a.range(10, 3).count(), 0);
        assert_eq!(a.range(7, 7).count(), 1);
        assert_eq!(a.remove(&7), Some(7));
        assert!(a.is_empty());
        a.rebuild();
        assert!(a.find_le(&7).is_none());
    }

    /// 稀疏大跨度键（模拟 Beat）：块表 ∝ 元素数，不随键值膨胀。
    #[test]
    fn index_length_scales_with_len_not_key_span() {
        let mut a = JumpArray::<f64, u8>::new();
        for i in 0..10_000 {
            a.insert(i as f64 * 1000.0, 0).unwrap();
        }
        a.rebuild();
        let (blocks, len, _) = a.stats();
        assert!(blocks <= 40_000, "块表 {blocks} 应 ≈ 元素数 {len}");
        assert_eq!(a.find_le(&500_000.5).map(|(k, _)| *k), Some(500_000.0));
    }

    /// 聚簇数据：块内元素数超过 `LINEAR_MAX` 时走块内二分，结果仍与模型一致。
    #[test]
    fn handles_clustered_keys() {
        // 每组 100 个键密集排布（间距 1），组间跳 100_000 —— 典型的"块内堆积"。
        let mut a = JumpArray::<i64, u8>::new();
        let mut model: Vec<i64> = Vec::new();
        for i in 0..3000i64 {
            let k = (i / 100) * 100_000 + (i % 100);
            a.insert(k, 0).unwrap();
            model.push(k);
        }
        a.rebuild();
        let (blocks, len, _) = a.stats();
        assert_eq!(len, 3000);
        assert!(blocks <= 4000);
        for q in (-1..30).map(|i| i * 10_000) {
            let want = model.partition_point(|x| *x <= q);
            assert_eq!(
                a.find_le(&q).map(|(k, _)| *k),
                (want > 0).then(|| model[want - 1]),
                "find_le {q}"
            );
            let want_ge = model.partition_point(|x| *x < q);
            assert_eq!(
                a.find_ge(&q).map(|(k, _)| *k),
                model.get(want_ge).copied(),
                "find_ge {q}"
            );
        }
        // 逐个键验证（密集区内也走二分分支）。
        for k in &model {
            assert_eq!(a.find_le(k).map(|(x, _)| *x), Some(*k), "find_le {k}");
            assert_eq!(
                a.find_ge(k).map(|(x, _)| *x),
                Some(*k),
                "find_ge exact {k}"
            );
        }
    }

    /// `rebuild` 幂等：干净时重复调用不改任何状态（尤其不重复扫描）。
    #[test]
    fn rebuild_is_idempotent() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..1000 {
            a.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        let before = a.stats();
        assert!(!a.is_dirty(), "rebuild 后应干净");
        a.rebuild();
        a.rebuild();
        assert_eq!(a.stats(), before, "重复 rebuild 不应改变块表");
        assert!(!a.is_dirty());
        // 写入后转脏，再 rebuild 应恢复。
        a.insert(-1, 1).unwrap();
        assert!(a.is_dirty(), "写入后索引过期");
        a.rebuild();
        assert!(!a.is_dirty(), "rebuild 后恢复");
        let (blocks, len, _) = a.stats();
        assert_eq!(len, 1001);
        assert!(blocks > 0);
        assert_eq!(a.find_le(&-1).map(|(k, _)| *k), Some(-1), "新首键可取到");
        assert_eq!(a.find_le(&0).map(|(k, _)| *k), Some(0));
        assert!(a.find_le(&-2).is_none(), "base 已换到新首键");
    }

    /// 任意 `linear_max` 下结果一致（线性/二分两条分支都要正确）。
    #[test]
    fn linear_max_variants_agree() {
        let keys: Vec<i64> = (0..2000i64)
            .map(|i| (i / 100) * 100_000 + (i % 100))
            .collect();
        for lm in [1usize, 2, 8, 64] {
            let mut a = JumpArray::<i64, u8>::new().with_linear_max(lm);
            for k in &keys {
                a.insert(*k, 0).unwrap();
            }
            a.rebuild();
            for i in 0..300i64 {
                let q = i * 997;
                let le = keys.partition_point(|x| *x <= q);
                assert_eq!(
                    a.find_le(&q).map(|(k, _)| *k),
                    (le > 0).then(|| keys[le - 1]),
                    "lm={lm} find_le {q}"
                );
                let lt = keys.partition_point(|x| *x < q);
                assert_eq!(
                    a.find_ge(&q).map(|(k, _)| *k),
                    keys.get(lt).copied(),
                    "lm={lm} find_ge {q}"
                );
            }
        }
    }

    /// 窗口大于线性探测上限时，`range` 回退到二分，结果仍完整。
    #[test]
    fn range_falls_back_past_probe_cap() {
        let mut a = JumpArray::<i64, u8>::new().with_linear_max(1);
        let keys: Vec<i64> = (0..500).map(|i| i * 10).collect();
        for k in &keys {
            a.insert(*k, 0).unwrap();
        }
        a.rebuild();
        // 窗口覆盖 200 个元素（cap = max(1*8, 64) = 64，必然走回退）。
        let got: Vec<i64> = a.range(0, 1990).map(|(k, _)| *k).collect();
        assert_eq!(got, keys[..200].to_vec());
        let got: Vec<i64> = a.range(0, 4000).map(|(k, _)| *k).collect();
        assert_eq!(got, keys[..401].to_vec());
    }
}
