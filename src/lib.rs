//! `JumpArray`：kipphi 的"块索引 + 短走步"思想，落在**键值分数组（SoA）**的连续存储上。
//!
//! ## 结构
//!
//! ```text
//! keys:   Vec<K>    按 key 升序、连续存放
//! vals:   Vec<E>    与 keys 同下标一一对应
//! blocks: Vec<u32>  块起点 → 第一个 key ≥ 块起点的下标
//! ```
//!
//! * **块宽** `W = 2^ceil(log2(span/n))` ⇒ 块数 ≈ 元素数，与键的数值大小无关；
//! * **查询**：`b = floor((key - base)/W)` → `blocks[b]` → 在 **keys 数组**上
//!   向前走几步（块内元素数 ≤ `linear_max` 时线性扫，否则块内二分），命中后
//!   用同一下标取 `vals`；
//! * **区间遍历**：索引起跳后切出两个等长切片，zip 迭代——连续、无指针追逐；
//! * **键值分数组**：块内扫描/二分只碰键数组（无值载荷）；且没有 `(K,E)` 元组的
//!   对齐填充（例如 `(i64,u32)` 在 AoS 里占 16 B，这里只占 8+4=12 B）；
//! * **写入**：两个数组同步搬移 + 标脏。批量写完之后调一次 [`JumpArray::rebuild`]
//!   重建块表（O(n)）。升序追加走 `push` 快路径；
//! * **脏索引**：`rebuild` 之前查询自动退回二分（O(log n)），结果仍然正确；
//! * **小数组**：元素数 ≤ `binary_below`（默认 2048）时直接二分，跳过块索引；
//! * **块号精确计算**：块宽是 `2^shift`，块号由 [`Key::block_offset`] 在键自己的
//!   类型上算——整数键用整数运算（大整数不会因 f64 量化被挤进同一块），内部以
//!   f64 存储的键（`f64`、`Beat`）用 f64 运算；`to_f64` 只用来估 `shift`；
//! * **顺序访问**：配套 [`Cursor`]（`find_le_from` / `find_ge_from` / `range_from`），
//!   从上次位置双向线性推进，顺序窗口摊还 O(1)；游标带实例 id 与写入代数，过期或跨
//!   实例时自动退回冷路径。
//!
//! ## 与 kipphi 原版的差异
//!
//! 1. 数据载体是分数组而不是双向链表：区间遍历与短走步都是顺序读；
//! 2. 索引不是 `T | T[]` union，而是扁平 `u32` 下标，无 per-block 分配；
//! 3. 游标不参与"块内走步"：短走步就是在键数组上递增下标；游标只给顺序访问提供提示；
//! 4. 索引重建是显式的（批量写后一次），不做 `updateRange` 式增量修补。
//!
//! ## 复杂度
//!
//! | 操作 | 复杂度 |
//! |---|---|
//! | 写入（任意位置） | O(n) 搬移 + 标脏 |
//! | 写入（升序追加） | O(1) 摊还 + 标脏 |
//! | `rebuild` | O(n)（幂等：干净时 O(1)） |
//! | 点查询（索引有效） | O(1) 起跳 + 块内短扫 |
//! | 点查询（脏 / 小数组） | O(log n) |
//! | 区间 / 遍历 | O(长度)，连续切片 |
//! | 顺序查询/窗口（游标） | 摊还 O(1) 起跳；单次 O(距离)，超 32 步回退块索引 |
//! | 内存 | 载荷（无对齐填充）+ 4 B/元素 |

use std::cmp::Ordering;

/// 有序键：全序比较 + 块号计算。
///
/// 块号默认在 f64 上算（`to_f64`）；整数类型覆写为**精确的整数运算**，因此
/// 大整数键（超过 f64 的 53 位尾数）不会因为量化而把多个键挤进同一块。
pub trait Key: Copy + std::fmt::Debug + Default {
    fn cmp_key(&self, other: &Self) -> Ordering;
    fn to_f64(self) -> f64;

    /// 块号：`floor((self - base) / 2^shift)`，负数/越界由调用方截断。
    ///
    /// 默认实现走 f64；如果键自身精度是 f64（如 `f64`、内部以 f64 存储的
    /// `Beat`），这就是精确的。整数键应覆写为整数运算。
    #[inline]
    fn block_offset(&self, base: &Self, shift: i32) -> usize {
        let d = self.to_f64() - base.to_f64();
        if d <= 0.0 || d.is_nan() {
            return 0;
        }
        let x = d * 2f64.powi(-shift);
        if !x.is_finite() {
            return usize::MAX;
        }
        x as usize
    }
}

macro_rules! impl_int_key {
    ($($t:ty),*) => {$(
        impl Key for $t {
            #[inline] fn cmp_key(&self, other: &Self) -> Ordering { self.cmp(other) }
            #[inline] fn to_f64(self) -> f64 { self as f64 }
            /// 整数运算，精确。
            #[inline]
            fn block_offset(&self, base: &Self, shift: i32) -> usize {
                let d = (*self as i128) - (*base as i128);
                if d <= 0 {
                    return 0;
                }
                if shift >= 0 {
                    if shift >= 127 {
                        return 0;
                    }
                    (d >> shift) as usize
                } else {
                    let m = (-shift) as u32;
                    d.checked_shl(m).map(|v| v as usize).unwrap_or(usize::MAX)
                }
            }
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
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

const DEFAULT_LINEAR_MAX: usize = 8;
const DEFAULT_BINARY_BELOW: usize = 2048;
const RANGE_SCAN_MAX: usize = 8;

/// 块内元素数超过它就建二级锚。
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// 顺序访问游标（属于某个 [`JumpArray`] 实例）。
///
/// 语义：`pos` 是上一次查询落在 `keys` 里的下标（允许等于 `len`，表示末尾之后）。
/// `id`/`version` 不匹配时视为过期——查询自动走冷路径，结果仍然正确。
#[derive(Debug, Clone, Copy)]
pub struct Cursor {
    id: u64,
    version: u32,
    pos: usize,
}

impl Cursor {
    /// 游标当前下标（诊断/断言用；`len` 表示末尾之后）。
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// 有序容器（key 唯一、按 key 升序）。
///
/// `Clone` 复制内容并把 `id` 换成新实例：克隆副本与原件是**两个独立实例**，
/// 各自的游标不能跨用（会退回冷路径），这是有意为之。
pub struct JumpArray<K: Key, E> {
    keys: Vec<K>,
    vals: Vec<E>,
    /// 块起点 → 第一个 `key >= block_start` 的下标。
    blocks: Vec<u32>,
    /// 块号锚点：块 `b` 的起点 = `base + b * 2^shift`（**在键自己的类型上算**）。
    base: K,
    shift: i32,
    dirty: bool,
    linear_max: usize,
    binary_below: usize,
    /// 区间上界的线性探上限（窗口极短时省一次块表查找）。
    span_scan_max: usize,
    id: u64,
    /// 写入代数：`insert`/`remove` +1；`rebuild` 不改下标，故不变。
    version: u32,
}

impl<K: Key, E> Default for JumpArray<K, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, E: Clone> Clone for JumpArray<K, E> {
    /// 克隆内容，但**分配新的实例 id**：副本与原件是两个独立实例。
    ///
    /// `version`（写入代数）随内容一起复制；副本自己的游标照常工作，原件的游标在
    /// 副本上会因 `id` 不匹配而退回冷路径。
    fn clone(&self) -> Self {
        Self {
            keys: self.keys.clone(),
            vals: self.vals.clone(),
            blocks: self.blocks.clone(),
            base: self.base,
            shift: self.shift,
            dirty: self.dirty,
            linear_max: self.linear_max,
            binary_below: self.binary_below,
            span_scan_max: self.span_scan_max,
            id: NEXT_ID.fetch_add(1, AtomicOrdering::Relaxed),
            version: self.version,
        }
    }
}

/// 摘要形式：不打印元素内容，也不要求 `E: Debug`。
impl<K: Key, E> std::fmt::Debug for JumpArray<K, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JumpArray")
            .field("len", &self.keys.len())
            .field("blocks", &self.blocks.len())
            .field("dirty", &self.dirty)
            .field("shift", &self.shift)
            .finish_non_exhaustive()
    }
}

impl<K: Key, E> JumpArray<K, E> {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            vals: Vec::new(),
            blocks: Vec::new(),
            base: K::default(),
            shift: 0,
            dirty: true,
            linear_max: DEFAULT_LINEAR_MAX,
            binary_below: DEFAULT_BINARY_BELOW,
            span_scan_max: RANGE_SCAN_MAX,
            id: NEXT_ID.fetch_add(1, AtomicOrdering::Relaxed),
            version: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut s = Self::new();
        s.keys.reserve(capacity);
        s.vals.reserve(capacity);
        s
    }

    /// 调整"块内线性扫"阈值（块内元素数超过它则块内二分）。
    pub fn with_linear_max(mut self, linear_max: usize) -> Self {
        self.linear_max = linear_max.max(1);
        self
    }

    /// 元素数 ≤ `n` 时跳过块索引、直接用二分。
    ///
    /// 依据：小数组整个留在 L1，二分只有十来次命中；而块索引至少两次依赖加载
    /// （块表 + 键）加一次乘法。默认 2048（32 KB L1 / 每条 ~32 B ≈ 1k，取保守值）。
    pub fn with_binary_below(mut self, n: usize) -> Self {
        self.binary_below = n;
        self
    }

    /// 区间上界的线性探上限（默认 8）：窗口很短时线性探比"块表 + 二分"便宜。
    pub fn with_span_scan_max(mut self, n: usize) -> Self {
        self.span_scan_max = n;
        self
    }

    /// 从**已按 key 升序、且无重复**的条目批量构造（导入/反序列化路径）。
    pub fn from_sorted(items: Vec<(K, E)>) -> Self {
        debug_assert!(
            items.windows(2).all(|w| w[0].0.cmp_key(&w[1].0) == Ordering::Less),
            "from_sorted 要求 key 严格升序且无重复"
        );
        let n = items.len();
        let mut keys = Vec::with_capacity(n);
        let mut vals = Vec::with_capacity(n);
        for (k, v) in items {
            keys.push(k);
            vals.push(v);
        }
        let mut s = Self::new();
        s.keys = keys;
        s.vals = vals;
        s.dirty = true;
        s
    }

    /// 追加一段**已升序、且键都大于当前最大键**的条目。
    pub fn extend_sorted(&mut self, items: impl IntoIterator<Item = (K, E)>) {
        let mut prev = self.keys.last().copied();
        let before = self.keys.len();
        for (k, v) in items {
            if let Some(p) = prev {
                debug_assert!(
                    p.cmp_key(&k) == Ordering::Less,
                    "extend_sorted 要求严格升序，且都大于现有最大键"
                );
            }
            prev = Some(k);
            self.keys.push(k);
            self.vals.push(v);
        }
        if self.keys.len() != before {
            self.dirty = true;
            self.version = self.version.wrapping_add(1);
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// 索引诊断：`(块表长度, 元素数, 块宽)`。
    pub fn stats(&self) -> (usize, usize, f64) {
        (self.blocks.len(), self.keys.len(), 2f64.powi(self.shift))
    }

    /// 索引是否被写入弄脏（`true` ⇒ 查询退回二分）。
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// 查询当前实际走的是不是块索引（`false` ⇒ 二分：脏、未建，或元素数 ≤ `binary_below`）。
    pub fn index_used(&self) -> bool {
        self.use_index()
    }

    pub fn first(&self) -> Option<(&K, &E)> {
        self.keys.first().map(|k| (k, &self.vals[0]))
    }

    pub fn last(&self) -> Option<(&K, &E)> {
        let n = self.keys.len();
        self.keys.last().map(|k| (k, &self.vals[n - 1]))
    }

    /// 按下标取条目。
    pub fn entry(&self, i: usize) -> Option<(&K, &E)> {
        self.keys.get(i).map(|k| (k, &self.vals[i]))
    }

    /// 按 key 升序遍历。
    pub fn iter(&self) -> impl Iterator<Item = (&K, &E)> {
        self.keys.iter().zip(self.vals.iter())
    }

    /// 重建块表（幂等：干净时直接返回）。
    ///
    /// 块宽取 `2^shift`，`shift` 由"平均每个元素占的数值跨度"估算；块号本身由
    /// [`Key::block_offset`] 在**键自己的类型上精确计算**，`shift` 只决定块的粒度。
    pub fn rebuild(&mut self) {
        if !self.dirty {
            return;
        }
        self.blocks.clear();
        if self.keys.is_empty() {
            self.base = K::default();
            self.shift = 0;
            self.dirty = false;
            return;
        }
        let n = self.keys.len();
        let first = self.keys[0];
        let lastf = self.keys[n - 1].to_f64();
        let span_f = (lastf - first.to_f64()).max(0.0);
        let mut shift = 0i32;
        if span_f > 0.0 {
            let per = span_f / n as f64;
            shift = per.log2().ceil().clamp(-60.0, 60.0) as i32;
        }
        // 块数上限：`shift` 由 f64 估算，遇到"键跨度极大但元素少"的数据可能算出
        // 天量空块。把 shift 抬到块数 ≤ 4n+64 为止（只会让块更粗，不影响正确性）。
        let cap = 4 * n + 64;
        while self.keys[n - 1].block_offset(&first, shift) > cap {
            shift += 1;
        }
        self.base = first;
        self.shift = shift;
        let last_b = self.keys[n - 1].block_offset(&first, shift).min(cap);
        let mut blocks = vec![0u32; last_b + 1];
        let mut cur = 0usize;
        for (i, k) in self.keys.iter().enumerate() {
            let b = k.block_offset(&first, shift).min(last_b);
            if b > cur {
                for slot in blocks.iter_mut().take(b + 1).skip(cur + 1) {
                    *slot = i as u32;
                }
                cur = b;
            }
        }
        self.blocks = blocks;
        self.dirty = false;
    }

    #[inline]
    fn use_index(&self) -> bool {
        !self.dirty && !self.blocks.is_empty() && self.keys.len() > self.binary_below
    }

    #[inline]
    fn block_of(&self, k: &K) -> usize {
        if self.blocks.is_empty() {
            return 0;
        }
        let b = k.block_offset(&self.base, self.shift);
        if b >= self.blocks.len() {
            self.blocks.len() - 1
        } else {
            b
        }
    }

    #[inline]
    fn block_span(&self, b: usize) -> (usize, usize) {
        let start = self.blocks[b] as usize;
        let end = if b + 1 < self.blocks.len() {
            self.blocks[b + 1] as usize
        } else {
            self.keys.len()
        };
        (start, end)
    }

    fn lower_bound(&self, k: &K) -> usize {
        self.keys
            .partition_point(|x| x.cmp_key(k) == Ordering::Less)
    }

    fn upper_bound(&self, k: &K) -> usize {
        self.keys
            .partition_point(|x| x.cmp_key(k) != Ordering::Greater)
    }

    fn lower_bound_fast(&self, k: &K) -> usize {
        if !self.use_index() {
            return self.lower_bound(k);
        }
        let (start, end) = self.block_span(self.block_of(k));
        if end - start <= self.linear_max {
            let mut i = start;
            while i < end && self.keys[i].cmp_key(k) == Ordering::Less {
                i += 1;
            }
            i
        } else {
            start
                + self.keys[start..end].partition_point(|x| x.cmp_key(k) == Ordering::Less)
        }
    }

    fn upper_bound_fast(&self, k: &K) -> usize {
        if !self.use_index() {
            return self.upper_bound(k);
        }
        let (start, end) = self.block_span(self.block_of(k));
        if end - start <= self.linear_max {
            let mut i = start;
            while i < end && self.keys[i].cmp_key(k) != Ordering::Greater {
                i += 1;
            }
            i
        } else {
            start
                + self.keys[start..end].partition_point(|x| x.cmp_key(k) != Ordering::Greater)
        }
    }

    /// 插入；key 已存在时返回 `Err(value)`。写入后需 `rebuild` 才恢复快路径。
    pub fn insert(&mut self, key: K, value: E) -> Result<(), E> {
        if let Some(last) = self.keys.last()
            && key.cmp_key(last) == Ordering::Greater
        {
            self.keys.push(key);
            self.vals.push(value);
            self.dirty = true;
            self.version = self.version.wrapping_add(1);
            return Ok(());
        }
        let pos = self.lower_bound(&key);
        if pos < self.keys.len() && self.keys[pos].cmp_key(&key) == Ordering::Equal {
            return Err(value);
        }
        self.keys.insert(pos, key);
        self.vals.insert(pos, value);
        self.dirty = true;
        self.version = self.version.wrapping_add(1);
        Ok(())
    }

    /// 删除；不存在返回 `None`。
    pub fn remove(&mut self, key: &K) -> Option<E> {
        if let Some(last) = self.keys.last()
            && last.cmp_key(key) == Ordering::Equal
        {
            self.keys.pop();
            self.dirty = true;
            self.version = self.version.wrapping_add(1);
            return self.vals.pop();
        }
        let pos = self.lower_bound(key);
        if pos >= self.keys.len() || self.keys[pos].cmp_key(key) != Ordering::Equal {
            return None;
        }
        self.keys.remove(pos);
        self.dirty = true;
        self.version = self.version.wrapping_add(1);
        Some(self.vals.remove(pos))
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
            Some((&self.keys[i - 1], &self.vals[i - 1]))
        }
    }

    /// 最后一个 `key < k`。
    pub fn find_lt(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.lower_bound_fast(k);
        if i == 0 {
            None
        } else {
            Some((&self.keys[i - 1], &self.vals[i - 1]))
        }
    }

    /// 第一个 `key >= k`。
    pub fn find_ge(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.lower_bound_fast(k);
        self.keys.get(i).map(|key| (key, &self.vals[i]))
    }

    /// 第一个 `key > k`。
    pub fn find_gt(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.upper_bound_fast(k);
        self.keys.get(i).map(|key| (key, &self.vals[i]))
    }

    /// `[lo, hi]` 区间（含两端、升序、连续切片）。
    pub fn range(&self, lo: K, hi: K) -> RangeIter<'_, K, E> {
        if self.is_empty() || lo.cmp_key(&hi) == Ordering::Greater {
            return RangeIter::empty();
        }
        let from = self.lower_bound_fast(&lo);
        let to = self.range_end_from(from, &hi);
        RangeIter::new(&self.keys, &self.vals, from, to)
    }

    /// 按下标区间 `[from, to)` 直接迭代（**不做定位**；越界自动截断）。
    pub fn span_iter(&self, (from, to): (usize, usize)) -> RangeIter<'_, K, E> {
        let n = self.keys.len();
        let from = from.min(n);
        let to = to.clamp(from, n);
        RangeIter::new(&self.keys, &self.vals, from, to)
    }

    /// `from` 起、上界为 `hi` 的区间结束下标（排他）。
    ///
    /// 先线性探 `span_scan_max` 项（窗口极短时省一次块表查找；默认 8）；探不到就按
    /// `hi` 所在块定位：块尾 < hi ⇒ 答案就是块尾；否则在块内二分。
    fn range_end_from(&self, from: usize, hi: &K) -> usize {
        if !self.use_index() {
            return self.upper_bound(hi).max(from);
        }
        let n = self.keys.len();
        let mut to = from;
        while to < n && to - from < self.span_scan_max && self.keys[to].cmp_key(hi) != Ordering::Greater {
            to += 1;
        }
        if to - from < self.span_scan_max || to == n {
            return to;
        }
        let b = self.block_of(hi);
        let (start, end) = self.block_span(b);
        if start >= end {
            return end.max(from);
        }
        if self.keys[end - 1].cmp_key(hi) == Ordering::Less {
            return end;
        }
        let lo = from.max(start);
        lo + self.keys[lo..end].partition_point(|x| x.cmp_key(hi) != Ordering::Greater)
    }

    /// 创建属于本实例的游标（顺序访问用）。
    pub fn cursor(&self) -> Cursor {
        Cursor {
            id: self.id,
            version: self.version,
            pos: 0,
        }
    }

    fn cursor_ok(&self, cur: &Cursor) -> bool {
        cur.id == self.id && cur.version == self.version
    }

    fn cursor_lower_bound(&self, cur: &mut Cursor, k: &K) -> usize {
        const STEP_MAX: usize = 32;
        let n = self.keys.len();
        if !self.cursor_ok(cur) {
            cur.id = self.id;
            cur.version = self.version;
            let i = self.lower_bound_fast(k);
            cur.pos = i;
            return i;
        }
        let mut i = cur.pos.min(n);
        if i < n && self.keys[i].cmp_key(k) == Ordering::Equal {
            cur.pos = i;
            return i;
        }
        if i == n || self.keys[i].cmp_key(k) != Ordering::Less {
            let mut steps = 0usize;
            while i > 0 && self.keys[i - 1].cmp_key(k) != Ordering::Less && steps < STEP_MAX {
                i -= 1;
                steps += 1;
            }
            if i > 0 && self.keys[i - 1].cmp_key(k) != Ordering::Less {
                i = self.lower_bound_fast(k);
            }
            cur.pos = i;
            return i;
        }
        let mut steps = 0usize;
        while i < n && self.keys[i].cmp_key(k) == Ordering::Less && steps < STEP_MAX {
            i += 1;
            steps += 1;
        }
        if i < n && self.keys[i].cmp_key(k) == Ordering::Less {
            i = self.lower_bound_fast(k);
        }
        cur.pos = i;
        i
    }

    fn cursor_upper_bound(&self, cur: &mut Cursor, k: &K) -> usize {
        const STEP_MAX: usize = 32;
        let n = self.keys.len();
        if !self.cursor_ok(cur) {
            cur.id = self.id;
            cur.version = self.version;
            let i = self.upper_bound_fast(k);
            cur.pos = i;
            return i;
        }
        let mut i = cur.pos.min(n);
        if i == n || self.keys[i].cmp_key(k) == Ordering::Greater {
            let mut steps = 0usize;
            while i > 0 && self.keys[i - 1].cmp_key(k) == Ordering::Greater && steps < STEP_MAX {
                i -= 1;
                steps += 1;
            }
            if i > 0 && self.keys[i - 1].cmp_key(k) == Ordering::Greater {
                i = self.upper_bound_fast(k);
            }
            cur.pos = i;
            return i;
        }
        let mut steps = 0usize;
        while i < n && self.keys[i].cmp_key(k) != Ordering::Greater && steps < STEP_MAX {
            i += 1;
            steps += 1;
        }
        if i < n && self.keys[i].cmp_key(k) != Ordering::Greater {
            i = self.upper_bound_fast(k);
        }
        cur.pos = i;
        i
    }

    /// 从游标出发的 `find_le`。
    pub fn find_le_from(&self, cur: &mut Cursor, k: &K) -> Option<(&K, &E)> {
        if self.is_empty() {
            *cur = self.cursor();
            return None;
        }
        let i = self.cursor_upper_bound(cur, k);
        if i == 0 {
            None
        } else {
            cur.pos = i - 1;
            self.entry(i - 1)
        }
    }

    /// 从游标出发的 `find_ge`。
    pub fn find_ge_from(&self, cur: &mut Cursor, k: &K) -> Option<(&K, &E)> {
        if self.is_empty() {
            *cur = self.cursor();
            return None;
        }
        let i = self.cursor_lower_bound(cur, k);
        self.entry(i)
    }

    /// 从游标出发的 `range`：顺序窗口直接前推，跨大步时回退块索引/二分。
    pub fn range_from(&self, cur: &mut Cursor, lo: K, hi: K) -> RangeIter<'_, K, E> {
        if self.is_empty() || lo.cmp_key(&hi) == Ordering::Greater {
            return RangeIter::empty();
        }
        let from = self.cursor_lower_bound(cur, &lo);
        let to = self.range_end_from(from, &hi);
        cur.pos = to;
        RangeIter::new(&self.keys, &self.vals, from, to)
    }
}

/// [`JumpArray::range`] / [`JumpArray::range_from`] 的迭代器（两个切片 zip）。
pub struct RangeIter<'a, K: Key, E> {
    keys: &'a [K],
    vals: &'a [E],
    start: usize,
    end: usize,
}

impl<'a, K: Key, E> RangeIter<'a, K, E> {
    fn empty() -> Self {
        Self {
            keys: &[],
            vals: &[],
            start: 0,
            end: 0,
        }
    }

    fn new(keys: &'a [K], vals: &'a [E], from: usize, to: usize) -> Self {
        Self {
            keys: &keys[from..to],
            vals: &vals[from..to],
            start: from,
            end: to,
        }
    }

    /// 区间在容器里的下标范围 `[start, end)`（配合 [`JumpArray::entry`]）。
    pub fn span(&self) -> (usize, usize) {
        (self.start, self.end)
    }
}

impl<'a, K: Key, E> Iterator for RangeIter<'a, K, E> {
    type Item = (&'a K, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let (key, keys) = self.keys.split_first()?;
        let (val, vals) = self.vals.split_first()?;
        self.keys = keys;
        self.vals = vals;
        Some((key, val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.keys.len(), Some(self.keys.len()))
    }
}

impl<'a, K: Key, E> ExactSizeIterator for RangeIter<'a, K, E> {}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_le(model: &[(i64, i64)], q: i64) -> Option<i64> {
        let i = model.partition_point(|x| x.0 <= q);
        (i > 0).then(|| model[i - 1].0)
    }

    /// 与"排序 Vec + 二分"逐操作对照（含 rebuild 时机、区间、游标）。
    #[test]
    fn matches_sorted_vec_semantics() {
        let mut rng: u64 = 0x5eed_face_1234_5678;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut a = JumpArray::<i64, i64>::new();
        let mut model: Vec<(i64, i64)> = Vec::new();
        let mut cur = a.cursor();
        for step in 0..4000 {
            let k = (next() % 30_000) as i64;
            if next() % 3 != 0 {
                let r = a.insert(k, k * 2);
                let dup = model.binary_search_by_key(&k, |x| x.0).is_ok();
                assert_eq!(r.is_err(), dup, "step {step} insert {k}");
                if !dup {
                    let pos = model.partition_point(|x| x.0 < k);
                    model.insert(pos, (k, k * 2));
                }
            } else {
                let r = a.remove(&k);
                match model.binary_search_by_key(&k, |x| x.0) {
                    Ok(pos) => {
                        model.remove(pos);
                        assert_eq!(r, Some(k * 2), "step {step} remove {k}");
                    }
                    Err(_) => assert!(r.is_none(), "step {step} remove {k}"),
                }
            }
            if step % 89 == 0 {
                a.rebuild();
            }
            let q = (next() % 30_000) as i64;
            assert_eq!(
                a.find_le(&q).map(|(k, _)| *k),
                model_le(&model, q),
                "step {step} find_le {q}"
            );
            let ge = model.partition_point(|x| x.0 < q);
            assert_eq!(
                a.find_ge(&q).map(|(k, _)| *k),
                model.get(ge).map(|x| x.0),
                "step {step} find_ge {q}"
            );
            let lt = model.partition_point(|x| x.0 < q);
            assert_eq!(
                a.find_lt(&q).map(|(k, _)| *k),
                (lt > 0).then(|| model[lt - 1].0),
                "step {step} find_lt {q}"
            );
            let gt = model.partition_point(|x| x.0 <= q);
            assert_eq!(
                a.find_gt(&q).map(|(k, _)| *k),
                model.get(gt).map(|x| x.0),
                "step {step} find_gt {q}"
            );
            let (lo, hi) = if q < 15_000 { (q, q + 200) } else { (q - 200, q) };
            let want: Vec<i64> = model
                .iter()
                .filter(|(k, _)| *k >= lo && *k <= hi)
                .map(|(k, _)| *k)
                .collect();
            let got: Vec<i64> = a.range(lo, hi).map(|(k, _)| *k).collect();
            assert_eq!(got, want, "step {step} range [{lo},{hi}]");
            // 游标路径与无游标结果一致
            let want_le = model_le(&model, q);
            assert_eq!(
                a.find_le_from(&mut cur, &q).map(|(k, _)| *k),
                want_le,
                "step {step} find_le_from {q}"
            );
            assert_eq!(
                a.range_from(&mut cur, lo, hi).map(|(k, _)| *k).collect::<Vec<_>>(),
                want,
                "step {step} range_from [{lo},{hi}]"
            );
        }
        a.rebuild();
        assert_eq!(
            a.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            model.iter().map(|x| x.0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn handles_edges() {
        let mut a = JumpArray::<i64, u8>::new();
        assert!(a.is_empty());
        assert_eq!(a.find_le(&5), None);
        assert_eq!(a.find_ge(&5), None);
        assert_eq!(a.range(0, 10).count(), 0);
        a.insert(5, 1).unwrap();
        a.rebuild();
        assert_eq!(a.first().map(|(k, _)| *k), Some(5));
        assert_eq!(a.last().map(|(k, _)| *k), Some(5));
        assert_eq!(a.find_le(&4), None);
        assert_eq!(a.find_le(&5).map(|(k, _)| *k), Some(5));
        assert_eq!(a.find_ge(&6), None);
        assert_eq!(a.range(5, 5).count(), 1);
        assert_eq!(a.range(6, 9).count(), 0);
    }

    /// 聚簇数据：100 键/组，组间跳 100_000。
    #[test]
    fn handles_clustered_keys() {
        let key = |i: usize| ((i / 100) as i64) * 100_000 + (i % 100) as i64;
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..5000 {
            a.insert(key(i), 0).unwrap();
        }
        a.rebuild();
        for p in 0..5000usize {
            let k = key(p);
            assert_eq!(a.find_le(&k).map(|(k, _)| *k), Some(k), "le {k}");
            assert_eq!(a.find_ge(&k).map(|(k, _)| *k), Some(k), "ge {k}");
            let got: Vec<i64> = a.range(k, k + 63).map(|(x, _)| *x).collect();
            let expect = 64.min(100 - (p % 100));
            assert_eq!(got.len(), expect, "窗口 {p}");
            assert_eq!(got[0], k);
        }
        // 组间空隙
        assert_eq!(a.find_le(&99_050).map(|(k, _)| *k), Some(99));
    }

    /// 块数随元素数、而不是键值跨度增长。
    #[test]
    fn index_length_scales_with_len_not_key_span() {
        let mut a = JumpArray::<f64, u8>::new();
        for i in 0..4000 {
            a.insert(i as f64 * 1e6, 0).unwrap();
        }
        a.rebuild();
        let (blocks, len, _) = a.stats();
        assert_eq!(len, 4000);
        assert!(blocks < 3 * len, "blocks={blocks} len={len}");
        assert!(a.index_used());
    }

    /// 大整数键：块号在 i64 上精确计算，不会被 f64 量化挤成少数块。
    #[test]
    fn large_integer_keys_keep_block_granularity() {
        let base: i64 = 1 << 60;
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..4096i64 {
            a.insert(base + i, 0).unwrap();
        }
        a.rebuild();
        let (blocks, len, _) = a.stats();
        assert_eq!(len, 4096);
        // 2^60 附近相邻整数在 f64 里是同一个值；走 f64 的话块数会远小于 n。
        assert!(blocks > 2000, "blocks={blocks}");
        for i in (0..4096i64).step_by(7) {
            let k = base + i;
            assert_eq!(a.find_le(&k).map(|(x, _)| *x), Some(k));
            assert_eq!(a.find_ge(&k).map(|(x, _)| *x), Some(k));
        }
        let got: Vec<i64> = a.range(base + 100, base + 163).map(|(x, _)| *x).collect();
        assert_eq!(got, (100..=163i64).map(|i| base + i).collect::<Vec<_>>());
    }

    /// 键跨度极大但元素很少时，块表不会爆炸（shift 会被抬高）。
    #[test]
    fn huge_span_does_not_explode_block_table() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..8i64 {
            a.insert(i << 58, 0).unwrap();
        }
        a.rebuild();
        let (blocks, _, _) = a.stats();
        assert!(blocks <= 4 * 8 + 64, "blocks={blocks}");
        assert_eq!(a.find_le(&(3i64 << 58)).map(|(k, _)| *k), Some(3i64 << 58));
    }

    /// `rebuild` 幂等：干净时不必重算。
    #[test]
    fn rebuild_is_idempotent() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..500 {
            a.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        let (b1, l1, w1) = a.stats();
        a.rebuild();
        assert_eq!(a.stats(), (b1, l1, w1));
        a.insert(1, 0).unwrap();
        assert!(a.is_dirty());
        a.rebuild();
        assert!(!a.is_dirty());
    }

    /// `linear_max` 只影响路径，不影响结果。
    #[test]
    fn linear_max_variants_agree() {
        let mut v: Vec<i64> = (0..3000).map(|i| ((i / 40) * 10_000 + i % 40) as i64).collect();
        v.dedup();
        let mut arrs: Vec<JumpArray<i64, u8>> = Vec::new();
        for lm in [1usize, 2, 8, 32, 64] {
            let mut a = JumpArray::new().with_linear_max(lm);
            for &k in &v {
                a.insert(k, 0).unwrap();
            }
            a.rebuild();
            arrs.push(a);
        }
        for q in (0..3000).map(|i| i * 7) {
            let expect = arrs[0].find_le(&q).map(|(k, _)| *k);
            for a in &arrs[1..] {
                assert_eq!(a.find_le(&q).map(|(k, _)| *k), expect, "q={q}");
            }
        }
    }

    /// 区间超过线性探上限时走块内二分，结果仍正确。
    #[test]
    fn range_falls_back_past_probe_cap() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..3000 {
            a.insert(i * 997, 0).unwrap();
        }
        a.rebuild();
        assert!(a.index_used(), "3000 条应走块索引");
        for (start, count) in [(0usize, 200usize), (300, 401)] {
            let lo = (start as i64) * 997;
            let hi = ((start + count - 1) as i64) * 997;
            let got: Vec<i64> = a.range(lo, hi).map(|(k, _)| *k).collect();
            let want: Vec<i64> = (start..start + count).map(|i| (i as i64) * 997).collect();
            assert_eq!(got, want);
        }
    }

    /// 上界跨块 / 超出末尾 / 空区间。
    #[test]
    fn range_upper_bound_crosses_blocks() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..1000i64 {
            a.insert(i * 10, 0).unwrap();
        }
        a.rebuild();
        let got: Vec<i64> = a.range(0, 5000).map(|(k, _)| *k).collect();
        assert_eq!(got, (0..=500).map(|i| i * 10).collect::<Vec<_>>());
        let got: Vec<i64> = a.range(9990, 1_000_000).map(|(k, _)| *k).collect();
        assert_eq!(got, vec![9990]);
        assert_eq!(a.range(5005, 5009).count(), 0);
        let got: Vec<i64> = a.range(7000, 7009).map(|(k, _)| *k).collect();
        assert_eq!(got, vec![7000]);
    }

    /// 游标窗口 = 无游标窗口（含窗口边界与 span 一致性）。
    #[test]
    fn cursor_windows_match_range() {
        let key = |i: usize| ((i / 100) as i64) * 100_000 + (i % 100) as i64;
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..10_000 {
            a.insert(key(i), 0).unwrap();
        }
        a.rebuild();
        let mut cur = a.cursor();
        for w in 0..(10_000 / 64) {
            let lo = key(w * 64);
            let hi = key(((w + 1) * 64).min(9999));
            let (from, to) = a.range_from(&mut cur, lo, hi).span();
            let got: Vec<i64> = a.span_iter((from, to)).map(|(k, _)| *k).collect();
            let want: Vec<i64> = a.range(lo, hi).map(|(k, _)| *k).collect();
            assert_eq!(got, want, "窗口 {w}");
            assert_eq!(cur.pos(), to);
        }
    }

    /// 写入让游标失效：之后仍然正确（回退冷路径）。
    #[test]
    fn cursor_recovers_after_writes() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..1000 {
            a.insert(i * 2, 0).unwrap();
        }
        a.rebuild();
        let mut cur = a.cursor();
        assert_eq!(a.find_le_from(&mut cur, &500).map(|(k, _)| *k), Some(500));
        a.insert(501, 0).unwrap();
        a.rebuild();
        assert_eq!(a.find_le_from(&mut cur, &500).map(|(k, _)| *k), Some(500));
        assert_eq!(a.find_le_from(&mut cur, &501).map(|(k, _)| *k), Some(501));
        assert_eq!(a.find_ge_from(&mut cur, &1500).map(|(k, _)| *k), Some(1500));
    }

    /// 游标双向移动与无游标结果一致。
    #[test]
    fn cursor_bidirectional_matches() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..2000 {
            a.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        let mut cur = a.cursor();
        for q in [0i64, 6000, 3000, 5999, 1, 2997, 5997] {
            let want = a.find_le(&q).map(|(k, _)| *k);
            let got = a.find_le_from(&mut cur, &q).map(|(k, _)| *k);
            assert_eq!(got, want, "le {q}");
            let want = a.find_ge(&q).map(|(k, _)| *k);
            let got = a.find_ge_from(&mut cur, &q).map(|(k, _)| *k);
            assert_eq!(got, want, "ge {q}");
        }
    }

    /// 克隆 = 新实例：内容一致，游标不能跨用（退回冷路径，结果正确）。
    #[test]
    fn clone_is_an_independent_instance() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..3000i64 {
            a.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        let mut cur = a.cursor();
        assert_eq!(a.find_le_from(&mut cur, &100).map(|(k, _)| *k), Some(99));

        let b = a.clone();
        assert_eq!(b.len(), 3000);
        // 旧游标在副本上：id 不匹配 ⇒ 冷路径，结果仍正确
        assert_eq!(b.find_le_from(&mut cur, &100).map(|(k, _)| *k), Some(99));
        assert_eq!(b.find_ge_from(&mut cur, &100).map(|(k, _)| *k), Some(102));
        // 副本自己的游标照常工作
        let mut cur_b = b.cursor();
        assert_eq!(b.find_le_from(&mut cur_b, &100).map(|(k, _)| *k), Some(99));
        assert_eq!(
            a.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            b.iter().map(|(k, _)| *k).collect::<Vec<_>>()
        );
        // Debug 是摘要（不含元素内容）
        let d = format!("{b:?}");
        assert!(d.contains("JumpArray"), "debug={d}");
        assert!(d.contains("len: 3000"), "debug={d}");
    }

    /// 脏状态下克隆：两边都保持脏、查询都正确。
    #[test]
    fn clone_preserves_dirty_state() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..500i64 {
            a.insert(i * 2, 0).unwrap();
        }
        assert!(a.is_dirty());
        let b = a.clone();
        assert!(b.is_dirty());
        assert_eq!(b.find_le(&500).map(|(k, _)| *k), Some(500));
        a.rebuild();
        let mut c = a.clone();
        c.rebuild();
        assert!(!c.is_dirty());
        assert_eq!(c.find_le(&500).map(|(k, _)| *k), Some(500));
    }

    /// 跨实例的游标：id 不匹配 ⇒ 冷路径，结果正确。
    #[test]
    fn cursor_from_other_instance_falls_back() {
        let mut a = JumpArray::<i64, u8>::new();
        let mut b = JumpArray::<i64, u8>::new();
        for i in 0..100 {
            a.insert(i * 2, 0).unwrap();
            b.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        b.rebuild();
        let mut cur = a.cursor();
        assert_eq!(a.find_le_from(&mut cur, &50).map(|(k, _)| *k), Some(50));
        assert_eq!(b.find_le_from(&mut cur, &50).map(|(k, _)| *k), Some(48));
        assert_eq!(b.find_ge_from(&mut cur, &50).map(|(k, _)| *k), Some(51));
    }

    /// 升序插入走追加快路径：顺序、去重、脏标记都对。
    #[test]
    fn append_fast_path_keeps_order_and_dedup() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..1000 {
            a.insert(i * 3, (i % 251) as u8).unwrap();
        }
        assert!(a.is_dirty());
        assert_eq!(a.len(), 1000);
        assert_eq!(a.first().map(|(k, _)| *k), Some(0));
        assert_eq!(a.last().map(|(k, _)| *k), Some(2997));
        assert!(a.insert(2997, 9).is_err());
        assert_eq!(a.len(), 1000);
        a.rebuild();
        assert_eq!(
            a.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            (0..1000).map(|i| i * 3).collect::<Vec<_>>()
        );
        assert_eq!(a.find_le(&2998).map(|(k, _)| *k), Some(2997));
        assert_eq!(a.find_ge(&2999), None);
    }

    /// 删末尾走快速路径。
    #[test]
    fn remove_last_is_fast_and_correct() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..100 {
            a.insert(i * 2, 0).unwrap();
        }
        a.rebuild();
        for i in (0..100i64).rev() {
            assert_eq!(a.remove(&(i * 2)), Some(0), "删 {i}");
            if i % 10 == 0 {
                a.rebuild();
            }
        }
        assert!(a.is_empty());
        assert_eq!(a.remove(&0), None);
    }

    /// 批量构造/追加。
    #[test]
    fn bulk_sorted_construct_and_extend() {
        let items: Vec<(i64, u8)> = (0..500).map(|i| (i * 3, (i % 7) as u8)).collect();
        let mut a = JumpArray::from_sorted(items.clone());
        assert_eq!(a.len(), 500);
        assert!(a.is_dirty(), "懒建索引：先脏");
        assert_eq!(a.find_le(&299).map(|(k, _)| *k), Some(297));
        a.rebuild();
        assert!(!a.is_dirty());
        let tail: Vec<(i64, u8)> = (500..1000).map(|i| (i * 3, 0)).collect();
        a.extend_sorted(tail);
        assert_eq!(a.len(), 1000);
        assert!(a.is_dirty());
        a.rebuild();
        assert_eq!(a.last().map(|(k, _)| *k), Some(2997));
        assert_eq!(a.find_ge(&2998), None);
        assert_eq!(a.find_le(&2999).map(|(k, _)| *k), Some(2997));
        assert_eq!(
            a.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            (0..1000).map(|i| i * 3).collect::<Vec<_>>()
        );
    }

    /// `binary_below` 开关不影响结果。
    #[test]
    fn binary_below_switch_keeps_results_correct() {
        for below in [0usize, 10, 1000, 100_000] {
            let mut a = JumpArray::<i64, u8>::new().with_binary_below(below);
            for i in 0..2000i64 {
                a.insert(i * 3, 0).unwrap();
            }
            a.rebuild();
            let keys: Vec<i64> = (0..2000).map(|i| i * 3).collect();
            for q in [0i64, 1, 100, 2999, 3000, 6002] {
                let le = keys.partition_point(|x| *x <= q);
                assert_eq!(
                    a.find_le(&q).map(|(k, _)| *k),
                    (le > 0).then(|| keys[le - 1]),
                    "below={below} find_le {q}"
                );
                let ge = keys.partition_point(|x| *x < q);
                assert_eq!(
                    a.find_ge(&q).map(|(k, _)| *k),
                    keys.get(ge).copied(),
                    "below={below} find_ge {q}"
                );
            }
            let got: Vec<i64> = a.range(100, 900).map(|(k, _)| *k).collect();
            assert_eq!(got, (34..=300).map(|i| i * 3).collect::<Vec<_>>());
        }
    }

    /// `span_scan_max` 只影响路径，不影响区间结果。
    #[test]
    fn span_scan_max_variants_agree() {
        for ssm in [0usize, 1, 4, 8, 16, 32] {
            let mut a = JumpArray::<i64, u8>::new().with_span_scan_max(ssm);
            for i in 0..3000 {
                a.insert(i * 997, 0).unwrap();
            }
            a.rebuild();
            for (start, count) in [(0usize, 1usize), (300, 2), (0, 8), (100, 64), (2000, 999)] {
                let lo = (start as i64) * 997;
                let hi = ((start + count - 1) as i64) * 997;
                let got: Vec<i64> = a.range(lo, hi).map(|(k, _)| *k).collect();
                let want: Vec<i64> = (start..start + count).map(|i| (i as i64) * 997).collect();
                assert_eq!(got, want, "ssm={ssm} [{start},{count}]");
            }
        }
    }

    /// 默认阈值下：小数组走二分，大数组走块索引。
    #[test]
    fn small_arrays_skip_the_index_by_default() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..1000i64 {
            a.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        assert!(!a.index_used(), "1000 条应走二分");
        for i in 1000..3000i64 {
            a.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        assert!(a.index_used(), "3000 条应走块索引");
        assert_eq!(a.find_le(&4500).map(|(k, _)| *k), Some(4500));
        assert_eq!(a.find_ge(&4501).map(|(k, _)| *k), Some(4503));
    }

    /// `x / 2^k == x * 2^-k`（f64 上逐位等价），故 `block_of` 可换乘法。
    #[test]
    fn div_equals_mul_inverse_for_pow2() {
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..10_000 {
            let x = ((next() % 2_000_000) as f64 - 1_000_000.0) / 997.0;
            for e in [-20i32, -10, -3, -1, 0, 1, 3, 10, 20] {
                let w = 2f64.powi(e);
                assert_eq!(
                    (x / w).floor().to_bits(),
                    (x * (1.0 / w)).floor().to_bits(),
                    "x={x} e={e}"
                );
            }
        }
    }
}

