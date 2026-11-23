# pairwise-docscan-perspective

## 领域

游戏、图形与多媒体

## 主题

文档扫描透视校正

## 说明

角点回退、光照估计与页面排序。

初始状态下 `imaging/docscan_perspective.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
