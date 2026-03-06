# 05 GitHub 部署（把仓库推上去）

## 1) 初始化 git
在仓库根目录：
```bash
git init
git add .
git commit -m "init: yolo pose pushup counter"
```

## 2) 创建 GitHub 仓库并推送
在 GitHub 网站创建一个空仓库（不要勾选 README，因为你本地已有），然后：

```bash
git remote add origin https://github.com/<your_name>/<repo_name>.git
git branch -M main
git push -u origin main
```

## 3) 大文件建议
- `data/`、`datasets/`、`outputs/`、`weights/` 都已被 `.gitignore` 忽略
- 如果你想在 GitHub Release 里放 `best.pt`，建议只在 Release 附件上传，不要进 git history

## 4) 写一个最简单的使用说明
README 已包含 Quickstart，你可以补充：
- 你推荐的拍摄角度
- 你自己的阈值参数（up/down）
