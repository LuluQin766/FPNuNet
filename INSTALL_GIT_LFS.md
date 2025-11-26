# Git LFS 安装和使用指南

## 问题说明
`CD47_IHCNUSC/images.zip` 文件大小为 217MB，超过了 GitHub 的 100MB 文件大小限制。需要使用 Git LFS 来管理大文件。

## 安装 Git LFS

### 方法 1: 使用 Homebrew（推荐）
```bash
brew install git-lfs
```

### 方法 2: 手动下载安装
1. 访问 https://git-lfs.github.com/
2. 下载 macOS 版本（根据你的架构选择 arm64 或 amd64）
3. 解压并运行安装脚本

### 方法 3: 使用代理下载
如果网络有问题，可以配置代理：
```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
brew install git-lfs
```

## 安装后的步骤

1. **初始化 Git LFS**
```bash
cd /Users/luluqin766/Documents/MI_project/FPNuNet
git lfs install
```

2. **确保 .gitattributes 已配置**
```bash
git add .gitattributes
```

3. **重新添加大文件（使用 LFS）**
```bash
git reset HEAD CD47_IHCNUSC/*.zip
git add CD47_IHCNUSC/*.zip
```

4. **提交并推送**
```bash
git commit -m "Add CD47_IHCNUSC dataset with Git LFS"
git push origin main_data
```

## 验证
推送后，可以使用以下命令验证 LFS 文件：
```bash
git lfs ls-files
```

