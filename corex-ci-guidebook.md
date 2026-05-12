# CoreX CI 使用指南

本仓库提供两份只面向 CoreX/Iluvatar 环境的 GitHub Actions workflow：

- `.github/workflows/corex-pr.yml`：每次 PR 触发，执行配置、编译 `corex_demo`，并运行 `simple90` 和 `stack120` 冒烟测试。
- `.github/workflows/corex-long.yml`：手动触发的长时间测试，运行 `wrecking_ball --frames 400`。

这两份 workflow 都必须运行在真实 CoreX/Iluvatar 机器上的 GitHub self-hosted runner。GitHub 官方托管 runner 没有 CoreX SDK、驱动和 GPU，不能完成这类验证。

## Runner 要求

注册 self-hosted runner 时需要带上这些 label：

```text
self-hosted
linux
corex
```

runner 机器至少需要提供以下命令：

```bash
cmake --version
ninja --version
python3 --version
git --version
/usr/local/corex/bin/clang++ --version
```

CoreX SDK 和驱动版本需要和项目实际运行环境匹配。当前 workflow 使用的关键配置如下：

```text
cmake/corex-clang-cuda-wrapper.sh
UIPC_MUDA_USE_COREX=ON
UIPC_USE_FLOAT=ON
UIPC_MUDA_SOURCE_DIR=external/muda
UIPC_COREX_CUDA_ARCHITECTURES=ivcore11
```

## 依赖模式：真正在线 vcpkg

当前 CI 设计为真正在线复现，不复用本地 `vcpkg_installed` 目录。每台 runner 都使用真实的 vcpkg git checkout，由项目 manifest 在线解析并安装依赖。

vcpkg 必须使用 blobless、非 shallow 的 git clone。这样比完整 blob clone 小很多，同时仍保留 vcpkg versioning 所需的 git history/tree 信息：

```bash
VCPKG_ROOT="${LIBUIPC_VCPKG_ROOT:-$HOME/vcpkg}"
git clone --filter=blob:none https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
"$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
```

不要使用 shallow clone，也不要使用 vcpkg tarball。这个项目的 vcpkg versioning 需要 git 历史来解析 `fmt`、`spdlog` 等 pinned package。本机验证中，shallow vcpkg 会在 manifest resolution 阶段失败；tarball vcpkg 因为没有 `.git` 元数据也会失败。

可选仓库变量：

```text
LIBUIPC_VCPKG_ROOT=/home/<runner-user>/vcpkg
LIBUIPC_DISABLE_ALL_PROXY=1
COREX_CUDA_ARCH=ivcore11
COREX_GPU=0
```

如果不设置 `LIBUIPC_VCPKG_ROOT`，workflow 默认使用 `$HOME/vcpkg`。

## 本机已验证的在线构建流程

下面这组命令已经在当前服务器上验证通过，没有使用 V11 的 `vcpkg_installed` 目录：

```bash
cd /root/libuipc_corex-corex-iluvatar-port

git clone --filter=blob:none https://github.com/microsoft/vcpkg.git _deps/vcpkg-online-blobless
_deps/vcpkg-online-blobless/bootstrap-vcpkg.sh -disableMetrics

env -u ALL_PROXY -u all_proxy VCPKG_FORCE_SYSTEM_BINARIES=1 cmake \
  -S /root/libuipc_corex-corex-iluvatar-port \
  -B /root/libuipc_corex-corex-iluvatar-port/build_corex_online_verified \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/root/libuipc_corex-corex-iluvatar-port/_deps/vcpkg-online-blobless/scripts/buildsystems/vcpkg.cmake \
  -DUIPC_USING_LOCAL_VCPKG=ON \
  -DUIPC_WITH_CUDA_BACKEND=ON \
  -DUIPC_MUDA_SOURCE_DIR=/root/libuipc_corex-corex-iluvatar-port/external/muda \
  -DUIPC_MUDA_USE_COREX=ON \
  -DUIPC_USE_FLOAT=ON \
  -DUIPC_COREX_CUDA_ARCHITECTURES=ivcore11 \
  -DCMAKE_CUDA_COMPILER=/root/libuipc_corex-corex-iluvatar-port/cmake/corex-clang-cuda-wrapper.sh \
  -DUIPC_BUILD_EXAMPLES=ON \
  -DUIPC_BUILD_TESTS=OFF \
  -DUIPC_BUILD_COREX_API_TESTS=ON \
  -DUIPC_BUILD_BENCHMARKS=OFF

cmake --build /root/libuipc_corex-corex-iluvatar-port/build_corex_online_verified \
  --target corex_demo --config Release -j8
```

成功生成的产物：

```text
build_corex_online_verified/Release/bin/libuipc_backend_cuda.so
build_corex_online_verified/Release/bin/corex_demo
```

## 必须固定的依赖版本

为了保证在线复现，需要显式固定以下两个依赖：

```text
eigen3=5.0.1
dylib=3.0.1
```

原因如下：

- `dylib=3.0.1`：当前 CoreX loader 代码使用 `dylib::library` 和 `dylib::decorations`，这是 dylib v3 API。如果不固定版本，当前 baseline 会解析到 `dylib 2.2.1`，导致编译失败。
- `eigen3=5.0.1`：旧的 `eigen3 3.4.0#5` 在 CoreX clang 编译 CUDA 文件时，会在 Eigen `SparseMatrix.h` 中触发 `std::max` 类型推导错误。升级到 `5.0.1` 后已验证可以继续完成 CUDA 后端编译。

这两个版本同时写在根目录 `vcpkg.json` 和 `scripts/gen_vcpkg_json.py` 中。`scripts/gen_vcpkg_json.py` 很关键，因为 CMake 配置阶段会重新生成 build 目录下的 `vcpkg.json`；如果只改根目录 `vcpkg.json`，实际配置时仍可能被脚本覆盖。

## 长时间运行

长时间测试需要在 GitHub 页面手动触发：

```text
Actions -> CoreX Long Run -> Run workflow
```

如果需要指定物理 GPU 映射，可以设置：

```text
COREX_VISIBLE_DEVICES=1
COREX_GPU=0
```

这会把物理设备 1 映射成进程内的 GPU 0，等价于历史命令中的：

```text
CUDA_VISIBLE_DEVICES=1 ... --gpu 0
```
