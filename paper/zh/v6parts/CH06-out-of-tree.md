## 六、把方言做成工具：外建工程、gdn-opt 与可复现构建

`gdn-opt` 的诞生涉及三类输入：手写 C++、TableGen 生成的声明与定义、预构建 LLVM/MLIR 提供的库和构建规则。外建（out-of-tree）工程把前两类放在独立源码树中，通过 CMake 包配置精确接入第三类。需要管理的重点包括生成顺序、注册入口、二进制版本和证据文件，CMake 语法本身只占很小一部分。

### 1. MLIR_DIR 把独立源码树接到同一份构建

顶层工程先定位 MLIR 包，再借用它导出的构建模块：

```cmake
if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "$ENV{HOME}/codebase/llvm-mlir/build/lib/cmake/mlir")
endif()
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
```

[dialect/CMakeLists.txt:7-30；子目录入口见 34-36 行]

`find_package(MLIR REQUIRED CONFIG)` 读取指定构建目录中的 `MLIRConfig.cmake`，取得头文件、库目标以及 LLVM/MLIR 的 CMake 模块位置。`TableGen`、`AddLLVM`、`AddMLIR` 随之提供 `mlir_tablegen`、`add_mlir_dialect_library`、`add_llvm_executable` 等与上游一致的规则。源码 include 与构建 include 同时出现，是因为 `.inc` 在构建树生成，手写头文件仍在源码树。

工程布局也把清理边界固定下来。`dialect/` 只保存手写源码和 `.td`，`build-dialect/` 保存 Ninja 图、生成头、目标文件与最终工具；删除后者不会触碰上游 worktree。`add_subdirectory(include)`、`add_subdirectory(lib)`、`add_subdirectory(tools/gdn-opt)` 将生成、库和驱动分成三个目标层次，阅读失败日志时可先判断缺的是生成文件、方言库符号还是驱动链接项[dialect/CMakeLists.txt:34-36]。

外建并不意味着版本可以任意组合。ODS 生成代码要匹配当前头文件，目标链接要匹配当前库，TableGen 后端与 C++ API 也会随上游变化。`MLIR_DIR` 应固定指向一次完整的 MLIR 构建；混用另一套 `mlir-tblgen`、头文件或共享库，会把版本偏差推迟到编译、链接甚至运行时。C++17、编译器定义和 LLVM 编译选项目也由同一包配置统一带入[dialect/CMakeLists.txt:1-5,19-32]。

### 2. TableGen 依赖必须先于方言库编译

`include/CMakeLists.txt` 把三组声明分别展开。方言与属性来自 `GdnDialect.td`，两个语义 op 来自 `GdnOps.td`，硬件入口 op 与枚举来自 `GdnHatch.td`。

```cmake
set(LLVM_TARGET_DEFINITIONS Gdn/GdnDialect.td)
mlir_tablegen(GdnDialect.h.inc -gen-dialect-decls)
mlir_tablegen(GdnDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(GdnAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(GdnAttrDefs.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS Gdn/GdnOps.td)
mlir_tablegen(GdnOps.h.inc -gen-op-decls)
mlir_tablegen(GdnOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS Gdn/GdnHatch.td)
mlir_tablegen(GdnHatchOps.h.inc -gen-op-decls)
mlir_tablegen(GdnHatchOps.cpp.inc -gen-op-defs)
mlir_tablegen(GdnHatchEnums.h.inc -gen-enum-decls)
mlir_tablegen(GdnHatchEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(GdnIncGen)
add_dependencies(GdnIncGen mlir-generic-headers)
```

[dialect/include/CMakeLists.txt:1-18]

每次重设 `LLVM_TARGET_DEFINITIONS`，后续 `mlir_tablegen` 才知道输入文件；不同 `-gen-*` 后端生成不同 C++ 片段。`GdnIncGen` 将这些产物汇成一个公共目标，方言库再通过 `DEPENDS GdnIncGen` 建立顺序依赖[dialect/lib/CMakeLists.txt:1-11]。缺少这条依赖时，并行干净构建可能在 `.inc` 尚未生成时编译 `GdnDialect.cpp`，而增量构建的偶然成功会掩盖这个问题。

`GdnDialect.cpp` 在伞头包含生成声明，在实现文件包含生成定义[dialect/lib/GdnDialect.cpp:1-26]。库的 `LINK_LIBS PUBLIC` 则列出 IR、Pass、相关树内方言与变换工具；`PUBLIC` 让这些使用要求传递给链接 `GdnDialect` 的工具[dialect/lib/CMakeLists.txt:12-27]。生成依赖表达“先有什么文件”，链接依赖表达“需要哪些目标”，两者不能互相替代。

伞头 `Gdn.h` 还集中包含方言、属性、语义 op、hatch op 和枚举的生成声明[dialect/include/Gdn/Gdn.h:18-27]。这意味着一个 `.td` 改动可能影响多个 C++ 编译单元，Ninja 应根据目标依赖重新生成并扩大重编范围。审查构建结果时，除了检查 `.inc` 是否出现，还应确认它来自当前构建目录，并由 Ninja 依赖图连接到当前 `.td`；单纯存在不能证明内容已经更新。否则容易把旧生成头造成的 API 差异误认为手写代码错误。

### 3. gdn-opt 与 mlir-opt 共用驱动，拥有不同注册集合

工具源码只有少量注册工作，随后把控制权交给 `MlirOptMain`：

```cpp
int main(int argc, char **argv) {
  registerAllPasses();
  registerPass([]() -> std::unique_ptr<Pass> {
    return gdn::createGdnToChunkPass();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return gdn::createGdnLowerHatchPass();
  });

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<gdn::GdnDialect>();

  return asMainReturnCode(MlirOptMain(
      argc, argv, "GDN dialect optimizer driver\n", registry));
}
```

[dialect/tools/gdn-opt/gdn-opt.cpp:15-30]

这里有两套注册表。pass 注册表把命令行名字映射到工厂；`DialectRegistry` 负责方言可用性，`GdnDialect::initialize()` 在方言加载时登记属性与具体 op[dialect/lib/GdnDialect.cpp:28-37]。`MlirOptMain` 提供输入解析、管线装配、验证、诊断和输出。`gdn-opt` 属于独立链接的 mlir-opt 类驱动：包含 `registerAllPasses` 与 `registerAllDialects` 提供的树内注册项，再增加 GDN 项，执行时无需启动外部 `mlir-opt` 进程。

注册发生在运行前，链接则决定这些注册函数能否出现在可执行文件中。解析器看到 `gdn.fwd` 时需要相应方言，命令行看到 `--gdn-to-chunk` 时需要相应 pass 工厂，两者缺一会在不同阶段失败。方言被动加载也不等于某条下降 pass 已自动加入管线；工具只提供可选能力，执行顺序仍由命令行或上层脚本给出。这一区分解释了为何成功构建 gdn-opt 后，还要分别检查 `--show-dialects`、pass 列表和实际改写输出。

链接清单中的 `MLIROptLib`、`MLIRRegisterAllDialects`、`MLIRRegisterAllPasses` 与 `GdnDialect` 正好对应这些能力[dialect/tools/gdn-opt/CMakeLists.txt:1-20]。选择哪个工具取决于 IR 内容。`run_l1.sh` 的生成物只含树内方言，直接使用上游 `mlir-opt`[scripts/run_l1.sh:19-23]；`run_l2.sh` 遇到 GDN hatch op 时先调用 `gdn-opt --gdn-lower-hatch`，随后仍由上游工具执行 transform 与 NVVM 管线[scripts/run_l2.sh:17-20,37-46,53-72]。共享驱动使两种 pass 能使用相同命令行协议，同时避免给纯树内 IR 强加自定义依赖。

### 4. Release+Assertions 与版本匹配要进入构建记录

方言构建脚本在首次配置时显式给出构建类型、断言、编译器和 MLIR 包：

```bash
if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -G Ninja -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_DIR="$HOME/codebase/llvm-mlir/build/lib/cmake/mlir"
fi

rg_gate_heavy
J=$(rg_jobs); (( J > 6 )) && J=6
export PATH=/usr/local/cuda/bin:$PATH
set +e
ninja -C "$BUILD" -j "$J" gdn-opt &
NPID=$!
rg_watchdog_start "$NPID"
wait "$NPID"; RC=$?
set -e
ls -la "$BUILD/bin/gdn-opt" || exit "$RC"
```

[scripts/build_dialect.sh:13-32，省略条件块之外的初始化和结尾回执]

Release 保留优化后的执行速度，Assertions 保留接口不变量检查，适合日常 pass 开发。上游构建脚本也使用同一组合，并配置 `BUILD_SHARED_LIBS=ON`、`LLVM_ENABLE_PROJECTS=mlir`[scripts/build_llvm.sh:23-39]。方言与上游的选项应一起归档，不能只记录方言目录里的几个 CMake 参数。

共享库构建还要求运行时解析到同一套库。若链接时使用一份 MLIR_DIR，运行环境却优先载入另一份兼容符号名称相同的共享库，编译期检查已经无法保护执行过程。版本偏差常见的表现包括生成声明与手写访问器不匹配、CMake 目标缺失、链接符号缺失以及加载方言时断言。排查时应先对照配置日志、链接器和运行时库路径，再修改方言源码；否则容易把环境问题固化成无谓的兼容分支。

`build.ninja` 存在时脚本只做增量 Ninja，这提供幂等恢复，也意味着修改 CMake 选项后重跑脚本不会自动重新配置。更换 `MLIR_DIR`、编译器或构建类型时，应使用新的构建目录或显式重新配置，并检查新的 `CMakeCache.txt`。上游脚本还通过 `git fetch origin main` 和 worktree 跟踪主线[scripts/build_llvm.sh:8-18]；同一脚本在不同日期可能取得不同源码。严格复现需要记录 `git rev-parse HEAD` 的完整结果，不能把分支名当作版本固定。

### 5. CPU 复现路径以命令、退出码和产物落盘为准

在已有匹配 MLIR 构建的前提下，`scripts/repro.sh g1` 给出全 CPU 的方言与语义验证路径：

```bash
step g1 "JAX executable spec self-consistency suite [CPU]" "5 min" \
  "$PY" tools/gdn_ref_check.py || return 1
step g1 "MLIR L1 forward numeric check (7 cases vs JAX) [CPU]" "3 min" \
  "$PY" tools/gdn_l1_check.py || return 1
step g1 "MLIR L1 backward numeric check (3 npz x 6 grads) [CPU]" "3 min" \
  "$PY" tools/gdn_l1_check.py --bwd || return 1
step g1 "gdn dialect build + gdn-opt smoke [CPU]" "3 min" \
  bash scripts/build_dialect.sh || return 1
```

[scripts/repro.sh:248-259；Makefile 入口见 Makefile:33-34]

`step` 打印命令并用 `tee` 追加到 `logs/repro/g1.log`，返回码来自被测命令本身；任一失败会阻止后续步骤[scripts/repro.sh:41-70]。这段实际命令完成三项数值检查和 gdn-opt 构建，冒烟 IR 的调用改写仍需用 `gdn-opt --gdn-to-chunk` 单独检查，不能仅凭步骤标签认定已经覆盖。完整 `build_llvm.sh` 还会配置 NVPTX 与 CUDA runner[scripts/build_llvm.sh:24-38]，所以 CPU 验证路径与“全新机器只需 X86 工具链”是两个不同命题。

可审查的一次构建至少留下五类材料：两个源码树的完整提交哈希；clang、CMake、Ninja 与 Python 依赖版本；`CMakeCache.txt` 中的 MLIR_DIR、构建类型和断言项；带退出码的配置及编译日志；冒烟 IR、降低后 IR 与数值检查输出。`make llvm` 已把上游构建输出追加到 `logs/build_llvm.log`[Makefile:9-10]，G0 报告也把它列为构建证据[evidence/G0/GATE_REPORT.md:5-11]。归档的 `evidence/G1/l1_check_output.txt` 与 `l1_bwd_check_output.txt` 则保存既有数值结果，当前重跑日志与历史证据应分开命名，避免把一次新执行误写成既有事实。

增量重建与干净重建也要分开记录。前者证明脚本能在已有缓存上恢复，后者才覆盖 TableGen 生成顺序和缺失依赖。对生成的 IR、检查输出和工具二进制计算 SHA-256，可用于确认后续阅读或重跑拿到的是同一工件；二进制哈希一致只说明工件一致，跨编译器的位级可复现仍需独立实验。`[build_dialect] DONE` 仅表示 Ninja 成功且目标文件存在，数值正确性要由独立检查输出支持[scripts/build_dialect.sh:25-32]。
