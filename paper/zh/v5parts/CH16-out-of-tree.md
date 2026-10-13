## 十六、外建方言项目：让 gdn-opt 装上树内 pass

方言与 pass 的源码要组织成一个可构建的工程，链接到一份预构建的 MLIR，产出一个既带自有 pass、又拥有全部树内 pass 的命令行工具。这种工程形态叫外建（out-of-tree）：方言代码完全住在 LLVM 仓库之外，作为一个独立 CMake 工程存在。gdn 方言的全部构建资产只有四个 CMakeLists、一个工具源文件与一个 shell 脚本，恰好构成外建模式的最小完整样本。

### 1. 外建的形态与理由

顶层 CMakeLists 声明工程并接上预构建的 MLIR：

```
cmake_minimum_required(VERSION 3.20.0)
project(gdn-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# 独立构建：指向本仓的 MLIR build（mlir/examples/standalone 的外建模式）
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

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/gdn-opt)
```
［dialect/CMakeLists.txt:1-36，节选；MLIRConfig 定位消息在 17 行］

工作机制：find_package(MLIR REQUIRED CONFIG) 读取预构建 MLIR 安装出的 MLIRConfig.cmake，由此拿到头文件、库与两个关键变量——MLIR_CMAKE_DIR 与 LLVM_CMAKE_DIR 指向 MLIR 自带的构建函数库；把它们追加进 CMAKE_MODULE_PATH 后，include(TableGen/AddLLVM/AddMLIR/HandleLLVMOptions) 把 mlir_tablegen、add_mlir_dialect_library、add_llvm_executable 这些树内构建函数原样借入本工程。三个 add_subdirectory 对应工程的物理布局，与职责一一映射：include/ 下放声明（三个 .td 加伞头 Gdn.h，生成物 .inc 落在构建目录的同名路径）、lib/ 下放三个实现文件（方言注册与 verifier 的 GdnDialect.cpp、两个 pass 文件）、tools/gdn-opt 下放工具源码；另有 test/smoke_l0.mlir 作为手工冒烟输入。构建发生在仓库根的 build-dialect/ 目录，产物是其中的 bin/gdn-opt——源码树与构建树彻底分离，清理只需删除构建目录。注释点明了模板出处：mlir/examples/standalone，上游为外建方言准备的官方示例。选外建的理由有三。其一，上游树保持纯净：gdn 方言无需进 LLVM 仓库，上游 checkout 不被改动，随时可以丢弃重建。其二，可跟随主干滚动：本项目建立在 2026-10 LLVM 主干之上（配套的自建 MLIR 为 1.9 GB Release 共享库，见 evidence/G0/GATE_REPORT.md 判据 1），主干前移时只需重编上游一份，方言工程对着新的 MLIR_DIR 独立重编，改动范围互不传染。其三，体量相称：方言库只有一个，增量重建以秒计，与上游动辄小时级的全量构建解耦。

### 2. 生成层：mlir_tablegen 与公共生成目标

include 目录的 CMakeLists 只做一件事——把第十三章讲过的三个 .td 变成 .inc：

```
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
［dialect/include/CMakeLists.txt:1-18，全文；GdnHatch.td 段 11-15 行］

写法要点：LLVM_TARGET_DEFINITIONS 为紧随的 mlir_tablegen 调用指定输入 .td（所以每个 .td 前都要重设一次）；每个 -gen-* 开关对应一类生成物；add_public_tablegen_target 把全部生成物捆成一个名为 GdnIncGen 的 CMake 目标，后续任何依赖生成头的库只需写 DEPENDS GdnIncGen，CMake 即自动排出"先跑 mlir-tblgen、再编译"的顺序。对 mlir-generic-headers 的追加依赖保证方言生成与 MLIR 自身的通用头生成不冲突。生成物在构建目录里镜像源码树的相对路径，配合顶层 CMakeLists 把源码与构建两处 include 目录都加进搜索路径（第 29-30 行的 include_directories），C++ 源文件里 #include "GdnOps.h.inc" 这类包含无需任何路径前缀即可解析——手写代码、.td 声明与生成代码三者在编译器眼里是同一棵包含树。

### 3. 方言库：add_mlir_dialect_library 与依赖声明

方言本体的三个 .cpp 编成一个库：

```
add_mlir_dialect_library(GdnDialect
  GdnDialect.cpp
  GdnPasses.cpp
  GdnHatch.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  GdnIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalgDialect
  MLIRNVGPUDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRTransformUtils
)
```
［dialect/lib/CMakeLists.txt:1-27，全文］

add_mlir_dialect_library 是 AddMLIR 提供的封装：按方言库的惯例设置编译选项与安装规则。三处依赖各有含义。DEPENDS GdnIncGen 接上生成层——库的三个源文件都包含 .inc，缺此声明会在干净构建上出现头文件找不到的竞态。LINK_LIBS 罗列全部 MLIR CMake 目标：方言注册要触碰的方言（Arith、Func、GPU、Linalg、NVGPU、NVVM、SCF、Tensor，与 GdnDialect.td 的 dependentDialects 一一呼应）、基础设施（IR、Pass、Support、SideEffectInterfaces、InferTypeOpInterface）与下降用到的 TransformUtils。声明为 PUBLIC 的效果是传递链接：下游工具链 GdnDialect 时自动获得整串依赖，无需自行罗列。这份清单同时是一份接口文档——扫一眼即可知道方言向下触碰了哪些树内部件。

### 4. gdn-opt：MlirOptMain 与双 registry

工具目录把 gdn-opt 建成可执行文件：

```
set(LIBS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIROptLib
    MLIRParser
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRSupport
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRTensorDialect
    GdnDialect
  )
add_llvm_executable(gdn-opt gdn-opt.cpp)

llvm_update_compile_flags(gdn-opt)
target_link_libraries(gdn-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(gdn-opt)
```
［dialect/tools/gdn-opt/CMakeLists.txt:1-20，全文］

add_llvm_executable 带入 LLVM 的编译旗标纪律，llvm_update_compile_flags 追加异常与 RTTI 的统一设置，mlir_check_all_link_libraries 做链接自检。工具源文件第十五章已完整展品（dialect/tools/gdn-opt/gdn-opt.cpp:15-30），这里补齐机制视角：MlirOptMain 是 mlir-opt 的通用驱动框架——旗标解析、文件遍历、pass 流水线组装、每步之后的 verify 与打印全部由它提供；调用方只交出两样东西，一是 DialectRegistry（本例先 registerAllDialects 装入全部树内方言，再 registry.insert<gdn::GdnDialect>() 追加自有方言），二是全局 pass 注册表（registerAllPasses 装入全部树内 pass，两个 registerPass 挂上 gdn-to-chunk 与 gdn-lower-hatch）。如此组装的直接收益写在命令行上：gdn-opt 既能跑 --gdn-to-chunk 与 --gdn-lower-hatch，也能跑全部树内旗标。树内旗标的真实威力有现成样本——G0 冒烟链在 mlir-opt 上把树内 pass 串成完整 GPU 下降：transform-interpreter 解释 transform 工件、erase-schedule 抹去调度痕迹、gpu-lower-to-nvvm-pipeline 以 cubin-chip=sm_120a cubin-features=+ptx87 直达 PTX，产出在 5090 真机执行且结果正确（evidence/G0/GATE_REPORT.md 判据 2）——由于 registry 与注册表完全同构，同一串旗标在 gdn-opt 上同样可用，自有 pass 与树内管线在一条命令里接力，无需在两个工具之间搬运中间文件。链接库清单里的 MLIRRegisterAllDialects 与 MLIRRegisterAllPasses 即"装全部"的载体，MLIROptLib 提供 MlirOptMain 本体。

### 5. 构建脚本：幂等与资源纪律

构建由脚本驱动，全 33 行：

```
if [ ! -f "$BUILD/build.ninja" ]; then
  cmake -G Ninja -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DMLIR_DIR="$HOME/codebase/llvm-mlir/build/lib/cmake/mlir"
fi

rg_gate_heavy
J=$(rg_jobs); (( J > 6 )) && J=6   # 小项目，少占预算
export PATH=/usr/local/cuda/bin:$PATH
set +e
ninja -C "$BUILD" -j "$J" gdn-opt &
NPID=$!
rg_watchdog_start "$NPID"
wait "$NPID"; RC=$?
set -e
ls -la "$BUILD/bin/gdn-opt" || exit "$RC"
```
［scripts/build_dialect.sh:13-32，节选；开头的 resource_guard 装载与 conda 环境激活在 5-11 行］

五个要点。第一，幂等：仅当 build.ninja 不存在才 configure，重跑脚本对已配置目录只做增量 ninja；脚本尾部先核验产物存在（第 31 行），缺失则以 ninja 退出码退出，随后打印 DONE 回执（第 32 行）。第二，MLIR_DIR 显式指向预构建上游，与顶层 CMakeLists 的默认值互为印证。第三，Release 加 LLVM_ENABLE_ASSERTIONS=ON：优化编译同时保留断言，方言与 MLIR 接口契约的违规在开发期即暴露。第四，资源纪律来自 resource_guard.sh：rg_gate_heavy 在重任务前做门禁，rg_jobs 给出可用核数并封顶 6，rg_watchdog_start 对后台 ninja 采样监控——这套守护在全仓构建期无 OOM 事件（evidence/G0/GATE_REPORT.md 判据 5）。第五，nvcc 进 PATH：即使方言本体不含 CUDA 源，configure 与链接阶段也需要 CUDA 工具可见。此外脚本在 configure 前先激活 conda 的 gdn 环境（10-11 行），使解释器与工具版本钉在环境定义里，与第 17、18 行显式指定 clang/clang++ 互为双保险——构建可复现的关键在于环境本身可复现。

### 6. 构建硬规则

外建工程贴着主干滚动，代价是吃主干的版本变动。构建纪律沉淀在 G0 报告的过程记录里，四条与方言工程直接相关。其一，Python 绑定版本：MLIR 主干的 nanobind 接口要求 2.x，pip 上 3.1 被拒，须钉在 2.9 以上 3 以下。其二，CMake 缓存污染：configure 一旦把 CUDA 记成 NOTFOUND，缓存会延续该结论，后续重试全部失败——configure 失败的处置是清掉 CMakeCache 重来，脚本对 build 目录的幂等检查也降低了半成品缓存存活的机会。其三，工具可见性：nvcc 不在 PATH 时 configure 阶段即报错，故脚本头部统一 export（上节展品第 24 行）。其四，汇编器配套：PTX ISA 9.3 的 PTX 只有 ptxas-blackwell 认识，普通 ptxas 13.3 只认到 8.8——sm_120a 管线末端必须选对汇编器。这四条规则的共同形态是：错误信息不指向真因（版本被拒、缓存残留、PATH 缺失、汇编器配套），定位成本高而规避成本低，因此值得写进脚本与文档，让任何一次环境重建都自动绕开。

至此，从 .td 声明到 pass 再到可执行工具的完整链条闭合：第十三章的 TableGen 定义经本章的生成层变成 C++，第十四章的属性与第十五章的 pass 编进 GdnDialect 库，gdn-opt 把它们与全部树内 pass 装进同一个命令行。方言工程的全部源码加起来不足千行，承载的却是 L0 语义层的全部表达与验证入口。
