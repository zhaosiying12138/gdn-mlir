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
