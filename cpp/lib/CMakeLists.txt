add_library(implicit2d SHARED
    src/ImplicitFunctionBuilder.cxx
    src/JsonParser.cxx
)

target_include_directories(implicit2d
    PUBLIC
    "${PROJECT_SOURCE_DIR}/lib/include"
)
