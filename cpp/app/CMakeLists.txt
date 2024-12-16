add_executable(app
    main.cxx
)

target_link_libraries(app
    PRIVATE
    implicit2d
)
