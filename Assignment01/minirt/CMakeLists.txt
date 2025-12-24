file(GLOB_RECURSE SOURCES *.cpp)
file(GLOB_RECURSE HEADERS *.h)

add_library(minirt ${SOURCES})

target_include_directories(minirt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

install(TARGETS minirt DESTINATION lib)
install(DIRECTORY . DESTINATION include/minirt FILES_MATCHING PATTERN *.h)