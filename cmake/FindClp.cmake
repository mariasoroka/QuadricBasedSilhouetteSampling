set(CLP_INCLUDE_PATH 
  ${CMAKE_SOURCE_DIR}/Clp/include/coin-or)

find_library(CLP_LIBRARY NAMES Clp PATHS 
  ${CMAKE_SOURCE_DIR}/Clp/lib
  /usr/lib 
  /usr/local/lib 
  /opt/local/lib)

if (CLP_INCLUDE_PATH AND CLP_LIBRARY)
  set(CLP_FOUND TRUE)
endif ()
