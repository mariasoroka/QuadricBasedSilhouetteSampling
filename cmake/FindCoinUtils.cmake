set(COINUTILS_INCLUDE_PATH
  ${CMAKE_SOURCE_DIR}/CoinUtils/include/coin-or)

find_library(COINUTILS_LIBRARY NAMES CoinUtils PATHS 
  ${CMAKE_SOURCE_DIR}/CoinUtils/lib
  /usr/lib 
  /usr/local/lib 
  /opt/local/lib)

if (COINUTILS_INCLUDE_PATH AND COINUTILS_LIBRARY)
  set(COINUTILS_FOUND TRUE)
endif ()
