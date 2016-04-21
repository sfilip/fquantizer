find_path(FPLLL_INCLUDE
    NAMES
    fplll.h
    PATHS
    $ENV{GMPDIR}
    ${INCLUDE_INSTALL_DIR}
)

find_library(FPLLL_LIBRARY fplll PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FPLLL DEFAULT_MSG
                                  FPLLL_INCLUDE FPLLL_LIBRARY)
mark_as_advanced(FPLLL_INCLUDE FPLLL_LIBRARY)
