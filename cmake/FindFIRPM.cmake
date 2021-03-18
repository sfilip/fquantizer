find_path(FIRPM_INCLUDE
    NAMES
    firpm.h
    PATHS
    $ENV{GMPDIR}
    ${INCLUDE_INSTALL_DIR}
)

find_library(FIRPM_LIBRARY firpm PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FIRPM DEFAULT_MSG
                                  FIRPM_INCLUDE FIRPM_LIBRARY)
mark_as_advanced(FIRPM_INCLUDE FIRPM_LIBRARY)
