file(GLOB SOURCE "*.cpp" "*.h")
# Historical note: an earlier corex-port iteration removed the
# `AffineBodyMaterial::apply_to / AffineBodyConstitution::create_material`
# wrappers from src/constitution/affine_body_constitution.cpp while
# keeping their declarations in the public header. To get the rest of
# the core test suite to build under the corex backend we used to skip
# the two test files that exercise that public API:
#
#     list(FILTER SOURCE EXCLUDE REGEX "/(constitution|object)\\.cpp$")
#
# The wrappers have since been restored in
# src/constitution/affine_body_constitution.cpp so both test files
# compile and link cleanly on both the NVIDIA and Corex paths. The
# filter is therefore no longer required and has been removed.
uipc_add_test(core ${SOURCE})
