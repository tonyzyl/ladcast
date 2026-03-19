set ext [info sharedlibextension]
if {${ext} eq ".dylib"} {
pkg_mkIndex . libsymdifftcl${ext}
} else {
pkg_mkIndex -lazy .
}
