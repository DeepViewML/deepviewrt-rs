#!/bin/sh

bindgen --allowlist-function 'nn_.*' deepview_rt.h > src/ffi.rs
