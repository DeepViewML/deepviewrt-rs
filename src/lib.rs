use deepviewrt_sys as ffi;
pub mod context;
pub mod engine;
pub mod error;
pub mod model;
pub mod tensor;
use std::ffi::CStr;

pub enum QuantizationType {
    TypeNone = 0,
    TypeAffinePerTensor = 1,
    TypeAffinePerChannel = 2,
}

pub fn version() -> &'static str {
    let version = unsafe { ffi::nn_version() };
    let ret_cstr = unsafe { CStr::from_ptr(version) };
    ret_cstr.to_str().unwrap()
}

pub fn init() {}
