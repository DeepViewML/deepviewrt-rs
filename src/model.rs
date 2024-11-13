use crate::error::Error;
use deepviewrt_sys as ffi;
use std::{
    ffi::{c_void, CStr, CString},
    slice,
};

pub fn name(model: &[u8]) -> Result<&str, Error> {
    let ret = unsafe { ffi::nn_model_name(model.as_ptr() as *const c_void) };
    if ret.is_null() {
        return Err(Error::WrapperError(String::from("nn_model_name is null")));
    }
    let cstr = unsafe { CStr::from_ptr(ret) };
    match cstr.to_str() {
        Ok(s) => Ok(s),
        Err(e) => Err(Error::WrapperError(e.to_string())),
    }
}

/*
pub fn serial(&self) -> Result<&str, Error> {

}
*/

// pub fn label_count(&self) -> Result<i32, Error> {
//     let ret = unsafe { ffi::nn_model_label_count(self.ptr) };
//     if ret == 0 {
//         return Err(Error::WrapperError(String::from(
//             "No labels or model is invalid",
//         )));
//     }
//     Ok(ret)
// }

// pub fn label(&self, index: i32) -> Result<&str, Error> {
//     let ret = unsafe { ffi::nn_model_label(self.ptr, index) };
//     if ret.is_null() {
//         return Err(Error::WrapperError(String::from("label was NULL")));
//     }
//     let cstr = unsafe { CStr::from_ptr(ret) };
//     match cstr.to_str() {
//         Ok(s) => Ok(s),
//         Err(e) => Err(Error::WrapperError(e.to_string())),
//     }
// }

pub fn inputs(model: &[u8]) -> Result<Vec<u32>, Error> {
    let mut len: usize = 0;
    let indices =
        unsafe { ffi::nn_model_inputs(model.as_ptr() as *const c_void, &mut len as *mut usize) };
    if indices.is_null() {
        return Err(Error::WrapperError(String::from(
            "could not get model inputs",
        )));
    }
    Ok(unsafe { slice::from_raw_parts(indices, len) }.to_vec())
}

pub fn outputs(model: &[u8]) -> Result<Vec<u32>, Error> {
    let mut len: usize = 0;

    let indices =
        unsafe { ffi::nn_model_outputs(model.as_ptr() as *const c_void, &mut len as *mut usize) };
    if indices.is_null() {
        return Err(Error::WrapperError(String::from(
            "could not get model outputs",
        )));
    }
    Ok(unsafe { slice::from_raw_parts(indices, len) }.to_vec())
}

pub fn layer_count(model: &[u8]) -> usize {
    unsafe { ffi::nn_model_layer_count(model.as_ptr() as *const c_void) }
}

pub fn layer_name(model: &[u8], index: usize) -> Result<&str, Error> {
    let ret = unsafe { ffi::nn_model_layer_name(model.as_ptr() as *const c_void, index) };
    if ret.is_null() {
        return Err(Error::WrapperError(String::from(
            "nn_model_layer_name returned null",
        )));
    }
    let cstr = unsafe { CStr::from_ptr(ret) };
    match cstr.to_str() {
        Ok(s) => Ok(s),
        Err(e) => Err(Error::WrapperError(e.to_string())),
    }
}

pub fn layer_lookup(model: &[u8], name: &str) -> Result<i32, Error> {
    let name = match CString::new(name) {
        Ok(s) => s,
        Err(e) => return Err(Error::WrapperError(e.to_string())),
    };

    let ret =
        unsafe { ffi::nn_model_layer_lookup(model.as_ptr() as *const c_void, name.into_raw()) };
    if ret == -1 {
        return Err(Error::WrapperError(String::from(
            "Could not get index of layer",
        )));
    }
    Ok(ret)
}

// pub fn layer_type(&self, index: usize) -> Result<&str, Error> {
//     let ret = unsafe { ffi::nn_model_layer_type(self.ptr, index) };
//     if ret.is_null() {
//         return Err(Error::WrapperError(String::from("index out of range")));
//     }

//     let cstr = unsafe { CStr::from_ptr(ret) };
//     match cstr.to_str() {
//         Ok(s) => Ok(s),
//         Err(e) => Err(Error::WrapperError(e.to_string())),
//     }
// }

// pub fn layer_type_id(&self, index: usize) -> Result<i16, Error> {
//     let ret = unsafe { ffi::nn_model_layer_type_id(self.ptr, index) };
//     if ret == 0 {
//         return Err(Error::WrapperError(String::from("index out of range")));
//     }

//     return Ok(ret);
// }

// pub fn layer_datatype(&self, index: usize) -> Result<&str, Error> {
//     let ret = unsafe { ffi::nn_model_layer_datatype(self.ptr, index) };
//     if ret.is_null() {
//         return Err(Error::WrapperError(String::from("index out of range")));
//     }

//     let cstr = unsafe { CStr::from_ptr(ret) };
//     match cstr.to_str() {
//         Ok(s) => Ok(s),
//         Err(e) => Err(Error::WrapperError(e.to_string())),
//     }
// }

// pub fn layer_datatype_id(&self, index: usize) -> Result<TensorType, Error> {
//     let ret = unsafe { ffi::nn_model_layer_datatype_id(self.ptr, index) };
//     if ret == 0 {
//         return Err(Error::WrapperError(String::from(
//             "Model is invalid or Index is out of range",
//         )));
//     }
//     match TensorType::try_from(ret as u32) {
//         Ok(tensor_type) => {
//             return Ok(tensor_type);
//         }
//         Err(_) => {
//             return Err(Error::WrapperError(String::from("Not a valid
// TensorType")));         }
//     }
// }

// pub fn layer_zeros(&self, index: usize) -> Result<&[i32], Error> {
//     let mut n_zeros: isize = -1;
//     let ret = unsafe {
//         ffi::nn_model_layer_zeros(self.ptr, index, &mut n_zeros as *mut isize
// as *mut usize)     };
//     if ret.is_null() || n_zeros == -1 {
//         return Err(Error::WrapperError(String::from("Index out of range")));
//     }
//     return unsafe { Ok(std::slice::from_raw_parts(ret, n_zeros as usize)) };
// }

// pub fn layer_scales(&self, index: usize) -> Result<&[f32], Error> {
//     let mut n_scales: isize = -1;
//     let ret = unsafe {
//         ffi::nn_model_layer_scales(self.ptr, index, &mut n_scales as *mut
// isize as *mut usize)     };
//     if ret.is_null() || n_scales == -1 {
//         return Err(Error::WrapperError(String::from("Index out of range")));
//     }
//     return unsafe { Ok(std::slice::from_raw_parts(ret, n_scales as usize)) };
// }

// pub fn layer_axis(&self, index: usize) -> Result<i32, Error> {
//     let ret = unsafe { ffi::nn_model_layer_axis(self.ptr, index) };
//     if ret == 0 {
//         //TODO 0 indicates faliure, but docs say -1 (should be clarified
//         return Err(Error::WrapperError(String::from("Index out of range")));
//     }
//     return Ok(ret);
// }

// pub fn layer_shape(&self, index: usize) -> Result<&[i32], Error> {
//     let n_dims: isize = -1;
//     let ret = unsafe { ffi::nn_model_layer_shape(self.ptr, index, &mut
// (n_dims as usize)) };     if ret.is_null() || n_dims == -1 {
//         return Err(Error::WrapperError(String::from("Index out of range")));
//     }
//     return unsafe { Ok(std::slice::from_raw_parts(ret, n_dims as usize)) };
// }

/*
pub fn layer_inputs(&self, index: usize) {}

pub fn layer_parameter() {}

pub fn layer_parameter_shape() {}
*/
