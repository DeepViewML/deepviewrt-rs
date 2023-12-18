use crate::{engine::Engine, error::Error};
use deepviewrt_sys as ffi;
use std::{
    cell::Cell,
    ffi::c_void,
    io,
    ops::{Deref, DerefMut},
};

#[derive(Debug)]
pub enum TensorType {
    RAW = 0,
    STR = 1,
    I8 = 2,
    U8 = 3,
    I16 = 4,
    U16 = 5,
    I32 = 6,
    U32 = 7,
    I64 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    F64 = 12,
}

impl TryFrom<u32> for TensorType {
    type Error = ();

    fn try_from(value: u32) -> Result<TensorType, Self::Error> {
        match value {
            0 => return Ok(TensorType::RAW),
            1 => return Ok(TensorType::STR),
            2 => return Ok(TensorType::I8),
            3 => return Ok(TensorType::U8),
            4 => return Ok(TensorType::I16),
            5 => return Ok(TensorType::U16),
            6 => return Ok(TensorType::I32),
            7 => return Ok(TensorType::U32),
            8 => return Ok(TensorType::I64),
            9 => return Ok(TensorType::U64),
            10 => return Ok(TensorType::F16),
            11 => return Ok(TensorType::F32),
            12 => return Ok(TensorType::F64),
            _ => return Err(()),
        };
    }
}

pub struct Tensor {
    owned: bool,
    ptr: *mut ffi::NNTensor,
    engine: Cell<Option<Engine>>,
    scales: Option<Vec<f32>>,
}

pub struct TensorData<'a, T> {
    tensor: &'a Tensor,
    data: &'a [T],
}

impl<'a, T> Deref for TensorData<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

pub struct TensorDataMut<'a, T> {
    tensor: &'a mut Tensor,
    data: &'a mut [T],
}

impl<'a, T> Deref for TensorDataMut<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T> DerefMut for TensorDataMut<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.data
    }
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Deref for Tensor {
    type Target = ffi::NNTensor;

    fn deref(&self) -> &Self::Target {
        return unsafe { &*(self.ptr) };
    }
}

impl Tensor {
    pub fn new() -> Result<Self, Error> {
        let ptr = unsafe {
            ffi::nn_tensor_init(
                std::ptr::null::<c_void>() as *mut c_void,
                std::ptr::null::<ffi::nn_engine>() as *mut ffi::nn_engine,
            )
        };
        if ptr.is_null() {
            let err_kind = io::Error::last_os_error().kind();
            return Err(Error::IoError(err_kind));
        }

        return Ok(Self {
            owned: true,
            engine: Cell::new(None),
            ptr,
            scales: None,
        });
    }

    pub fn alloc(&self, ttype: TensorType, n_dims: i32, shape: &[i32; 3]) -> Result<(), Error> {
        let ttype_c_uint = (ttype as u32) as std::os::raw::c_uint;
        let ret = unsafe { ffi::nn_tensor_alloc(self.ptr, ttype_c_uint, n_dims, shape.as_ptr()) };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }

        return Ok(());
    }

    pub fn dequantize(&self, dest: &mut Self) -> Result<(), Error> {
        let ret = unsafe { ffi::nn_tensor_dequantize(dest.to_mut_ptr(), self.ptr) };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }

        return Ok(());
    }

    pub fn set_tensor_type(&self, tensor_type: TensorType) -> Result<(), Error> {
        let tensor_type_ = TensorType::try_from(tensor_type as u32).unwrap();
        let ret = unsafe { ffi::nn_tensor_set_type(self.ptr, tensor_type_ as ffi::NNTensorType) };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }
        return Ok(());
    }

    pub fn tensor_type(&self) -> TensorType {
        let ret = unsafe { ffi::nn_tensor_type(self.ptr) };
        return TensorType::try_from(ret).unwrap();
    }

    pub fn engine(&self) -> Option<&Engine> {
        let ret = unsafe { ffi::nn_tensor_engine(self.ptr) };
        if ret.is_null() {
            return None;
        }
        let engine = Engine::wrap(ret).unwrap();
        self.engine.set(Some(engine));
        return unsafe { (&*self.engine.as_ptr()).as_ref() };
    }

    pub fn shape(&self) -> &[i32] {
        let ret = unsafe { ffi::nn_tensor_shape(self.ptr) };
        let ra = unsafe { std::slice::from_raw_parts(ret, 4) };
        return ra;
    }

    pub fn dims(&self) -> i32 {
        return unsafe { ffi::nn_tensor_dims(self.ptr) };
    }

    pub fn volume(&self) -> i32 {
        return unsafe { ffi::nn_tensor_volume(self.ptr) };
    }

    pub fn size(&self) -> i32 {
        return unsafe { ffi::nn_tensor_size(self.ptr) };
    }

    pub fn axis(&self) -> i16 {
        return unsafe { ffi::nn_tensor_axis(self.ptr) as i16 };
    }

    pub fn zeros(&self) -> Result<&[i32], Error> {
        let mut zeros: usize = 0;
        let ret = unsafe { ffi::nn_tensor_zeros(self.ptr, &mut zeros as *mut usize) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from("zeros returned null")));
        }
        return unsafe { Ok(std::slice::from_raw_parts(ret, zeros)) };
    }

    pub fn set_scales(&mut self, scales: &[f32]) -> Result<(), Error> {
        self.scales = Some(scales.to_vec());
        if scales.len() < (self.axis() as usize) || scales.len() != 1 {
            return Err(Error::WrapperError(String::from(
                "scales should either have length of 1 or equal to channel_dimension (axis)",
            )));
        }
        unsafe {
            ffi::nn_tensor_set_scales(self.ptr, scales.len(), scales.as_ptr() as *const f32, 0)
        };
        return Ok(());
    }

    pub fn randomize(&mut self) -> Result<(), Error> {
        let err = unsafe { ffi::nn_tensor_randomize(self.ptr) };
        if err != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(err));
        }
        Ok(())
    }

    fn mapro(&self) -> Result<*const ::std::os::raw::c_void, Error> {
        let ret = unsafe { ffi::nn_tensor_mapro(self.ptr) };
        if ret.is_null() {
            return Err(Error::WrapperError("nn_tensor_mapro failed".to_string()));
        }
        return Ok(ret);
    }

    fn maprw(&self) -> Result<*mut ::std::os::raw::c_void, Error> {
        let ret = unsafe { ffi::nn_tensor_maprw(self.ptr) };
        if ret.is_null() {
            return Err(Error::WrapperError("nn_tensor_mapro failed".to_string()));
        }
        return Ok(ret);
    }

    pub fn mapro_u8<'a>(&'a self) -> Result<TensorData<'a, u8>, Error> {
        let ptr = self.mapro()? as *const u8;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn mapro_u16<'a>(&'a self) -> Result<TensorData<'a, u16>, Error> {
        let ptr = self.mapro()? as *const u16;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn mapro_u32<'a>(&'a self) -> Result<TensorData<'a, u32>, Error> {
        let ptr = self.mapro()? as *const u32;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn mapro_u64<'a>(&'a self) -> Result<TensorData<'a, u64>, Error> {
        let ptr = self.mapro()? as *const u64;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn mapro_i8<'a>(&'a self) -> Result<TensorData<'a, i8>, Error> {
        let ptr = self.mapro()? as *const i8;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn mapro_i16<'a>(&'a self) -> Result<TensorData<'a, i16>, Error> {
        let ptr = self.mapro()? as *const i16;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn mapro_i32<'a>(&'a self) -> Result<TensorData<'a, i32>, Error> {
        let ptr = self.mapro()? as *const i32;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn mapro_i64<'a>(&'a self) -> Result<TensorData<'a, i64>, Error> {
        let ptr = self.mapro()? as *const i64;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn mapro_f32<'a>(&'a self) -> Result<TensorData<'a, f32>, Error> {
        let ptr = self.mapro()? as *const f32;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn mapro_f64<'a>(&'a self) -> Result<TensorData<'a, f64>, Error> {
        let ptr = self.mapro()? as *const f64;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts(ptr, size as usize) };
        return Ok(TensorData {
            tensor: self,
            data: sret,
        });
    }

    pub fn maprw_f32<'a>(&'a mut self) -> Result<TensorDataMut<'a, f32>, Error> {
        let ptr = self.maprw()? as *mut f32;
        let size = self.size();
        let sret = unsafe { std::slice::from_raw_parts_mut(ptr, size as usize) };
        Ok(TensorDataMut {
            tensor: self,
            data: sret,
        })
    }

    unsafe fn unmap(&self) {
        unsafe { ffi::nn_tensor_unmap(self.ptr) };
    }

    pub unsafe fn from_ptr(ptr: *mut ffi::NNTensor, owned: bool) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::WrapperError(String::from("ptr is null")));
        }

        return Ok(Tensor {
            owned,
            engine: Cell::new(None),
            ptr,
            scales: None,
        });
    }

    pub fn to_mut_ptr(&self) -> *mut ffi::NNTensor {
        return self.ptr;
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if self.owned {
            unsafe {
                ffi::nn_tensor_release(self.ptr);
            };
        }
    }
}

impl<'a, T> Drop for TensorData<'a, T> {
    fn drop(&mut self) {
        unsafe { self.tensor.unmap() };
    }
}
