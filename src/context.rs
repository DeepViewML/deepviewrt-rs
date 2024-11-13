use crate::{engine::Engine, error::Error, model, tensor::Tensor};
use deepviewrt_sys as ffi;
use std::{
    cell::{Cell, RefCell},
    ffi::CString,
    ptr,
};

pub struct Context {
    owned: bool,
    ptr: *mut ffi::NNContext,
    engine: Cell<Option<Engine>>,
    model: Vec<u8>,
    tensors: RefCell<Vec<(i32, Tensor)>>,
}

impl Context {
    pub fn new(
        engine: Option<Engine>,
        memory_size: usize,
        cache_size: usize,
    ) -> Result<Context, Error> {
        let ret = unsafe {
            ffi::nn_context_init(
                match &engine {
                    Some(engine) => engine.to_ptr_mut(),
                    None => ptr::null_mut(),
                },
                memory_size,
                ptr::null_mut(),
                cache_size,
                ptr::null_mut(),
            )
        };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from(
                "nn_context_init returned null",
            )));
        }
        let tensors_ref: Vec<(i32, Tensor)> = Vec::new();
        let tensors = RefCell::new(tensors_ref);
        Ok(Context {
            owned: true,
            ptr: ret,
            engine: Cell::new(engine),
            model: Vec::new(),
            tensors,
        })
    }

    /*
    pub fn cache(&self) -> Option<Tensor> {

    }

    pub fn mempool(&self) -> Option<Tensor> {

    }
    */

    pub fn engine(&self) -> Option<&Engine> {
        let engine_ptr = self.engine.as_ptr();
        if !engine_ptr.is_null() {
            let engine_ref = unsafe { &*engine_ptr };
            if engine_ref.is_some() {
                return engine_ref.as_ref();
            }
        }

        let ret = unsafe { ffi::nn_context_engine(self.ptr) };
        if ret.is_null() {
            return None;
        }
        let engine = Engine::wrap(ret).unwrap();
        self.engine.set(Some(engine));
        return unsafe { (*self.engine.as_ptr()).as_ref() };
    }

    pub fn model(&self) -> &Vec<u8> {
        &self.model
    }

    pub fn input(&self, index: usize) -> Result<&Tensor, Error> {
        let inputs = model::inputs(&self.model)?;
        self.tensor_index(inputs[index] as usize)
    }

    pub fn input_mut(&mut self, index: usize) -> Result<&mut Tensor, Error> {
        let inputs = model::inputs(&self.model)?;
        self.tensor_index_mut(inputs[index] as usize)
    }

    pub fn output(&self, index: usize) -> Result<&Tensor, Error> {
        let outputs = model::outputs(&self.model)?;
        self.tensor_index(outputs[index] as usize)
    }

    pub fn output_mut(&mut self, index: usize) -> Result<&mut Tensor, Error> {
        let outputs = model::outputs(&self.model)?;
        self.tensor_index_mut(outputs[index] as usize)
    }

    // pub fn model(&self) -> Option<&Model> {
    //     if !self.model.as_ptr().is_null() {
    //         let model_ref = unsafe { &*self.model.as_ptr() };
    //         if model_ref.is_some() {
    //             return model_ref.as_ref();
    //         }
    //     }

    //     let ret = unsafe { ffi::nn_context_model(self.ptr) };
    //     if ret.is_null() {
    //         return None;
    //     }
    //     let model = match unsafe { Model::try_from_ptr(ret) } {
    //         Ok(model) => Some(model),
    //         Err(e) => {
    //             eprintln!("{}", e);
    //             None
    //         }
    //     };
    //     self.model.set(model);

    //     let model_ref = unsafe { &*self.model.as_ptr() };
    //     if model_ref.is_some() {
    //         return model_ref.as_ref();
    //     }
    //     None
    // }

    pub fn load_model(&mut self, model: Vec<u8>) -> Result<(), Error> {
        self.unload_model();
        self.model = model;
        let ret = unsafe {
            ffi::nn_context_model_load(
                self.ptr,
                self.model.len(),
                self.model.as_ptr() as *const std::ffi::c_void,
            )
        };
        if ret != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(ret));
        }
        Ok(())
    }

    pub fn unload_model(&mut self) {
        unsafe { ffi::nn_context_model_unload(self.ptr) };
        let tensors_ref: Vec<(i32, Tensor)> = Vec::new();
        self.tensors = RefCell::new(tensors_ref);
        self.model = Vec::new();
        // self.model.set(None);
    }

    pub fn run(&self) -> Result<(), Error> {
        let err = unsafe { ffi::nn_context_run(self.ptr) };
        if err != ffi::NNError_NN_SUCCESS {
            return Err(Error::from(err));
        }
        Ok(())
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor, Error> {
        let cname = match CString::new(name) {
            Ok(cname) => cname,
            Err(e) => return Err(Error::WrapperError(e.to_string())),
        };

        let ret = unsafe { ffi::nn_context_tensor(self.ptr, cname.into_raw()) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from("No tensor found")));
        }

        // let index = match self.model() {
        //     Some(model) => model.layer_lookup(name),
        //     None => return Err(Error::WrapperError(String::from("Could not get
        // index"))), }
        // .unwrap();

        let cname = CString::new(name).unwrap();
        let tensor_ptr = unsafe { ffi::nn_context_tensor(self.ptr, cname.into_raw()) };
        if tensor_ptr.is_null() {
            return Err(Error::WrapperError(format!("tensor not found: {}", name)));
        }

        let tensor = unsafe { Tensor::from_ptr(tensor_ptr, false).unwrap() };
        Ok(tensor)
    }

    pub fn tensor_index_mut(&mut self, index: usize) -> Result<&mut Tensor, Error> {
        let ret = unsafe { ffi::nn_context_tensor_index(self.ptr, index) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from("No tensor found")));
        }
        let tensor = unsafe { Tensor::from_ptr(ret, false).unwrap() };

        match self.tensors.try_borrow_mut() {
            Ok(mut borrowed) => {
                borrowed.push((index as i32, tensor));
            }
            Err(e) => {
                return Err(Error::WrapperError(e.to_string()));
            }
        }
        let tensors_ref = self.tensors.get_mut();
        {
            for (index_, tensor) in tensors_ref {
                if index_ == &(index as i32) {
                    return Ok(tensor);
                }
            }

            Err(Error::WrapperError(String::from("Tensor not found")))
        }
    }

    pub fn tensor_index(&self, index: usize) -> Result<&Tensor, Error> {
        let ret = unsafe { ffi::nn_context_tensor_index(self.ptr, index) };
        if ret.is_null() {
            return Err(Error::WrapperError(String::from("No tensor found")));
        }
        let tensor = unsafe { Tensor::from_ptr(ret, false).unwrap() };

        match self.tensors.try_borrow_mut() {
            Ok(mut borrowed) => {
                borrowed.push((index as i32, tensor));
            }
            Err(e) => {
                return Err(Error::WrapperError(e.to_string()));
            }
        }
        let tensors_ref = unsafe { &*self.tensors.as_ptr() };
        {
            for (index_, tensor) in tensors_ref {
                if index_ == &(index as i32) {
                    return Ok(tensor);
                }
            }

            Err(Error::WrapperError(String::from("Tensor not found")))
        }
    }

    pub unsafe fn from_ptr(ptr: *mut ffi::NNContext) -> Result<Self, Error> {
        if ptr.is_null() {
            return Err(Error::WrapperError(String::from("ptr is null")));
        }

        let tensors_ref: Vec<(i32, Tensor)> = Vec::new();
        let tensors = RefCell::new(tensors_ref);

        Ok(Self {
            owned: false,
            ptr,
            engine: Cell::new(None),
            model: Vec::new(),
            // model: Cell::new(None),
            tensors,
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.owned {
            unsafe { ffi::nn_context_release(self.ptr) };
        }
    }
}
