//! Assembling the Naga module: the top-level build entry points.

use super::builders::Builders;
use super::cfg::Cfg;
use super::emit::{append, push_emit, scalar};
use super::flow::uniform_values;
use super::image_cache::ImageCache;
use super::lowerer::Lowerer;
use super::memory::scalar_of;
use super::stage::{Stage, StageIo, scan_io};
use super::stage_io::{stage_arguments, stage_output};
use crate::error::TranslateError;
use crate::parse::{LlvmFunc, Ty};
use std::collections::{BTreeMap, HashMap, HashSet};

impl Builders {
    pub(crate) fn build(
        &mut self,
        func: &LlvmFunc,
        cfg: &Cfg<'_>,
        stage: Stage,
    ) -> Result<naga::Function, TranslateError> {
        crate::site::enter_function(&func.name);
        let mut function = naga::Function {
            name: Some(func.name.clone()),
            ..Default::default()
        };
        let io = scan_io(func, stage)?;
        let uniform = uniform_values(func);
        self.module_constants = func
            .globals
            .iter()
            .map(|global| (global.name.clone(), global.bytes.clone()))
            .collect();
        let args = stage_arguments(&mut self.module, func, stage, &io, &mut function)?;
        let stage_out = stage_output(&mut self.module, stage, &io, &mut function)?;
        let mut lowerer = Lowerer {
            builders: self,
            cfg,
            func: function,
            io,
            id_x: args.id_x,
            global_id: args.global_id,
            local_index: args.local_index,
            vertex_index: args.vertex_index,
            instance_index: args.instance_index,
            stage_out,
            position: args.position,
            inputs: args.inputs,
            phi_vars: HashMap::new(),
            spill: HashMap::new(),
            env: HashMap::new(),
            saw_return: false,
            divergent_depth: 0,
            uniform,
            exit_ids: HashMap::new(),
            exit_id_var: None,
            pointers: HashMap::new(),
            ptr_bases: HashMap::new(),
            constants: HashMap::new(),
            diverged: false,
            returns_value: false,
            images: ImageCache::default(),
        };
        if let Some((id_x, (_, name))) = args.id_x.zip(func.params.first()) {
            lowerer.env.insert(name.clone(), id_x);
        }
        lowerer.alloc_phis()?;
        lowerer.emit_top()?;
        let saw_return = lowerer.saw_return;
        let function = lowerer.func;
        if !saw_return {
            return Err(TranslateError::Parse(
                "function without lowered ret".to_string(),
            ));
        }
        Ok(function)
    }

    /// Lower a helper function: scalar parameters, scalar or void result.
    ///
    /// A helper has no stage interface and no invocation id; its body is the
    /// same subset as an entry's, and `ret` returns the value instead of
    /// storing it.
    pub(crate) fn build_helper(
        &mut self,
        func: &LlvmFunc,
        cfg: &Cfg<'_>,
    ) -> Result<naga::Function, TranslateError> {
        let mut function = naga::Function {
            name: Some(func.name.clone()),
            ..Default::default()
        };
        let mut args = Vec::new();
        for (index, (ty, name)) in func.params.iter().enumerate() {
            let ty = match ty {
                Ty::Float => scalar(&mut self.module, naga::ScalarKind::Float),
                Ty::I32 => scalar(&mut self.module, naga::ScalarKind::Uint),
                _ => return Err(TranslateError::unsupported()),
            };
            function.arguments.push(naga::FunctionArgument {
                name: Some(name.clone()),
                ty,
                binding: None,
            });
            let arg = append(
                &mut function,
                naga::Expression::FunctionArgument(index as u32),
            );
            push_emit(&mut function, &mut naga::Block::new(), arg);
            args.push((name.clone(), arg));
        }
        function.result = match func.ret_ty {
            Ty::Void => None,
            ret => Some(naga::FunctionResult {
                ty: scalar_of(&mut self.module, ret).ok_or_else(TranslateError::unsupported)?,
                binding: None,
            }),
        };
        let mut lowerer = Lowerer {
            builders: self,
            cfg,
            func: function,
            io: StageIo::default(),
            id_x: None,
            global_id: None,
            local_index: None,
            vertex_index: None,
            instance_index: None,
            stage_out: None,
            position: None,
            inputs: BTreeMap::new(),
            phi_vars: HashMap::new(),
            spill: HashMap::new(),
            env: HashMap::new(),
            saw_return: false,
            divergent_depth: 0,
            uniform: HashSet::new(),
            exit_ids: HashMap::new(),
            exit_id_var: None,
            pointers: HashMap::new(),
            ptr_bases: HashMap::new(),
            constants: HashMap::new(),
            diverged: false,
            returns_value: true,
            images: ImageCache::default(),
        };
        for (name, arg) in args {
            lowerer.env.insert(name, arg);
        }
        lowerer.alloc_phis()?;
        lowerer.emit_top()?;
        let saw_return = lowerer.saw_return;
        let function = lowerer.func;
        if !saw_return {
            return Err(TranslateError::Parse(
                "function without lowered ret".to_string(),
            ));
        }
        Ok(function)
    }
}
