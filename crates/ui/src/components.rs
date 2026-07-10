use std::sync::{Arc, Mutex};

use boa_engine::js_string;
use boa_engine::native_function::NativeFunction;
use boa_engine::object::builtins::JsArray;
use boa_engine::object::JsObject;
use boa_engine::{Context, JsNativeError, JsResult, JsValue};
use ornis_core::{Entity, SmartStore};

/// A UI component holding style properties for an entity.
#[derive(Debug, Clone)]
pub struct UIStyle {
    pub color: [f32; 4],
    pub font_size: f32,
    pub width: f32,
    pub height: f32,
}

impl Default for UIStyle {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 1.0],
            font_size: 16.0,
            width: 100.0,
            height: 20.0,
        }
    }
}

impl UIStyle {
    pub fn to_js(&self, ctx: &mut Context) -> JsResult<JsValue> {
        let obj = JsObject::with_object_proto(ctx.intrinsics());
        let color = JsArray::from_iter(
            self.color.iter().map(|&v| JsValue::from(f64::from(v))),
            ctx,
        );
        obj.set(js_string!("color"), color, false, ctx)?;
        obj.set(
            js_string!("font_size"),
            JsValue::from(f64::from(self.font_size)),
            false,
            ctx,
        )?;
        obj.set(
            js_string!("width"),
            JsValue::from(f64::from(self.width)),
            false,
            ctx,
        )?;
        obj.set(
            js_string!("height"),
            JsValue::from(f64::from(self.height)),
            false,
            ctx,
        )?;
        Ok(obj.into())
    }

    pub fn from_js(value: &JsValue, ctx: &mut Context) -> JsResult<Self> {
        let obj = value
            .as_object()
            .ok_or_else(|| JsNativeError::typ().with_message("UIStyle: expected object"))?;

        let color = if let Ok(color_val) = obj.get(js_string!("color"), ctx) {
            if let Some(color_arr) = color_val.as_object() {
                if let Ok(arr) = JsArray::from_object(color_arr.clone()) {
                    let mut c = [1.0f32; 4];
                    for (i, item) in c.iter_mut().enumerate() {
                        if let Ok(v) = arr.get(i, ctx) {
                            *item = v.to_number(ctx).unwrap_or(1.0) as f32;
                        }
                    }
                    c
                } else {
                    [1.0; 4]
                }
            } else {
                [1.0; 4]
            }
        } else {
            [1.0; 4]
        };

        let font_size = obj
            .get(js_string!("font_size"), ctx)
            .ok()
            .and_then(|v| v.to_number(ctx).ok())
            .map(|v| v as f32)
            .unwrap_or(16.0);

        let width = obj
            .get(js_string!("width"), ctx)
            .ok()
            .and_then(|v| v.to_number(ctx).ok())
            .map(|v| v as f32)
            .unwrap_or(100.0);

        let height = obj
            .get(js_string!("height"), ctx)
            .ok()
            .and_then(|v| v.to_number(ctx).ok())
            .map(|v| v as f32)
            .unwrap_or(20.0);

        Ok(UIStyle { color, font_size, width, height })
    }
}

/// Bridge between JS and the ECS SmartStore.
#[derive(Clone)]
pub struct EcsBridge {
    store: Arc<Mutex<SmartStore>>,
    /// Cached generations for entity IDs (for creating valid Entity handles from JS).
    generations: Arc<Mutex<Vec<u32>>>,
}

impl EcsBridge {
    pub fn new() -> Self {
        let mut store = SmartStore::new();
        store.register::<UIStyle>();
        Self {
            store: Arc::new(Mutex::new(store)),
            generations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn create_entity(&self) -> u32 {
        let store = self.store.lock().unwrap();
        let entity = store.create_entity();
        let id = entity.id() as usize;
        let mut gens = self.generations.lock().unwrap();
        if id >= gens.len() {
            gens.resize(id + 1, 0);
        }
        gens[id] = entity.generation();
        entity.id()
    }

    pub fn destroy_entity(&self, id: u32) {
        let (entity, store) = {
            let gens = self.generations.lock().unwrap();
            let idx = id as usize;
            let generation = gens.get(idx).copied().unwrap_or(0);
            (Entity::new_with_gen(id, generation), self.store.lock().unwrap())
        };
        store.destroy_entity(entity);
    }

    fn make_entity(&self, id: u32) -> Entity {
        let gens = self.generations.lock().unwrap();
        let idx = id as usize;
        let generation = gens.get(idx).copied().unwrap_or(0);
        Entity::new_with_gen(id, generation)
    }

    pub fn set_uistyle(&self, entity_id: u32, value: &JsValue, ctx: &mut Context) -> JsResult<()> {
        let entity = self.make_entity(entity_id);
        let style = UIStyle::from_js(value, ctx)?;
        let mut store = self.store.lock().unwrap();
        store.insert::<UIStyle>(entity, style);
        Ok(())
    }

    pub fn get_uistyle(&self, entity_id: u32, ctx: &mut Context) -> JsResult<JsValue> {
        let entity = self.make_entity(entity_id);
        let store = self.store.lock().unwrap();
        if let Some(lane) = store.read_lane::<UIStyle>() {
            if let Some(style) = lane.get(entity) {
                return style.to_js(ctx);
            }
        }
        Ok(JsValue::undefined())
    }

    pub fn has_uistyle(&self, entity_id: u32) -> bool {
        let entity = self.make_entity(entity_id);
        let store = self.store.lock().unwrap();
        store.read_lane::<UIStyle>().map(|l| l.contains(entity)).unwrap_or(false)
    }

    pub fn register_js_functions(&self, ctx: &mut Context) {
        let bridge = self.clone();

        let create_entity_fn = unsafe {
            NativeFunction::from_closure(move |_, _args, _ctx| {
                Ok(JsValue::from(bridge.create_entity()))
            })
        };

        let destroy_entity_fn = {
            let bridge = self.clone();
            unsafe {
                NativeFunction::from_closure(move |_, args, ctx| {
                    let id = args
                        .first()
                        .ok_or_else(|| JsNativeError::typ().with_message("destroyEntity: entity id required"))?
                        .to_number(ctx)
                        .map_err(|_| JsNativeError::typ().with_message("destroyEntity: expected number"))? as u32;
                    bridge.destroy_entity(id);
                    Ok(JsValue::undefined())
                })
            }
        };

        let get_uistyle_fn = {
            let bridge = self.clone();
            unsafe {
                NativeFunction::from_closure(move |_, args, ctx| {
                    let id = args
                        .first()
                        .ok_or_else(|| JsNativeError::typ().with_message("getUIStyle: entity id required"))?
                        .to_number(ctx)
                        .map_err(|_| JsNativeError::typ().with_message("getUIStyle: expected number"))? as u32;
                    bridge.get_uistyle(id, ctx)
                })
            }
        };

        let set_uistyle_fn = {
            let bridge = self.clone();
            unsafe {
                NativeFunction::from_closure(move |_, args, ctx| {
                    if args.len() < 2 {
                        return Err(JsNativeError::typ()
                            .with_message("setUIStyle: entityId, data required")
                            .into());
                    }
                    let id = args[0]
                        .to_number(ctx)
                        .map_err(|_| JsNativeError::typ().with_message("setUIStyle: entityId must be a number"))?
                        as u32;
                    bridge.set_uistyle(id, &args[1], ctx).map(|_| JsValue::undefined())
                })
            }
        };

        let has_uistyle_fn = {
            let bridge = self.clone();
            unsafe {
                NativeFunction::from_closure(move |_, args, ctx| {
                    let id = args
                        .first()
                        .ok_or_else(|| JsNativeError::typ().with_message("hasUIStyle: entity id required"))?
                        .to_number(ctx)
                        .map_err(|_| JsNativeError::typ().with_message("hasUIStyle: expected number"))? as u32;
                    Ok(JsValue::from(bridge.has_uistyle(id)))
                })
            }
        };

        let realm = ctx.realm().clone();
        let ornis = JsObject::with_object_proto(ctx.intrinsics());
        let _ = ornis.set(js_string!("createEntity"), JsValue::from(create_entity_fn.to_js_function(&realm)), false, ctx);
        let _ = ornis.set(js_string!("destroyEntity"), JsValue::from(destroy_entity_fn.to_js_function(&realm)), false, ctx);
        let _ = ornis.set(js_string!("getUIStyle"), JsValue::from(get_uistyle_fn.to_js_function(&realm)), false, ctx);
        let _ = ornis.set(js_string!("setUIStyle"), JsValue::from(set_uistyle_fn.to_js_function(&realm)), false, ctx);
        let _ = ornis.set(js_string!("hasUIStyle"), JsValue::from(has_uistyle_fn.to_js_function(&realm)), false, ctx);

        let global = ctx.global_object();
        let _ = global.set(js_string!("Ornis"), JsValue::from(ornis), false, ctx);
    }
}

impl Default for EcsBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> Context {
        Context::default()
    }

    #[test]
    fn test_uistyle_roundtrip() {
        let mut ctx = make_ctx();
        let style = UIStyle {
            color: [1.0, 0.0, 0.0, 1.0],
            font_size: 24.0,
            width: 300.0,
            height: 80.0,
        };
        let js_val = style.to_js(&mut ctx).unwrap();
        let got = UIStyle::from_js(&js_val, &mut ctx).unwrap();
        assert_eq!(got.color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(got.font_size, 24.0);
        assert_eq!(got.width, 300.0);
        assert_eq!(got.height, 80.0);
    }

    #[test]
    fn test_bridge_create_entity() {
        let bridge = EcsBridge::new();
        let id = bridge.create_entity();
        assert!(bridge.make_entity(id).id() == id);
    }

    #[test]
    fn test_bridge_uistyle_crud() {
        let bridge = EcsBridge::new();
        let eid = bridge.create_entity();
        let mut ctx = make_ctx();

        assert!(!bridge.has_uistyle(eid));

        let style = UIStyle {
            color: [0.0, 1.0, 0.0, 1.0],
            font_size: 18.0,
            width: 200.0,
            height: 50.0,
        };
        let js_val = style.to_js(&mut ctx).unwrap();
        bridge.set_uistyle(eid, &js_val, &mut ctx).unwrap();

        assert!(bridge.has_uistyle(eid));

        let got_js = bridge.get_uistyle(eid, &mut ctx).unwrap();
        let got = UIStyle::from_js(&got_js, &mut ctx).unwrap();
        assert_eq!(got.color, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(got.font_size, 18.0);

        bridge.destroy_entity(eid);
        assert!(!bridge.has_uistyle(eid));
    }

    #[test]
    fn test_bridge_missing_entity() {
        let bridge = EcsBridge::new();
        let mut ctx = make_ctx();

        // Non-existent entity ID
        let got = bridge.get_uistyle(999, &mut ctx).unwrap();
        assert!(got.is_undefined());
        assert!(!bridge.has_uistyle(999));
    }
}
