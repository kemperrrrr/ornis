//! Render graph — pass orchestration layer (Phase 0).
//!
//! An immediate-mode graph in the spirit of Frostbite FrameGraph and
//! Ponies&Light: passes are declared in execution order, and each pass
//! declares which resources it reads and writes. The graph computes
//! resource lifetimes (transient windows `[first_use, last_use]`) and
//! assigns them to pool slots so that non-overlapping resources with the
//! same specification share one slot (object-level aliasing).
//!
//! Model:
//! - `RenderGraph::build()` → `GraphLayout` — pure logic, no GPU needed;
//! - `RenderGraph::execute()` yields one [`PassContext`] per pass
//!   (insertion order; disabled passes are skipped);
//! - creating real `wgpu::Texture` objects per slot is the executor's job
//!   (Phase 1). On wgpu, barriers and layout transitions are handled by
//!   wgpu itself, so the graph owns lifetimes and pooling, not
//!   synchronization.
//!
//! Invariants (panic with a clear message when violated):
//! - a resource must not be read before it is written (imported resources
//!   are exempt);
//! - unknown resource/pass ids are errors;
//! - within a single pass, no two live resources may share a pool slot
//!   (guaranteed by construction).

use std::collections::HashMap;

/// Texture size policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizePolicy {
    /// Size matches the surface (swapchain) size.
    MatchSurface,
    /// Fixed size.
    Fixed { width: u32, height: u32 },
}

/// Texture specification — the pool reuse key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureSpec {
    pub format: wgpu::TextureFormat,
    pub samples: u32,
    pub size: SizePolicy,
}

/// Logical resource handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u32);

/// Logical pass handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassId(pub u32);

/// Per-resource information in a layout.
#[derive(Debug)]
pub struct ResourceLayout {
    pub id: ResourceId,
    pub name: String,
    pub spec: TextureSpec,
    /// Index of the first pass that uses the resource; `usize::MAX` if unused.
    pub first_use: usize,
    /// Index of the last pass that uses the resource; `0` if unused.
    pub last_use: usize,
    /// Pool slot (`None` — the resource is not used by any enabled pass).
    pub slot: Option<usize>,
}

impl ResourceLayout {
    /// Whether the resource is alive at the pass with `pass_index`.
    pub fn alive_at(&self, pass_index: usize) -> bool {
        self.first_use != usize::MAX && self.first_use <= pass_index && pass_index <= self.last_use
    }
}

/// Pool slot: a group of resources with the same [`TextureSpec`] whose
/// lifetime windows do not overlap.
#[derive(Debug)]
pub struct PoolSlot {
    pub index: usize,
    pub spec: TextureSpec,
    /// Resources sharing the slot (non-overlapping windows).
    pub resources: Vec<ResourceId>,
    pub first_pass: usize,
    pub last_pass: usize,
}

/// A pass in the executable layout.
#[derive(Debug)]
pub struct PassLayout {
    pub id: PassId,
    pub name: String,
    /// Resources read by the pass.
    pub reads: Vec<ResourceId>,
    /// Resources written by the pass; `Some(Color)` carries a clear value.
    pub writes: Vec<(ResourceId, Option<wgpu::Color>)>,
}

/// Result of `RenderGraph::build()` — the computed frame layout.
#[derive(Debug)]
pub struct GraphLayout {
    surface_size: (u32, u32),
    /// Passes in execution order (insertion order, disabled passes dropped).
    passes: Vec<PassLayout>,
    /// Resources (parallel to `RenderGraph::resources`).
    resources: Vec<ResourceLayout>,
    /// Pool slots.
    slots: Vec<PoolSlot>,
    /// Live resources per pass (by index into `passes`).
    pass_alive: Vec<Vec<ResourceId>>,
}

impl GraphLayout {
    /// Textual layout dump for debugging/reporting.
    pub fn debug_dump(&self) -> String {
        let mut s = format!(
            "render graph: {} passes, {} resources, {} pool slots (surface {:?})\n",
            self.passes.len(),
            self.resources.len(),
            self.slots.len(),
            self.surface_size
        );
        for (i, pass) in self.passes.iter().enumerate() {
            let reads: Vec<&str> = pass
                .reads
                .iter()
                .map(|&r| self.resources[r.0 as usize].name.as_str())
                .collect();
            let writes: Vec<&str> = pass
                .writes
                .iter()
                .map(|&(r, _)| self.resources[r.0 as usize].name.as_str())
                .collect();
            s += &format!(
                "  pass {i} '{}' read[{}] write[{}]\n",
                pass.name,
                reads.join(", "),
                writes.join(", ")
            );
        }
        for rl in &self.resources {
            if rl.first_use == usize::MAX {
                s += &format!("  resource '{}' UNUSED\n", rl.name);
            } else {
                s += &format!(
                    "  resource '{}' ({:?}) passes {}..={} slot {:?}\n",
                    rl.name, rl.spec, rl.first_use, rl.last_use, rl.slot
                );
            }
        }
        for slot in &self.slots {
            let names: Vec<&str> = slot
                .resources
                .iter()
                .map(|&r| self.resources[r.0 as usize].name.as_str())
                .collect();
            s += &format!(
                "  slot #{} {:?} passes {}..={}: {}\n",
                slot.index,
                slot.spec,
                slot.first_pass,
                slot.last_pass,
                names.join(", ")
            );
        }
        s
    }
}

#[derive(Debug)]
struct ResourceNode {
    name: String,
    spec: TextureSpec,
    /// Imported (external) resource: the "first touch must be a write"
    /// rule does not apply.
    imported: bool,
}

#[derive(Debug)]
struct PassNode {
    name: String,
    reads: Vec<ResourceId>,
    writes: Vec<(ResourceId, Option<wgpu::Color>)>,
    enabled: bool,
}

/// The pass graph being assembled.
#[derive(Debug)]
pub struct RenderGraph {
    resources: Vec<ResourceNode>,
    passes: Vec<PassNode>,
    surface_size: (u32, u32),
}

impl RenderGraph {
    /// Creates an empty graph; `surface_size` feeds `SizePolicy::MatchSurface`.
    pub fn new(surface_size: (u32, u32)) -> Self {
        Self {
            resources: Vec::new(),
            passes: Vec::new(),
            surface_size,
        }
    }

    /// Updates the surface size (window resize) before the next `build()`.
    pub fn set_surface_size(&mut self, width: u32, height: u32) {
        self.surface_size = (width, height);
    }

    /// Registers a graph-owned resource (texture).
    pub fn create_resource(&mut self, name: impl Into<String>, spec: TextureSpec) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceNode {
            name: name.into(),
            spec,
            imported: false,
        });
        id
    }

    /// Registers an imported (external) resource that passes only read
    /// (e.g. an uploaded shadow map). The "first touch must be a write"
    /// rule does not apply to it.
    pub fn import_resource(&mut self, name: impl Into<String>, spec: TextureSpec) -> ResourceId {
        let id = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceNode {
            name: name.into(),
            spec,
            imported: true,
        });
        id
    }

    /// Starts declaring a pass; passes execute in insertion order.
    pub fn add_pass(&mut self, name: impl Into<String>) -> PassBuilder<'_> {
        let id = PassId(self.passes.len() as u32);
        self.passes.push(PassNode {
            name: name.into(),
            reads: Vec::new(),
            writes: Vec::new(),
            enabled: true,
        });
        PassBuilder { graph: self, id }
    }

    /// Enables/disables a pass (culling): a disabled pass is dropped from
    /// the layout, and its resources get no slots unless used elsewhere.
    ///
    /// # Panics
    /// Panics if the pass is unknown.
    pub fn set_pass_enabled(&mut self, id: PassId, enabled: bool) {
        let node = self
            .passes
            .get_mut(id.0 as usize)
            .unwrap_or_else(|| panic!("unknown pass {id:?}"));
        node.enabled = enabled;
    }

    fn resolve_resource(&self, id: ResourceId, pass_name: &str) -> &ResourceNode {
        self.resources
            .get(id.0 as usize)
            .unwrap_or_else(|| panic!("unknown resource {id:?} in pass '{pass_name}'"))
    }

    /// Computes the frame layout (lifetimes + pool slots). Recomputes from
    /// the current graph state on every call.
    ///
    /// # Panics
    /// Panics if invariants are violated (read-before-write, etc.).
    pub fn build(&self) -> GraphLayout {
        self.compute_layout()
    }

    fn compute_layout(&self) -> GraphLayout {
        let enabled: Vec<usize> = self
            .passes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.enabled)
            .map(|(i, _)| i)
            .collect();

        let passes: Vec<PassLayout> = enabled
            .iter()
            .map(|&i| {
                let node = &self.passes[i];
                PassLayout {
                    id: PassId(i as u32),
                    name: node.name.clone(),
                    reads: node.reads.clone(),
                    writes: node.writes.clone(),
                }
            })
            .collect();

        let mut resources: Vec<ResourceLayout> = self
            .resources
            .iter()
            .enumerate()
            .map(|(i, node)| ResourceLayout {
                id: ResourceId(i as u32),
                name: node.name.clone(),
                spec: node.spec,
                first_use: usize::MAX,
                last_use: 0,
                slot: None,
            })
            .collect();

        // Lifetimes over enabled passes.
        for (pi, pass) in passes.iter().enumerate() {
            for rid in pass.reads.iter().chain(pass.writes.iter().map(|(r, _)| r)) {
                let rl = &mut resources[rid.0 as usize];
                rl.first_use = rl.first_use.min(pi);
                rl.last_use = rl.last_use.max(pi);
            }
        }

        // "First touch must be a write" rule (imported resources exempt).
        for (pi, pass) in passes.iter().enumerate() {
            for &rid in &pass.reads {
                let node = &self.resources[rid.0 as usize];
                let rl = &resources[rid.0 as usize];
                if !node.imported && rl.first_use == pi {
                    let written_earlier = pass.writes.iter().any(|(w, _)| *w == rid);
                    let first_write = passes[..pi]
                        .iter()
                        .any(|p| p.writes.iter().any(|(w, _)| *w == rid));
                    if !written_earlier && !first_write {
                        panic!(
                            "resource '{}' is read in pass '{}' (index {pi}) before any write; \
                             use import_resource() for external inputs, or write it in an earlier pass",
                            node.name, pass.name
                        );
                    }
                }
            }
        }

        // Interval partitioning: greedy first-fit over slots with a free
        // window and a matching spec.
        let mut used: Vec<ResourceId> = resources
            .iter()
            .filter(|rl| rl.first_use != usize::MAX)
            .map(|rl| rl.id)
            .collect();
        used.sort_by_key(|&id| {
            (
                resources[id.0 as usize].first_use,
                resources[id.0 as usize].last_use,
            )
        });

        let mut slots: Vec<PoolSlot> = Vec::new();
        for id in used {
            let rl = &resources[id.0 as usize];
            match slots
                .iter()
                .position(|s| s.spec == rl.spec && s.last_pass < rl.first_use)
            {
                Some(i) => {
                    slots[i].resources.push(id);
                    slots[i].last_pass = rl.last_use;
                    resources[id.0 as usize].slot = Some(i);
                }
                None => {
                    let i = slots.len();
                    slots.push(PoolSlot {
                        index: i,
                        spec: rl.spec,
                        resources: vec![id],
                        first_pass: rl.first_use,
                        last_pass: rl.last_use,
                    });
                    resources[id.0 as usize].slot = Some(i);
                }
            }
        }

        // Live resources per pass.
        let pass_alive: Vec<Vec<ResourceId>> = (0..passes.len())
            .map(|pi| {
                resources
                    .iter()
                    .filter(|rl| rl.alive_at(pi))
                    .map(|rl| rl.id)
                    .collect()
            })
            .collect();

        // Internal invariant check: a slot must not be shared within one pass.
        for (pi, alive) in pass_alive.iter().enumerate() {
            let mut seen: HashMap<usize, ResourceId> = HashMap::new();
            for &rid in alive {
                let rl = &resources[rid.0 as usize];
                let Some(slot) = rl.slot else {
                    continue;
                };
                if let Some(prev) = seen.insert(slot, rid) {
                    panic!(
                        "layout bug: pass {pi} aliases slot #{slot} for resources {prev:?} and {rid:?}"
                    );
                }
            }
        }

        GraphLayout {
            surface_size: self.surface_size,
            passes,
            resources,
            slots,
            pass_alive,
        }
    }

    /// Executes the graph: for each pass in layout order, `run` is invoked
    /// with a [`PassContext`] (live resources and their slots).
    ///
    /// # Panics
    /// Panics if the layout has not been computed yet (call `build()` first).
    pub fn execute(&self, layout: &GraphLayout, mut run: impl FnMut(PassContext<'_>)) {
        for index in 0..layout.passes.len() {
            run(PassContext { layout, index });
        }
    }
}

/// Context of the pass being executed: live resources and their pool slots.
#[derive(Debug)]
pub struct PassContext<'a> {
    layout: &'a GraphLayout,
    index: usize,
}

impl<'a> PassContext<'a> {
    /// The pass in execution order.
    pub fn pass(&self) -> &'a PassLayout {
        &self.layout.passes[self.index]
    }

    /// Index of the pass within the layout.
    pub fn pass_index(&self) -> usize {
        self.index
    }

    /// Resources alive on this pass.
    pub fn alive(&self) -> &'a [ResourceId] {
        &self.layout.pass_alive[self.index]
    }

    /// Resource metadata.
    pub fn resource(&self, id: ResourceId) -> &'a ResourceLayout {
        &self.layout.resources[id.0 as usize]
    }

    /// Pool slot for the resource, if it is alive on this pass.
    pub fn slot_of(&self, id: ResourceId) -> Option<usize> {
        let rl = self.resource(id);
        if rl.alive_at(self.index) {
            rl.slot
        } else {
            None
        }
    }
}

/// Builder for declaring a pass.
#[derive(Debug)]
pub struct PassBuilder<'a> {
    graph: &'a mut RenderGraph,
    id: PassId,
}

impl PassBuilder<'_> {
    /// Id of the pass being declared.
    pub fn id(&self) -> PassId {
        self.id
    }

    /// Declares a resource as read by the pass.
    ///
    /// # Panics
    /// Panics on an unknown resource or a read-before-write violation
    /// (detected at `build()`).
    pub fn read(self, id: ResourceId) -> Self {
        self.graph
            .resolve_resource(id, &self.graph.passes[self.id.0 as usize].name);
        self.graph.passes[self.id.0 as usize].reads.push(id);
        self
    }

    /// Declares a resource as written by the pass (no clear).
    pub fn write(self, id: ResourceId) -> Self {
        self.graph
            .resolve_resource(id, &self.graph.passes[self.id.0 as usize].name);
        self.graph.passes[self.id.0 as usize]
            .writes
            .push((id, None));
        self
    }

    /// Declares a resource as written by the pass with a clear value
    /// (typically the frame background).
    pub fn write_clear(self, id: ResourceId, clear: wgpu::Color) -> Self {
        self.graph
            .resolve_resource(id, &self.graph.passes[self.id.0 as usize].name);
        self.graph.passes[self.id.0 as usize]
            .writes
            .push((id, Some(clear)));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(format: wgpu::TextureFormat, samples: u32) -> TextureSpec {
        TextureSpec {
            format,
            samples,
            size: SizePolicy::MatchSurface,
        }
    }

    #[test]
    fn lifetime_window_basic() {
        let mut g = RenderGraph::new((1920, 1080));
        let albedo = g.create_resource("albedo", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let hdr = g.create_resource("hdr", spec(wgpu::TextureFormat::Rgba16Float, 1));
        let depth = g.create_resource("depth", spec(wgpu::TextureFormat::Depth32Float, 1));

        g.add_pass("gbuffer").write(albedo).write(depth);
        g.add_pass("lighting").read(albedo).read(depth).write(hdr);

        let layout = g.build();
        assert_eq!(layout.passes.len(), 2);
        let a = &layout.resources[albedo.0 as usize];
        assert_eq!(
            (a.first_use, a.last_use),
            (0, 1),
            "albedo: gbuffer → lighting"
        );
        let h = &layout.resources[hdr.0 as usize];
        assert_eq!(
            (h.first_use, h.last_use),
            (1, 1),
            "hdr lives only on lighting"
        );
        let d = &layout.resources[depth.0 as usize];
        assert_eq!((d.first_use, d.last_use), (0, 1));
        // Different formats → different slots.
        assert_ne!(a.slot, h.slot);
        assert_eq!(layout.slots.len(), 3);
    }

    #[test]
    fn transient_slot_reuse_same_spec() {
        // a lives [0,1], b lives [2,3], same spec → one slot (aliasing).
        let mut g = RenderGraph::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));

        g.add_pass("p0").write(a);
        g.add_pass("p1").read(a);
        g.add_pass("p2").write(b);
        g.add_pass("p3").read(b);

        let layout = g.build();
        assert_eq!(
            layout.slots.len(),
            1,
            "non-overlapping windows share a slot"
        );
        assert_eq!(layout.slots[0].resources, vec![a, b]);
        assert_eq!(layout.resources[a.0 as usize].slot, Some(0));
        assert_eq!(layout.resources[b.0 as usize].slot, Some(0));
    }

    #[test]
    fn overlapping_resources_need_distinct_slots() {
        // a [0,1], b [1,2] — overlap on pass 1 → two slots.
        let mut g = RenderGraph::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));

        g.add_pass("p0").write(a);
        g.add_pass("p1").read(a).write(b);
        g.add_pass("p2").read(b);

        let layout = g.build();
        assert_eq!(layout.slots.len(), 2);
        assert_ne!(
            layout.resources[a.0 as usize].slot,
            layout.resources[b.0 as usize].slot
        );
    }

    #[test]
    #[should_panic(expected = "before any write")]
    fn read_before_write_panics() {
        let mut g = RenderGraph::new((320, 240));
        let x = g.create_resource("x", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        g.add_pass("p0").read(x);
        g.add_pass("p1").write(x);
        g.build();
    }

    #[test]
    #[should_panic(expected = "unknown resource")]
    fn unknown_resource_panics() {
        let mut g = RenderGraph::new((320, 240));
        g.add_pass("p0").read(ResourceId(99));
    }

    #[test]
    fn imported_resource_may_be_read_first() {
        let mut g = RenderGraph::new((320, 240));
        let shadow = g.import_resource("shadow", spec(wgpu::TextureFormat::R32Float, 1));
        g.add_pass("p0").read(shadow);
        g.add_pass("p1").read(shadow);
        let layout = g.build(); // does not panic
        let rl = &layout.resources[shadow.0 as usize];
        assert_eq!((rl.first_use, rl.last_use), (0, 1));
        assert_eq!(rl.slot, Some(0));
    }

    #[test]
    fn disabled_pass_culls_its_resources() {
        let mut g = RenderGraph::new((320, 240));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let p1 = g.add_pass("p1").write(a).id();
        g.add_pass("p2").write(b);

        g.set_pass_enabled(p1, false);
        let layout = g.build();
        assert_eq!(layout.passes.len(), 1);
        assert_eq!(layout.passes[0].name, "p2");
        let ra = &layout.resources[a.0 as usize];
        assert_eq!(
            ra.first_use,
            usize::MAX,
            "a is not used by any enabled pass"
        );
        assert_eq!(ra.slot, None);
        assert_eq!(layout.slots.len(), 1, "only b gets a slot");
    }

    #[test]
    fn execute_delivers_live_resources_and_slots() {
        let mut g = RenderGraph::new((640, 480));
        let a = g.create_resource("a", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let b = g.create_resource("b", spec(wgpu::TextureFormat::Rgba16Float, 1));
        g.add_pass("gbuffer").write(a);
        g.add_pass("lighting").read(a).write(b);
        g.add_pass("composite").read(b);

        let layout = g.build();
        let mut visits: Vec<(usize, Vec<ResourceId>, Option<usize>)> = Vec::new();
        g.execute(&layout, |ctx| {
            let slot_a = ctx.slot_of(a);
            visits.push((ctx.pass_index(), ctx.alive().to_vec(), slot_a));
        });

        assert_eq!(visits[0], (0, vec![a], Some(0)), "gbuffer: a alive");
        assert_eq!(
            visits[1],
            (1, vec![a, b], Some(0)),
            "lighting: a and b alive"
        );
        assert_eq!(visits[2], (2, vec![b], None), "composite: a is dead");
        assert_eq!(
            ctx_pass_names(&layout),
            vec!["gbuffer", "lighting", "composite"]
        );
    }

    fn ctx_pass_names(layout: &GraphLayout) -> Vec<String> {
        layout.passes.iter().map(|p| p.name.clone()).collect()
    }

    #[test]
    fn debug_dump_lists_structure() {
        let mut g = RenderGraph::new((1280, 720));
        let albedo = g.create_resource("albedo", spec(wgpu::TextureFormat::Rgba8Unorm, 1));
        let hdr = g.create_resource("hdr", spec(wgpu::TextureFormat::Rgba16Float, 1));
        g.add_pass("gbuffer").write(albedo);
        g.add_pass("lighting").read(albedo).write(hdr);

        let dump = g.build().debug_dump();
        assert!(dump.contains("2 passes"), "dump: {dump}");
        assert!(dump.contains("'gbuffer'"), "dump: {dump}");
        assert!(dump.contains("'hdr'"), "dump: {dump}");
        assert!(dump.contains("pool slots"), "dump: {dump}");
        assert!(dump.contains("albedo"), "dump: {dump}");
    }

    #[test]
    fn clear_value_is_carried_to_layout() {
        let mut g = RenderGraph::new((640, 480));
        let hdr = g.create_resource("hdr", spec(wgpu::TextureFormat::Rgba16Float, 1));
        let pid = {
            let builder = g.add_pass("lighting");
            let pid = builder.id();
            builder.write_clear(hdr, wgpu::Color::BLACK);
            pid
        };
        let layout = g.build();
        assert_eq!(
            layout.passes[pid.0 as usize].writes,
            vec![(hdr, Some(wgpu::Color::BLACK))]
        );
    }
}
