# Gosub Integration Plan

## Goal
Add Gosub as alternative UI rendering backend alongside existing html5ever+lightningcss+taffy+vello stack.

## Architecture

### Current Stack (keep as-is)
```
HTML → html5ever → DOM → lightningcss → Styles → taffy → Layout → vello → Render
JS: boa_engine (limited React/Vue support)
```

### Gosub Stack (new)
```
HTML → gosub_html5 → DOM → gosub_css3 → Styles → gosub_taffy → Layout → gosub_renderer_vello → Render
JS: gosub_v8 (full React/Vue support)
```

### Feature Flag
- `ui-gosub` feature enables Gosub backend
- Default: old stack
- Both can coexist, runtime selection via config

## Implementation Steps

### Phase 1: Infrastructure (Week 1-2)
1. Create `crates/ui-gosub` crate
2. Add Gosub dependencies to Cargo.toml
3. Setup feature flags
4. Basic integration test (render HTML → PNG)

### Phase 2: Text Rendering (Week 3-4)
1. Replace fake-bold with parley shaping
2. Implement FontSystem trait wrapper
3. Migrate text.rs to use parley
4. Verify kerning/hinting quality

### Phase 3: CSS/Layout (Week 5-6)
1. Integrate gosub_css3 (if more complete than lightningcss)
2. Connect gosub_taffy layout
3. Handle responsive/overflow cases
4. Verify tab overlap fixes

### Phase 4: Full Integration (Week 7-10)
1. Connect Gosub render pipeline to Ornis ECS
2. Implement JS bridge (gosub_v8 or keep boa)
3. Add UI toggle (old vs new backend)
4. Performance benchmarking

## Key Differences from Current Code

### Text Rendering
- **Current**: vello::peniko glyph rasterization + fake-bold stroke
- **Gosub**: parley shaping → proper kerning/hinting/weights
- **Benefit**: No fake-bold hack, proper font weights

### CSS Support
- **Current**: lightningcss (good but not complete)
- **Gosub**: gosub_css3 (spec-compliant)
- **Benefit**: Better overflow/scroll/media query support

### JS Support
- **Current**: boa_engine (basic DOM, no React/Vue)
- **Gosub**: gosub_v8 (full V8, complete JS)
- **Benefit**: Run React/Vue in-game

### Architecture
- **Current**: Direct vello Scene building
- **Gosub**: RenderBackend trait + paint commands
- **Benefit**: More abstract, easier to extend

## Risks

1. **Gosub 0.1.x API instability** - May break on updates
2. **Dependency bloat** - V8 adds ~30MB
3. **Integration complexity** - Two UI stacks to maintain
4. **Performance** - Need to benchmark vs current approach

## Mitigation

1. Pin Gosub versions, update manually
2. Make V8 optional (feature flag)
3. Keep old stack as fallback
4. Benchmark early, optimize if needed

## Success Criteria

- [ ] Can render HTML/CSS/JS with Gosub backend
- [ ] Text rendering matches Chromium quality
- [ ] No tab overlap on resize
- [ ] React/Vue apps run in-game
- [ ] Performance within 10% of current stack
- [ ] Feature flag switches between stacks at runtime

## Timeline

- **Week 1-2**: Infrastructure + basic render test
- **Week 3-4**: Text rendering migration (parley)
- **Week 5-6**: CSS/layout integration
- **Week 7-10**: Full integration + JS + benchmarks
- **Total**: ~10 weeks to production-ready

## Maintenance

- Both stacks maintained in parallel
- Old stack: bug fixes only
- New stack: active development
- Migration path for existing UI code
