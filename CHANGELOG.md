# Changelog

## v1.9.21 — Rest Pose After Import

- Force rest pose after every animation import path — clears action, mutes all NLA tracks, resets pose bone transforms, triggers frame update

## v1.9.20 — Rest Pose Polish

- Clear active action after batch anim import

## v1.9.19 — Master Folder Without AddonPreferences

- Removed `AddonPreferences` class entirely (crash source on some installs)
- Master folder now configurable directly from main panel via **"Set Folder..."** button
- Persisted to `<Blender config>/rf_character_master.txt`
- Added **"Clear Master Folder"** action

## v1.9.14 — Auto-Load Animations

- Master Animation Folder preference — set once, auto-loads animations on every V3C import
- New `_auto_load_anims_from_folder()` helper shared between V3C import and manual load
- Added **"Auto-Load from Master Folder"** button in Required Animations panel

## v1.9.13 — Texture Atlas Rewrite

- Replaced unreliable pixel copy with `foreach_get`/`foreach_set`
- Auto-reload images if not already loaded before atlas composite
- Handle RGB/grayscale/RGBA source images
- Per-image error reporting for atlas failures

## v1.9.12 — Texture Name Sync + Panel Reorg

- Moved search/grouping from Required to **Loaded Animations** panel
- Required Animations panel simplified to clean checklist with All/Missing/Loaded filter
- New **"Sync Texture Names"** — forces `.tga` extension on image names and aligns material names
- V3C exporter prefers Base Color-connected image node
- Validator checks texture names

## v1.9.11 — Texture Atlas

- **"Combine Textures → Atlas"** operator for custom mesh materials
- Grid-based packing (128/256/512/1024/2048 atlas sizes)
- TGA and PNG output formats
- Automatic UV remapping to atlas regions
- Optional material replacement

## v1.9.10 — UI Overhaul

- Search box, category grouping, show-loaded/missing toggles
- Recent Files list (last 5 V3Cs with one-click re-import)
- **"Validate"** operator for pre-export checks (bones, weights, materials, chunks)
- **"Batch Export V3C"** for multiple armatures at once

## v1.9.9 — Folder Memory

- Remember animation folder per-armature and scene-wide
- Auto-reuse on subsequent "Load Missing" clicks

## v1.9.8 — Multiplayer Animation Database

- Parsed `pc_multi.tbl` + `entity.tbl` for 20 MP characters
- Merged SP+MP databases with union lookup for shared characters
- Covers miner1, multi_civilian, multi_female, multi_merc skeletons

## v1.9.7 — UI Polish

- Removed confusing checkmark icon from animation status line

## v1.9.6 — GLM Crash Fix

- Removed `normals_split_custom_set` (Blender 5.0 crash source)
- Each surface wrapped in try/except so one bad surface doesn't kill the import

## v1.9.5 — GLM Crash Fix

- Removed selection operations from surface creation loop (StructRNA crash)

## v1.9.3 — GLM Surfaces as Separate Meshes

- Each surface imports as its own Blender object

## v1.9.2 — GLM Texture Auto-Detection

- Three-tier lookup: `.skin` files → embedded shader names → directory texture matching

## v1.9.0 — GLM Importer

- Built-in Ghoul 2 `.glm` parser for Jedi Academy/Jedi Outcast meshes
- Proper LOD surface offset handling
- Skin file texture mapping

## v1.8.x — Core Exporter Maturity

- Weight normalization matching Redux
- Double-sided face flag (0x20) from `use_backface_culling`
- Auto-split chunks at 5,400 verts (uint16 alloc field limit)
- DAE/Collada parser for Blender 5.0+
- Crash hardening in V3C importer
