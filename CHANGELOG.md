# Changelog

## v1.10.4 — Review fixes

Checked against all 95 stock character meshes (import, export, compare) and 1,009 stock animations.

V3C
- The reader took a batch's triangle count from its padded byte size, adding a zero triangle for every odd count (and, with plane data, misreading everything after it). It now uses the stored count.
- Re-exporting an imported RF mesh keeps every material in its place, used or not, so skins (which count materials) still line up. 14 stock meshes, all first-person weapons, lost unused slots before.
- Import keeps reversed duplicate triangles (double-sided spots) by giving them their own vertices; Blender used to drop them. Exact duplicates are still left out.
- Every stock mesh now round-trips with the same texture slots, flags, hitboxes and prop points; normals within 2°.

RFA
- Re-encoded (edited) animations wrote speeds into the position keys' in/out fields. Stock files hold the two inner Bezier points of the curve to the next key there (measured over all stock files); the exporter now writes points for a straight path between keys, as Blender shows it.
- Edited animations keep the original header (ramp in/out, totals) and the face morph data of cutscene animations.

Importing your mesh
- DAE import rewritten: up axis and units applied (Y-up files no longer lie on their back), node transforms and skin bind poses, first UV set read with its real layout, `<polygons>` faces, n-gons kept, smooth faces with the file's normals, and textured materials (diffuse image or colour).
- GLM import keeps the file's normals on smooth faces (it came in flat-shaded, which now exports as hard edges).
- After Import Mesh, just the new mesh parts are selected, ready for the next step.
- New Fit to RF Mesh: scales the selected parts together to the RF character's height and stands them on its spot.

Other
- Batch anims, picking an animation, Rest Pose and Delete Animation find the RF armature themselves instead of failing when it isn't selected.
- Auto-Load's error no longer points to a removed Preferences page.

## v1.10.3

- Fixed the Textures panel failing (Blender listing every icon name) when the picked texture had a warning: Blender has no 'WARNING' icon.
- Combine Textures rebuilt:
  - Bigger textures get bigger tiles (powers of two, halved only as needed to fit) instead of equal squares.
  - Each tile has a border copied from its opposite edge, so filtering and tiling don't pick up the neighbouring texture.
  - Faces whose UVs lie outside 0..1 (tiling UVs) are moved back by whole repeats before going into their tile; faces that stretch across a repeat are counted in a warning.
  - Float images (EXR, 16-bit) keep their colours; the atlas is saved as uncompressed TGA (or PNG), 32-bit only when a combined texture uses alpha.
  - Default atlas name is `<character>_atlas`, so it can't replace a stock texture of the character's name.
- Combine Textures and Convert Textures for RF remember the last output folder across Blender sessions and files.

## v1.10.2 — Textures panel

- New Convert Textures for RF: writes every texture of your mesh as an RF-ready TGA into one folder (power-of-two size up to a chosen limit, 24/32-bit, safe names of 31 characters or fewer) and switches your materials to those files. Materials with only a colour get a texture of that colour.
- The texture list is clickable. The picked texture can be renamed, given a new image (Replace Image) or have its faces selected (Select Faces).
- Combine Textures moved into the Textures panel; with nothing selected it works on the parts bound to the RF armature.
- New "In multiplayer" box: for each stock multiplayer character that uses the mesh you are replacing, shows whether your texture names are used or which files its skin loads, with Use These Names to rename your textures to that skin's files.
- Removed Copy Skin Line and the RF-mesh slot list.

## v1.10.1

- Generate Texture Atlas lists every texture with a tick box. Only ticked textures go into the atlas; faces using unticked textures keep their material and UVs, and materials without an image are kept too. The atlas material gets alpha when a ticked texture used it, and an existing atlas material is updated in place instead of being deleted.
- The atlas leaves the imported RF meshes alone, and a mesh shared by several selected objects is only remapped once.
- Copy Skin Line copies at once (no dialog). The skin name and the full line are shown in the Textures panel, a popup shows what was copied, and a warning appears if the clipboard could not be written.

## v1.10.0 — Custom Character Workflow

Export
- One texture slot per texture instead of one per batch. A part over 5,400 vertices no longer takes extra slots, and export stops with a message when a character needs more than RF's 7 slots.
- Writes one material per slot, so skin order (entity.tbl `$Skin`) is the slot order shown in the Textures panel.
- Exports every visible part bound to the RF armature as one mesh, not only the active object. "Selected parts only" exports just the selection.
- Writes parts where they sit in the scene, including location and mirrored scale. The scene is no longer changed (transforms used to be applied to the object).
- No longer exports the RF mesh when the armature is selected and a mesh of yours is in view but not bound; it says what to do instead.
- Modifiers other than Armature (Mirror, Subdivision, ...) are written as they look in the viewport ("Apply modifiers", on by default).
- Hard edges, flat faces and custom normals are kept. "Smooth all normals" restores the old smooth-everything output.
- Alpha flag only on materials that use their texture's alpha. Blender 4.2+ marked every material as alpha, because new materials report Dithered/Hashed.
- Material flags read from an imported V3C (such as 0x10 on stock visors) are kept on re-export.
- Saves each texture slot as an uncompressed TGA next to the V3C: 24-bit, or 32-bit for alpha, resized to a power of two ("Save textures", on by default).
- Texture names drop Blender's `.001` suffixes (`foo.png.001` → `foo.tga`); names over 31 characters or with non-ASCII characters stop the export.
- Chunks are also split at 8,000 triangles (uint16 size field).

Weights
- Copy Weights from RF Mesh works on several selected parts at once, in the rest pose, and also when the RF mesh is hidden.
- Binding keeps parts where they are, removes Armature modifiers and vertex groups left from other rigs ("Remove old rig"), and leaves one Armature modifier.
- New Limit to 4 Bones: keeps the 4 strongest RF bones per vertex and normalizes, so the viewport matches the game. Copy Weights and Bind with Auto Weights do this by default.
- Check Weights checks every part and selects the problem vertices on all of them.

Animations
- An imported RFA that was edited in Blender now exports the edits. The original file bytes are written only while the keys are unchanged. Animations imported before 1.10 keep the old behaviour; re-import them to edit them.

Panel
- The sidebar follows the steps: 1 Import, 2 Your Mesh (with show/hide), 3 Weights, 4 Textures, 5 Export. Animations and Advanced (Bind with Auto Weights, Combine Textures, Sync Texture Names) are separate, closed panels.
- New Textures panel: the slots in order with editable names, size, alpha and problems; the RF mesh's own slots for reference; Copy Skin Line.
- Validate checks the same parts, slots and limits as export.

Import
- RF meshes import with smooth faces, so their custom normals are stored more precisely and re-export with the stock vertex count.

## v1.9.25

- Texture names strip repeated image extensions (`foo.png.tga` → `foo.tga`).
- Export option "Rename materials to texture names".
- Validate warns about double extensions.
- Batch Export V3Cs removed.

## v1.9.22 — Repo cleanup (Unreleased)

- Addon folder renamed from `rf_character_import` to `rf_character_tools` to match the repo name and reflect that the addon does both import and export.
- Removed obsolete v1.8.2 folder from main.

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
