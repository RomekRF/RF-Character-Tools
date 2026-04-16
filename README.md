# RF Character Tools

Blender add-on for importing and exporting Red Faction character meshes (V3C/V3M) and animations (RFA).

## Features

- **V3C/V3M Import:** Character meshes with armature, bone weights, materials, collision spheres, prop points, and LODs
- **V3C Export:** Full mesh export with skeleton, weights, collision spheres, and prop points
- **RFA Import/Export:** Batch import stock animations, preview in Blender, re-export with byte-exact round-trip
- **Custom Mesh Workflow:** Bind external meshes to the RF armature, transfer weights from the original, validate weight coverage
- **Animation Management:** Browse and switch animations, delete individually or in bulk
- **Blender 4.0–5.0+** compatible (layered action API supported)

## Installation

1. Download the latest release zip
2. In Blender: Edit → Preferences → Add-ons → Install
3. Select the zip file
4. Enable "RF Character Tools (V3C/V3M/RFA)"

## Workflow

1. Import a V3C character mesh
2. Find Textures to link materials
3. Batch import animations (or load from the Required Animations list)
4. Import a custom mesh, bind to armature, transfer weights
5. Check Weights to verify full coverage
6. Export V3C (and re-export animations if needed)

## Custom Mesh Tips

- Position your mesh to match the original character's proportions at the joints
- After Transfer Weights, use Check Weights to find and fix any unweighted vertices
- Weight paint problem areas (shoulders, hips, knees) for best results
- The original RF mesh stays available for reference via the Mesh visibility toggle

## Panel Location

3D Viewport → Sidebar (N) → RF Character

## Compatibility

- Blender 4.0 through 5.0+
- Red Faction PC (V3C/V3M format version 0x40000)
- Works with [Alpine Faction](https://github.com/GooberRF/alpinefaction) (DDS textures, no hard caps)

## Credits

- [rafalh/rf-tools](https://github.com/rafalh/rf-tools) (vmesh) — V3C format reference and coordinate conversion verification
- [GooberRF/redux](https://github.com/GooberRF/redux) — V3C/RFA export reference
- [GooberRF/alpinefaction](https://github.com/GooberRF/alpinefaction) — Enhanced RF client

## License

MIT License
