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
- [Romek](https://github.com/RomekRF) - Creating blender add-on
- [rafalh/rf-tools](https://github.com/rafalh/rf-tools) (vmesh) — V3C format reference and coordinate conversion verification
- [GooberRF/redux](https://github.com/GooberRF/redux) — V3C/RFA export reference
- [GooberRF/alpinefaction](https://github.com/GooberRF/alpinefaction) — Enhanced RF client

## License
MIT License

<img width="2208" height="1173" alt="image" src="https://github.com/user-attachments/assets/628d7a30-0c15-4de3-bb61-bf88613f1a2f" />
<img width="2473" height="1269" alt="image" src="https://github.com/user-attachments/assets/b908d393-de25-45ff-bf62-208918a252f1" />
<img width="2560" height="1440" alt="20260506_225632_rmkglass_house" src="https://github.com/user-attachments/assets/ede0989a-32d3-4953-950e-a464576789a5" />
<img width="2560" height="1440" alt="20260419_213625_DM-Jerusalemv2" src="https://github.com/user-attachments/assets/47c3e8ee-36d7-4835-8c62-5971132561da" />
<img width="2560" height="1440" alt="20260428_194135_DM-ImperialB03" src="https://github.com/user-attachments/assets/6dc1cbad-dd63-4d1b-8ab4-ef72dcb93f96" />
<img width="2560" height="1440" alt="20260403_155310_DM-RFU2-Temple-Of-Doom" src="https://github.com/user-attachments/assets/bb441ebe-5da5-4de3-9c44-bcef62189f20" />
<img width="2560" height="1440" alt="20260402_210341_DM_mf_Death_Star_Hanger" src="https://github.com/user-attachments/assets/8193ca85-18d8-4048-9de9-c10d77687068" />

