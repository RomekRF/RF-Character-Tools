# RF Character Tools

Blender add-on for importing and exporting Red Faction character meshes (V3C/V3M) and animations (RFA).

## Features

- **V3C/V3M Import:** Character meshes with armature, bone weights, materials, collision spheres, prop points, and LODs
- **V3C Export:** Full mesh export with skeleton, weights, collision spheres, and prop points
- **RFA Import/Export:** Batch import stock animations, preview in Blender, re-export with byte-exact round-trip
- **Custom Mesh Workflow:** Copy weights from the original onto meshes made of any number of parts, check weight coverage, export with up to 7 textures
- **Animation Management:** Browse and switch animations, delete individually or in bulk
- **Blender 4.0–5.0+** compatible (layered action API supported)

## Installation

1. Download the latest release zip
2. In Blender: Edit → Preferences → Add-ons → Install
3. Select the zip file
4. Enable "RF Character Tools (V3C/V3M/RFA)"

## Workflow

The sidebar follows these steps:

1. **Import the RF character.** Import V3C, then Find Textures to show the stock textures.
2. **Your mesh.** Import Mesh, then move, rotate and scale it over the RF mesh. Transforms don't need applying; export writes the mesh where it sits in the scene.
3. **Weights.** Select every part of your mesh and press Copy Weights from RF Mesh. It binds the parts to the RF armature, copies the RF character's weights from the nearest point of its surface, removes leftover rigs (for example from an FBX) and keeps the 4 strongest bones per vertex, as the game does. Check Weights selects any vertex left without weights; fix spots in Weight Paint, then press Limit to 4 Bones.
4. **Textures.** Press Convert Textures for RF: every texture of your mesh is written as an RF-ready TGA into one folder and your materials switch to those files, so Blender shows what the game gets. Click a texture in the list to rename it, replace its image or select the faces that use it. RF allows 7 textures per character; Combine Textures merges some into one.
5. **Export.** Export V3C writes every visible part bound to the armature as one mesh, and saves each texture as an uncompressed `.tga` next to the V3C.

Animations and the master animation folder are in the Animations panel. Bind with Auto Weights is under Advanced.

## Textures

- Convert Textures for RF writes uncompressed TGA: power-of-two size up to a limit you choose (1024 by default), 24-bit, or 32-bit when the material uses the image's alpha. Names keep to ASCII letters, digits, `_` and `-`, 31 characters at most with `.tga`. A material with only a colour gets a 64×64 texture of that colour. Your original files are not changed.
- Each texture slot is one texture. Parts that use the same image share a slot.
- Combine Textures lists every texture with a tick box; only ticked textures are merged into the atlas, the rest keep their own materials. Bigger textures get bigger tiles, each tile has a wrapped border, and faces with tiling UVs (outside 0..1) are moved into their tile.
- Convert and Combine remember the last folder you used.

## Multiplayer characters and skins

Some multiplayer characters wear a skin (`$SkinName` in pc_multi.tbl, `$Skin` in entity.tbl). A skin replaces the mesh's textures by number, so the game loads the skin's files instead of the names in your V3C. The Textures panel shows, for each stock multiplayer character that uses the mesh you are replacing, which files the game loads:

- No skin (for example Executive Parker on parker_suit.v3c): your texture names are used as they are.
- A skin (for example Envirosuit Parker on miner.v3c): Use These Names renames your textures, in order, to the skin's files, so the files you ship replace those stock textures. They replace them for every character that uses the same files, and only as many textures as the skin has fit.

## Custom Mesh Tips

- Line your mesh up with the RF character at the joints; Copy Weights takes each vertex's weights from the nearest point of the RF mesh
- Weight paint problem areas (shoulders, hips, knees) for best results
- The RF mesh, hitboxes, prop points and LODs can be shown and hidden in the Your Mesh panel
- Hard edges and flat faces are exported as they look in Blender; Smooth all normals in the export options smooths every vertex instead

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

