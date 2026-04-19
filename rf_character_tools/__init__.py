bl_info = {
    "name": "RF Character Tools (V3C/V3M/RFA)",
    "author": "RF VFX Tools",
    "version": (1, 9, 21),
    "blender": (4, 0, 0),
    "location": "File > Import/Export, 3D View > Sidebar > RF Character",
    "description": "Import and export Red Faction character meshes, animations, and rigs",
    "category": "Import-Export",
}

import bpy
import struct
import math
import json
import os
import base64
import traceback
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, CollectionProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from mathutils import Vector, Quaternion, Matrix


# ═══════════════════════════════════════════════════════════════════════════════
#  COORDINATE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════
#
# RF: left-handed, Y-up  →  Blender: right-handed, Z-up
# Redux pipeline: RF → RH Y-up (negate X) → glTF → Blender (RH Z-up)
#
# Position:  RF(x,y,z)       → BL(-x, -z, y)
# Quaternion: RF(qx,qy,qz,qw) → BL Quaternion(qw, -qx, -qz, qy)
# UV: v_bl = 1 - v_rf
# Winding: swap indices 1↔2 (CW→CCW)

def rf_to_bl_pos(rx, ry, rz):
    """Convert RF position to Blender position."""
    return Vector((-rx, -rz, ry))

def rf_to_bl_quat(qx, qy, qz, qw):
    """Convert RF quaternion (x,y,z,w) to Blender Quaternion (w,x,y,z)."""
    return Quaternion((qw, -qx, -qz, qy))

def rh_to_bl_pos(x, y, z):
    """Convert RH Y-up position to Blender Z-up."""
    return Vector((x, -z, y))

def rh_to_bl_quat_xyzw(qx, qy, qz, qw):
    """Convert RH Y-up quaternion (x,y,z,w) to Blender Quaternion (w,x,y,z)."""
    return Quaternion((qw, qx, -qz, qy))


# ── Reverse conversions (Blender → RF) for export ──

def bl_to_rf_pos(v):
    """Convert Blender position back to RF coordinates.
    Reverse of rf_to_bl_pos: BL(X,Y,Z) → RF(-X, Z, -Y)."""
    return (-v[0], v[2], -v[1])

def bl_to_rf_quat(q):
    """Convert Blender Quaternion back to RF quaternion (x,y,z,w).
    Reverse of the import chain: BL→RH→RF.
    BL Quat(w,x,y,z) → RH(x, z, -y, w) → RF(-x, z, -(-y), w) = (-x, z, y, w).
    But import RF→RH negates x, so reverse RH→RF also negates x:
    RF = (-rh_x, rh_y, rh_z, rh_w) where RH = (bl.x, bl.z, -bl.y, bl.w)."""
    rh_x, rh_y, rh_z, rh_w = q.x, q.z, -q.y, q.w
    return (-rh_x, rh_y, rh_z, rh_w)  # RH→RF: negate x


# ═══════════════════════════════════════════════════════════════════════════════
#  BINARY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _ascii(data, off, length):
    raw = data[off:off+length]
    end = raw.find(b'\x00')
    return raw[:end if end >= 0 else length].decode('ascii', errors='replace')

def _zstr(data, off):
    end = data.index(b'\x00', off)
    return data[off:end].decode('ascii', errors='replace'), end + 1

def _a16(o):
    return o + ((0x10 - (o % 0x10)) % 0x10)


# ═══════════════════════════════════════════════════════════════════════════════
#  V3C/V3M PARSER
# ═══════════════════════════════════════════════════════════════════════════════

_V3M_SIG = 0x52463344; _V3C_SIG = 0x5246434D; _V3D_VER = 0x00040000
_SEC_SUBM = 0x5355424D; _SEC_CSPH = 0x43535048; _SEC_BONE = 0x424F4E45
LOD_CHAR = 0x02; LOD_ORIGMAP = 0x01; LOD_PLANES = 0x20


def parse_v3c(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    off = 0
    sig = struct.unpack_from("<I", data, off)[0]; off += 4
    ver = struct.unpack_from("<I", data, off)[0]; off += 4
    if sig not in (_V3M_SIG, _V3C_SIG):
        raise ValueError(f"Not V3M/V3C (0x{sig:08X})")
    is_v3c = (sig == _V3C_SIG)
    off += 4*7
    num_cspheres = struct.unpack_from("<i", data, off)[0]; off += 4

    result = {'is_character': is_v3c, 'submeshes': [], 'bones': [], 'cspheres': []}

    while off < len(data) - 8:
        sec_type = struct.unpack_from("<I", data, off)[0]
        sec_size = struct.unpack_from("<I", data, off+4)[0]
        off += 8
        if sec_type == 0: break
        elif sec_type == _SEC_SUBM:
            sm, off = _parse_submesh(data, off)
            result['submeshes'].append(sm)
        elif sec_type == _SEC_BONE:
            result['bones'] = _parse_bones(data, off, sec_size)
            off += sec_size
        elif sec_type == _SEC_CSPH:
            # Each CSPH section contains exactly ONE collision sphere
            cs = _parse_one_csphere(data, off)
            result['cspheres'].append(cs)
            off += sec_size
        else:
            off += sec_size
    return result


def _parse_submesh(data, off):
    sm = {}
    sm['name'] = _ascii(data, off, 24); off += 24
    sm['parent_name'] = _ascii(data, off, 24); off += 24
    off += 4  # version
    num_lods = struct.unpack_from("<i", data, off)[0]; off += 4

    lod_distances = []
    for i in range(num_lods):
        d = struct.unpack_from("<f", data, off)[0]; off += 4
        lod_distances.append(d)

    off += 12 + 4 + 12 + 12  # bounding: offset + radius + bboxmin + bboxmax

    sm['lods'] = []
    for li in range(num_lods):
        lod, off = _parse_lod(data, off, lod_distances[li])
        sm['lods'].append(lod)

    num_materials = struct.unpack_from("<i", data, off)[0]; off += 4
    sm['materials'] = []
    sm['material_flags'] = []
    for mi in range(num_materials):
        diffuse = _ascii(data, off, 32); off += 32
        off += 4*4  # emissive, specular, glossiness, reflection
        off += 32   # reflection map
        mat_flags = struct.unpack_from("<I", data, off)[0]; off += 4
        sm['materials'].append(diffuse)
        sm['material_flags'].append(mat_flags)

    num_unk = struct.unpack_from("<i", data, off)[0]; off += 4
    off += num_unk * 28

    return sm, off


def _parse_lod(data, off, distance):
    lod = {'distance': distance}
    flags = struct.unpack_from("<I", data, off)[0]; off += 4
    num_verts = struct.unpack_from("<i", data, off)[0]; off += 4
    num_chunks = struct.unpack_from("<H", data, off)[0]; off += 2
    data_block_size = struct.unpack_from("<i", data, off)[0]; off += 4

    lod['flags'] = flags
    lod['num_verts'] = num_verts

    data_block = data[off:off+data_block_size]
    off += data_block_size
    off += 4  # unk1

    chunk_infos = []
    for ci in range(num_chunks):
        vals = struct.unpack_from("<7HI", data, off); off += 18
        chunk_infos.append(vals)

    num_proppoints = struct.unpack_from("<i", data, off)[0]; off += 4
    num_textures = struct.unpack_from("<i", data, off)[0]; off += 4

    lod['textures'] = []
    for t in range(num_textures):
        tex_id = data[off]; off += 1
        tex_name, off = _zstr(data, off)
        lod['textures'].append((tex_id, tex_name))

    has_planes = bool(flags & LOD_PLANES)
    has_origmap = bool(flags & LOD_ORIGMAP)

    lod['chunks'], lod['prop_points'] = _unpack_data_block(
        data_block, chunk_infos, num_proppoints, num_verts, has_planes, has_origmap)

    return lod, off


def _unpack_data_block(db, chunk_infos, num_proppoints, num_verts, has_planes, has_origmap):
    num_chunks = len(chunk_infos)
    db_off = 0

    chunk_tex_indices = []
    for ci in range(num_chunks):
        db_off += 0x20
        tex_idx = struct.unpack_from("<i", db, db_off)[0]; db_off += 4
        db_off += 0x14
        chunk_tex_indices.append(tex_idx)
    db_off = _a16(db_off)

    chunks = []
    for ci in range(num_chunks):
        info = chunk_infos[ci]
        ci_verts, ci_faces, ci_vecs, ci_facesalloc, ci_samepos, ci_wi, ci_uvs, ci_rflags = info

        # ci_verts is the authoritative vertex count.
        # Byte size fields (ci_vecs, ci_wi, ci_uvs) include 16-byte alignment padding,
        # so dividing by element size can overcount. Use ci_verts for reading,
        # byte sizes only for advancing the data block offset.
        num_verts_chunk = ci_verts
        num_tris = ci_facesalloc // 8
        num_samepos = ci_samepos // 2

        positions = []
        for i in range(num_verts_chunk):
            x, y, z = struct.unpack_from("<3f", db, db_off); db_off += 12
            positions.append((x, y, z))
        db_off = _a16(db_off)

        normals = []
        for i in range(num_verts_chunk):
            nx, ny, nz = struct.unpack_from("<3f", db, db_off); db_off += 12
            normals.append((nx, ny, nz))
        db_off = _a16(db_off)

        uvs = []
        for i in range(num_verts_chunk):
            u, v = struct.unpack_from("<2f", db, db_off); db_off += 8
            uvs.append((u, v))
        db_off = _a16(db_off)

        triangles = []
        for i in range(num_tris):
            i0, i1, i2, fl = struct.unpack_from("<4H", db, db_off); db_off += 8
            triangles.append((i0, i1, i2))
        db_off = _a16(db_off)

        if has_planes:
            db_off += num_tris * 16
            db_off = _a16(db_off)

        db_off += num_samepos * 2
        db_off = _a16(db_off)

        bone_links = []
        if ci_wi > 0:
            for i in range(num_verts_chunk):
                weights = struct.unpack_from("<4B", db, db_off); db_off += 4
                bones = struct.unpack_from("<4B", db, db_off); db_off += 4
                bone_links.append((weights, bones))
            db_off = _a16(db_off)

        # Skip origmap data if present (per-chunk, uses total LOD vertex count)
        if has_origmap:
            db_off += num_verts * 2
            db_off = _a16(db_off)

        chunks.append({
            'positions': positions, 'normals': normals, 'uvs': uvs,
            'triangles': triangles, 'bone_links': bone_links,
            'texture_index': chunk_tex_indices[ci]
        })

    prop_points = []
    for p in range(num_proppoints):
        pp_name = _ascii(db, db_off, 0x44); db_off += 0x44
        qx, qy, qz, qw = struct.unpack_from("<4f", db, db_off); db_off += 16
        px, py, pz = struct.unpack_from("<3f", db, db_off); db_off += 12
        parent_bone = struct.unpack_from("<i", db, db_off)[0]; db_off += 4
        prop_points.append({
            'name': pp_name, 'pos': (px,py,pz),
            'quat': (qx,qy,qz,qw), 'parent_bone': parent_bone
        })

    return chunks, prop_points


def _parse_bones(data, off, size):
    bones = []
    num_bones = struct.unpack_from("<I", data, off)[0]; off += 4
    for bi in range(num_bones):
        name = _ascii(data, off, 24); off += 24
        qx, qy, qz, qw = struct.unpack_from("<4f", data, off); off += 16
        tx, ty, tz = struct.unpack_from("<3f", data, off); off += 12
        parent_index = struct.unpack_from("<i", data, off)[0]; off += 4
        bones.append({
            'name': name,
            'inv_bind_quat': (qx, qy, qz, qw),
            'inv_bind_pos': (tx, ty, tz),
            'parent_index': parent_index
        })
    return bones


def _parse_one_csphere(data, off):
    """Parse a single collision sphere (44 bytes): name(24) + parent(4) + pos(12) + radius(4)."""
    name = _ascii(data, off, 24); off += 24
    parent_bone = struct.unpack_from("<i", data, off)[0]; off += 4
    px, py, pz = struct.unpack_from("<3f", data, off); off += 12
    radius = struct.unpack_from("<f", data, off)[0]; off += 4
    return {'name': name, 'parent_bone': parent_bone,
            'pos': (px, py, pz), 'radius': radius}


# ═══════════════════════════════════════════════════════════════════════════════
#  RFA PARSER
# ═══════════════════════════════════════════════════════════════════════════════

RFA_MAGIC = 0x46564D56
RFA_TPS = 4800.0


def parse_rfa(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != RFA_MAGIC:
        raise ValueError(f"Not RFA (0x{magic:08X})")
    version = struct.unpack_from("<I", data, 4)[0]
    # bytes 8-27: preserved for round-trip (ramp_in, ramp_out, etc.)
    header_extra = data[8:28]
    start_time = struct.unpack_from("<I", data, 16)[0]
    end_time = struct.unpack_from("<I", data, 20)[0]
    num_bones = struct.unpack_from("<I", data, 24)[0]

    bone_offsets = [struct.unpack_from("<I", data, 80+i*4)[0] for i in range(num_bones)]

    bones = []
    for bi in range(num_bones):
        off = bone_offsets[bi]
        unk_float = struct.unpack_from("<f", data, off)[0]
        num_rot, num_pos = struct.unpack_from("<2H", data, off+4)

        rot_keys = []
        roff = off + 8
        for ki in range(num_rot):
            time = struct.unpack_from("<i", data, roff)[0]
            qx, qy, qz, qw = struct.unpack_from("<4h", data, roff+4)
            roff += 16
            rot_keys.append({'time': time,
                             'quat': (qx/16383.0, qy/16383.0, qz/16383.0, qw/16383.0)})

        pos_keys = []
        poff = roff
        for ki in range(num_pos):
            time = struct.unpack_from("<i", data, poff)[0]
            in_tan = struct.unpack_from("<3f", data, poff+4)
            value = struct.unpack_from("<3f", data, poff+16)
            out_tan = struct.unpack_from("<3f", data, poff+28)
            poff += 40
            pos_keys.append({'time': time, 'in_tangent': in_tan,
                             'value': value, 'out_tangent': out_tan})

        bones.append({'rot_keys': rot_keys, 'pos_keys': pos_keys,
                      'blend_weight': unk_float})

    return {'version': version, 'header_extra': header_extra,
            'start_time': start_time, 'end_time': end_time,
            'num_bones': num_bones, 'bones': bones,
            'raw_bytes': data}


# ═══════════════════════════════════════════════════════════════════════════════
#  BIND POSE RECOVERY (Redux-exact)
# ═══════════════════════════════════════════════════════════════════════════════

def _quat_rotate(q, v):
    """Rotate vector v by quaternion q=(x,y,z,w)."""
    qx,qy,qz,qw = q
    vx,vy,vz = v
    tx = 2*(qy*vz - qz*vy)
    ty = 2*(qz*vx - qx*vz)
    tz = 2*(qx*vy - qy*vx)
    return (vx + qw*tx + qy*tz - qz*ty,
            vy + qw*ty + qz*tx - qx*tz,
            vz + qw*tz + qx*ty - qy*tx)


def _quat_mul(a, b):
    """Multiply quaternions a*b, both as (x,y,z,w)."""
    ax,ay,az,aw = a
    bx,by,bz,bw = b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)


def _quat_normalize(q):
    ln = math.sqrt(sum(c**2 for c in q))
    return tuple(c/ln for c in q) if ln > 0 else q


def recover_bind_poses_rh(bones):
    """Redux-exact bind pose recovery. Works in RH Y-up space.
    Returns (bind_rot_rh[], bind_pos_rh[], local_transforms[])."""
    n = len(bones)
    bind_rot = [None]*n
    bind_pos = [None]*n

    for bi, b in enumerate(bones):
        qx,qy,qz,qw = b['inv_bind_quat']
        tx,ty,tz = b['inv_bind_pos']
        rh_q = _quat_normalize((-qx, qy, qz, qw))
        rh_t = (-tx, ty, tz)
        br = _quat_normalize((-rh_q[0], -rh_q[1], -rh_q[2], rh_q[3]))
        bp = _quat_rotate(br, (-rh_t[0], -rh_t[1], -rh_t[2]))
        bind_rot[bi] = br
        bind_pos[bi] = bp

    local_transforms = []
    for bi, b in enumerate(bones):
        pi = b['parent_index']
        if pi >= 0 and pi < n:
            pr = bind_rot[pi]
            inv_pr = _quat_normalize((-pr[0], -pr[1], -pr[2], pr[3]))
            lr = _quat_normalize(_quat_mul(inv_pr, bind_rot[bi]))
            dx = bind_pos[bi][0] - bind_pos[pi][0]
            dy = bind_pos[bi][1] - bind_pos[pi][1]
            dz = bind_pos[bi][2] - bind_pos[pi][2]
            lp = _quat_rotate(inv_pr, (dx, dy, dz))
        else:
            lr = bind_rot[bi]
            lp = bind_pos[bi]
        local_transforms.append({'rotation': lr, 'translation': lp})

    return bind_rot, bind_pos, local_transforms


# ═══════════════════════════════════════════════════════════════════════════════
#  BLENDER MESH IMPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _import_mesh(v3c, arm_obj, bone_names, lod_index=0, name_suffix=""):
    """Create Blender mesh from V3C data, parent to armature, assign weights."""
    sm = v3c['submeshes'][0]
    if lod_index >= len(sm['lods']):
        return None
    lod = sm['lods'][lod_index]
    chunks = lod['chunks']
    tex_list = lod['textures']
    num_bones = len(v3c['bones'])

    # Merge all chunks into one mesh
    all_verts = []
    all_faces = []
    all_uvs = []
    all_normals = []
    all_bone_links = []  # per-vertex (weights, bone_indices)
    all_mat_indices = []
    vert_offset = 0

    for ci, chunk in enumerate(chunks):
        for p in chunk['positions']:
            all_verts.append(rf_to_bl_pos(*p))
        for n in chunk['normals']:
            all_normals.append(rf_to_bl_pos(*n))
        for u, v in chunk['uvs']:
            all_uvs.append((u, 1.0 - v))  # flip V axis
        for tri in chunk['triangles']:
            # Flip winding: RF is LH (CW), Blender is RH (CCW) — swap indices 1↔2
            all_faces.append((tri[0]+vert_offset, tri[2]+vert_offset, tri[1]+vert_offset))
            all_mat_indices.append(chunk['texture_index'])
        for bl_w, bl_b in chunk['bone_links']:
            cj = tuple(b if b < num_bones else 0 for b in bl_b)
            cw = tuple(w if b < num_bones else 0 for w, b in zip(bl_w, bl_b))
            all_bone_links.append((cw, cj))
        vert_offset += len(chunk['positions'])

    # Create mesh
    mesh_name = sm['name'] + name_suffix
    mesh = bpy.data.meshes.new(mesh_name)
    if not all_verts or not all_faces:
        return None
    mesh.from_pydata([tuple(v) for v in all_verts], [], all_faces)
    mesh.validate(clean_customdata=False)

    # Materials with proper image texture nodes
    mat_flags_list = sm.get('material_flags', [0] * len(tex_list))
    for tex_id, tex_name in tex_list:
        mat_name = tex_name.replace('.tga', '')
        mat_flag = mat_flags_list[tex_id] if tex_id < len(mat_flags_list) else 0
        has_alpha = bool(mat_flag & 0x08)

        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(mat_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            bsdf = nodes.get('Principled BSDF')
            if not bsdf:
                bsdf = nodes.new('ShaderNodeBsdfPrincipled')

            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (-300, 300)

            img = bpy.data.images.get(tex_name)
            if not img:
                img = bpy.data.images.new(tex_name, width=1, height=1)
                img.filepath = tex_name
                img.source = 'FILE'
            tex_node.image = img

            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            mat.use_backface_culling = False

            if has_alpha:
                links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
                if hasattr(mat, 'blend_method'):
                    mat.blend_method = 'CLIP'
                if hasattr(mat, 'shadow_method'):
                    mat.shadow_method = 'CLIP'
                if hasattr(mat, 'surface_render_method'):
                    mat.surface_render_method = 'DITHERED'

        mesh.materials.append(mat)

    # Material assignment
    for fi, mat_idx in enumerate(all_mat_indices):
        if fi < len(mesh.polygons):
            mesh.polygons[fi].material_index = mat_idx

    # UVs
    if all_uvs:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        for li, loop in enumerate(mesh.loops):
            vi = loop.vertex_index
            if vi < len(all_uvs):
                uv_layer.data[li].uv = all_uvs[vi]

    # Normals
    if all_normals:
        try:
            normals_list = [tuple(all_normals[l.vertex_index]) if l.vertex_index < len(all_normals)
                            else (0, 0, 1) for l in mesh.loops]
            if hasattr(mesh, 'normals_split_custom_set'):
                mesh.normals_split_custom_set(normals_list)
        except Exception:
            pass  # Skip custom normals if it fails (known crash in some Blender versions)

    mesh.update()

    # Create object
    obj = bpy.data.objects.new(mesh_name, mesh)
    obj['rf_original_mesh'] = True
    obj['rf_lod_level'] = lod_index
    obj['rf_lod_distance'] = lod['distance']
    bpy.context.collection.objects.link(obj)

    # Vertex groups (for bone weights)
    if bone_names and all_bone_links:
        for bname in bone_names:
            obj.vertex_groups.new(name=bname)

        for vi, (weights, bone_ids) in enumerate(all_bone_links):
            for wi in range(4):
                if weights[wi] > 0 and bone_ids[wi] < len(bone_names):
                    # Divide by 255 (not sum) to preserve exact byte values on round-trip
                    w_float = weights[wi] / 255.0
                    if w_float > 0:
                        obj.vertex_groups[bone_names[bone_ids[wi]]].add(
                            [vi], w_float, 'REPLACE')

    # Parent to armature
    if arm_obj:
        obj.parent = arm_obj
        mod = obj.modifiers.new(name='Armature', type='ARMATURE')
        mod.object = arm_obj

    return obj


# ═══════════════════════════════════════════════════════════════════════════════
#  BLENDER COLLISION SPHERE IMPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _import_cspheres(v3c, arm_obj, bone_names):
    """Create collision sphere empties parented to bones."""
    cspheres = v3c.get('cspheres', [])
    if not cspheres or not arm_obj:
        return []

    created = []
    for cs in cspheres:
        name = cs['name']
        parent_bi = cs['parent_bone']
        rx, ry, rz = cs['pos']
        radius = cs['radius']

        # Create empty with sphere display
        empty = bpy.data.objects.new(f"CS_{name}", None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = radius
        empty.show_in_front = True
        bpy.context.collection.objects.link(empty)

        # Parent to bone
        empty.parent = arm_obj
        if 0 <= parent_bi < len(bone_names):
            bone_name = bone_names[parent_bi]
            empty.parent_type = 'BONE'
            empty.parent_bone = bone_name
            # Bone parenting offsets by bone length along Y — compensate
            bone = arm_obj.data.bones.get(bone_name)
            bone_len = bone.length if bone else 0
            # Convert RF offset to Blender space and subtract bone tail offset
            bl_offset = rf_to_bl_pos(rx, ry, rz)
            empty.location = (bl_offset[0], bl_offset[1] - bone_len, bl_offset[2])
        else:
            empty.location = rf_to_bl_pos(rx, ry, rz)

        empty.show_name = False
        empty.color = (0.0, 1.0, 0.5, 0.3)
        created.append(empty)

    return created


# ═══════════════════════════════════════════════════════════════════════════════
#  BLENDER PROP POINT (DUMMY) IMPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _import_prop_points(v3c, arm_obj, bone_names):
    """Create prop point empties (dummies) parented to bones.
    These are attachment points for weapons, flags, eyes, etc."""
    sm = v3c['submeshes'][0]
    lod = sm['lods'][0]
    prop_points = lod.get('prop_points', [])
    if not prop_points or not arm_obj:
        return []

    # Display type by name for visual clarity
    _DISPLAY = {
        'eye': ('SPHERE', 0.015),
        'mouth': ('SPHERE', 0.01),
        'helmet': ('CUBE', 0.02),
        '$prop_flag': ('PLAIN_AXES', 0.05),
        'corpse_carry': ('PLAIN_AXES', 0.04),
    }

    created = []
    for pp in prop_points:
        name = pp['name']
        parent_bi = pp['parent_bone']
        rx, ry, rz = pp['pos']
        qx, qy, qz, qw = pp['quat']

        disp_type, disp_size = _DISPLAY.get(name, ('ARROWS', 0.03))

        empty = bpy.data.objects.new(f"PP_{name}", None)
        empty.empty_display_type = disp_type
        empty.empty_display_size = disp_size
        empty.show_in_front = True
        empty['rf_prop_point'] = True
        bpy.context.collection.objects.link(empty)

        # Parent to bone
        empty.parent = arm_obj
        if 0 <= parent_bi < len(bone_names):
            bone_name = bone_names[parent_bi]
            empty.parent_type = 'BONE'
            empty.parent_bone = bone_name
            bone = arm_obj.data.bones.get(bone_name)
            bone_len = bone.length if bone else 0
            bl_offset = rf_to_bl_pos(rx, ry, rz)
            empty.location = (bl_offset[0], bl_offset[1] - bone_len, bl_offset[2])
        else:
            empty.location = rf_to_bl_pos(rx, ry, rz)

        # Apply orientation
        bl_rot = rf_to_bl_quat(qx, qy, qz, qw)
        empty.rotation_mode = 'QUATERNION'
        empty.rotation_quaternion = bl_rot

        empty.show_name = True
        created.append(empty)

    return created
# ═══════════════════════════════════════════════════════════════════════════════

def _import_armature(v3c):
    """Create Blender armature from V3C bone data. Returns armature object and bone name list."""
    bones_data = v3c['bones']
    if not bones_data:
        return None, []

    bind_rot_rh, bind_pos_rh, local_transforms = recover_bind_poses_rh(bones_data)

    # Compute world positions and rotations in Blender space
    world_pos_bl = []
    world_rot_bl = []
    for bi in range(len(bones_data)):
        world_pos_bl.append(rh_to_bl_pos(*bind_pos_rh[bi]))
        world_rot_bl.append(rh_to_bl_quat_xyzw(*bind_rot_rh[bi]))

    # Build child map
    children = [[] for _ in range(len(bones_data))]
    for bi, b in enumerate(bones_data):
        pi = b['parent_index']
        if pi >= 0 and pi < len(bones_data):
            children[pi].append(bi)

    # Create armature
    arm = bpy.data.armatures.new("Armature")
    arm.display_type = 'STICK'
    arm_obj = bpy.data.objects.new("Armature", arm)
    bpy.context.collection.objects.link(arm_obj)

    # Ensure clean context for mode switching
    try:
        if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    try:
        bpy.ops.object.select_all(action='DESELECT')
    except Exception:
        pass

    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm.edit_bones

    bone_names = []
    bl_bones = []

    for bi, b in enumerate(bones_data):
        eb = edit_bones.new(b['name'])
        bone_names.append(b['name'])

        head = world_pos_bl[bi]
        rot_mat = world_rot_bl[bi].to_matrix()

        # Y axis of the bind rotation = bone direction in Blender
        y_axis = rot_mat @ Vector((0, 1, 0))
        z_axis = rot_mat @ Vector((0, 0, 1))

        # Bone length: distance to nearest child, or fraction of parent distance
        bone_len = 0.03
        if children[bi]:
            min_dist = min((world_pos_bl[ci] - head).length for ci in children[bi])
            if min_dist > 0.005:
                bone_len = min_dist * 0.5
        else:
            pi = b['parent_index']
            if pi >= 0 and pi < len(bones_data):
                pdist = (head - world_pos_bl[pi]).length
                bone_len = max(pdist * 0.3, 0.02)

        eb.head = head
        eb.tail = head + y_axis.normalized() * bone_len
        eb.align_roll(z_axis)

        bl_bones.append(eb)

    # Set parents
    for bi, b in enumerate(bones_data):
        pi = b['parent_index']
        if pi >= 0 and pi < len(bones_data):
            bl_bones[bi].parent = bl_bones[pi]

    bpy.ops.object.mode_set(mode='OBJECT')

    return arm_obj, bone_names


# ═══════════════════════════════════════════════════════════════════════════════
#  BLENDER ANIMATION IMPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _import_rfa(rfa, arm_obj, bones_data, anim_name):
    """Import RFA animation onto armature. Uses the proven RH conversion."""
    if not arm_obj or not arm_obj.data:
        return

    num_anim_bones = min(rfa['num_bones'], len(bones_data))
    fps = bpy.context.scene.render.fps

    # Recover rest pose (same as used during armature creation)
    _, _, local_transforms_rest = recover_bind_poses_rh(bones_data)

    # Create action
    action = bpy.data.actions.new(name=anim_name)
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = action

    # Store RFA metadata for round-trip export
    action['rf_version'] = rfa.get('version', 1)
    if 'header_extra' in rfa:
        action['rf_header_extra'] = base64.b64encode(rfa['header_extra']).decode('ascii')
    blend_weights = [rfa['bones'][bi].get('blend_weight', 1.0)
                     for bi in range(min(rfa['num_bones'], len(bones_data)))]
    action['rf_blend_weights'] = json.dumps(blend_weights)

    # Store original RFA binary for perfect round-trip (bypasses float conversion)
    if 'raw_bytes' in rfa:
        action['rf_original_rfa'] = base64.b64encode(rfa['raw_bytes']).decode('ascii')

    # Blender 5.0+ uses layered actions with fcurve_ensure_for_datablock
    use_new_api = hasattr(action, 'fcurve_ensure_for_datablock')

    def make_fcurve(data_path, index):
        if use_new_api:
            return action.fcurve_ensure_for_datablock(arm_obj, data_path, index=index)
        else:
            return action.fcurves.new(data_path=data_path, index=index)

    # For each bone, convert animation keys to pose bone space
    for bi in range(num_anim_bones):
        rb = rfa['bones'][bi]
        bone_name = bones_data[bi]['name']
        pb = arm_obj.pose.bones.get(bone_name)
        if not pb:
            continue

        # Use Blender's actual bone matrix for precise delta computation
        if pb.parent:
            local_rest_mat = pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        else:
            local_rest_mat = pb.bone.matrix_local
        bl_rest_rot = local_rest_mat.to_quaternion()
        bl_rest_pos = local_rest_mat.to_translation()

        # Rotation keyframes
        if rb['rot_keys']:
            data_path = f'pose.bones["{bone_name}"].rotation_quaternion'
            for ch in range(4):
                fc = make_fcurve(data_path, ch)
                kf_points = []
                for rk in rb['rot_keys']:
                    t = rk['time'] / RFA_TPS
                    frame = t * fps + 1
                    # Convert anim rotation to RH then Blender
                    aqx, aqy, aqz, aqw = rk['quat']
                    anim_rh = _quat_normalize((-aqx, aqy, aqz, aqw))
                    anim_bl = rh_to_bl_quat_xyzw(*anim_rh)
                    # Pose rotation = rest_inv * anim
                    rest_inv = bl_rest_rot.inverted()
                    pose_rot = rest_inv @ anim_bl
                    # Ensure shortest rotation path
                    if pose_rot.w < 0:
                        pose_rot = Quaternion((-pose_rot.w, -pose_rot.x, -pose_rot.y, -pose_rot.z))
                    kf_points.append((frame, pose_rot[ch]))
                fc.keyframe_points.add(count=len(kf_points))
                for ki, (frame, value) in enumerate(kf_points):
                    fc.keyframe_points[ki].co = (frame, value)
                    fc.keyframe_points[ki].interpolation = 'LINEAR'

        # Translation keyframes
        if rb['pos_keys']:
            data_path = f'pose.bones["{bone_name}"].location'
            for ch in range(3):
                fc = make_fcurve(data_path, ch)
                kf_points = []
                for pk in rb['pos_keys']:
                    t = pk['time'] / RFA_TPS
                    frame = t * fps + 1
                    # Convert anim position to RH then Blender
                    vx, vy, vz = pk['value']
                    anim_pos_bl = rh_to_bl_pos(-vx, vy, vz)
                    # Pose location must be in bone-local space (rotated by rest orientation)
                    delta_parent = anim_pos_bl - bl_rest_pos
                    delta_local = bl_rest_rot.inverted().to_matrix() @ delta_parent
                    kf_points.append((frame, delta_local[ch]))
                fc.keyframe_points.add(count=len(kf_points))
                for ki, (frame, value) in enumerate(kf_points):
                    fc.keyframe_points[ki].co = (frame, value)
                    fc.keyframe_points[ki].interpolation = 'LINEAR'

    # Set frame range
    start_frame = (rfa['start_time'] / RFA_TPS) * fps + 1
    end_frame = (rfa['end_time'] / RFA_TPS) * fps + 1
    bpy.context.scene.frame_start = int(start_frame)
    bpy.context.scene.frame_end = int(end_frame)
    bpy.context.scene.frame_set(int(start_frame))


# ═══════════════════════════════════════════════════════════════════════════════
#  RFA EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _get_action_fcurves(action, arm_obj):
    """Get fcurves from an action, handling both legacy and Blender 5.0 layered APIs.
    Blender 5.0: action.layers[].strips[].channelbags[].fcurves
    Legacy (<5.0): action.fcurves directly."""
    # Blender 5.0+: layered action structure
    if hasattr(action, 'layers'):
        fcs = []
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, 'channelbags'):
                    for cbag in strip.channelbags:
                        fcs.extend(cbag.fcurves)
        if fcs:
            return fcs
    # Legacy fallback
    if hasattr(action, 'fcurves'):
        return list(action.fcurves)
    return []


def _collect_bone_keyframes(fcurves, bone_name):
    """Collect rotation and position keyframes for a bone from fcurves.
    Returns (rot_frames: sorted list, loc_frames: sorted list,
             rot_fcs: [4 fcurves or None], loc_fcs: [3 fcurves or None])."""
    rot_path = f'pose.bones["{bone_name}"].rotation_quaternion'
    loc_path = f'pose.bones["{bone_name}"].location'

    rot_fcs = [None] * 4
    loc_fcs = [None] * 3
    rot_frames = set()
    loc_frames = set()

    for fc in fcurves:
        if fc.data_path == rot_path and 0 <= fc.array_index < 4:
            rot_fcs[fc.array_index] = fc
            for kp in fc.keyframe_points:
                rot_frames.add(kp.co[0])
        elif fc.data_path == loc_path and 0 <= fc.array_index < 3:
            loc_fcs[fc.array_index] = fc
            for kp in fc.keyframe_points:
                loc_frames.add(kp.co[0])

    return sorted(rot_frames), sorted(loc_frames), rot_fcs, loc_fcs


def _export_rfa(arm_obj, action, filepath, bones_data):
    """Export a Blender action as an RFA binary file.

    Reverses the import conversion:
      Import:  RF → RH (negate x) → Blender → pose delta (rest_inv @ anim)
      Export:  pose delta → anim (rest @ pose) → Blender → RH → RF

    If original RFA binary was stored during import, writes it directly
    for perfect byte-exact round-trip of unmodified animations.
    """
    # Check for stored original RFA binary (perfect round-trip)
    original_b64 = action.get('rf_original_rfa', '')
    if original_b64:
        try:
            original_data = base64.b64decode(original_b64)
            with open(filepath, 'wb') as f:
                f.write(original_data)
            # Return timing from original
            start_time = struct.unpack_from("<I", original_data, 16)[0]
            end_time = struct.unpack_from("<I", original_data, 20)[0]
            return start_time, end_time
        except Exception:
            pass  # Fall through to normal export

    fps = bpy.context.scene.render.fps
    num_bones = len(bones_data)

    # Recover rest pose in RH space (same math as import)
    _, _, local_transforms_rest = recover_bind_poses_rh(bones_data)

    # Get Blender bone rest rotations/positions (same reference frame as import)
    bone_rest = []
    for bi in range(num_bones):
        bname = bones_data[bi]['name']
        bone = arm_obj.data.bones.get(bname)
        if bone:
            if bone.parent:
                lmat = bone.parent.matrix_local.inverted() @ bone.matrix_local
            else:
                lmat = bone.matrix_local
            bone_rest.append((lmat.to_quaternion(), lmat.to_translation()))
        else:
            bone_rest.append((Quaternion((1, 0, 0, 0)), Vector((0, 0, 0))))

    # Collect fcurves
    fcurves = _get_action_fcurves(action, arm_obj)

    # ── Per-bone: convert keyframes back to RF space ──
    bone_export = []
    for bi in range(num_bones):
        bname = bones_data[bi]['name']
        bl_rest_rot, bl_rest_pos = bone_rest[bi]

        rot_frames, loc_frames, rot_fcs, loc_fcs = _collect_bone_keyframes(fcurves, bname)

        # -- Rotation keys --
        rot_keys = []
        if rot_frames:
            for frame in rot_frames:
                pose_q = Quaternion((
                    rot_fcs[0].evaluate(frame) if rot_fcs[0] else 1.0,
                    rot_fcs[1].evaluate(frame) if rot_fcs[1] else 0.0,
                    rot_fcs[2].evaluate(frame) if rot_fcs[2] else 0.0,
                    rot_fcs[3].evaluate(frame) if rot_fcs[3] else 0.0,
                ))
                # Reverse import: anim_bl = rest_rot @ pose_rot
                anim_bl = bl_rest_rot @ pose_q
                # Convert Blender → RF
                rf_qx, rf_qy, rf_qz, rf_qw = bl_to_rf_quat(anim_bl)
                time_ticks = int(round((frame - 1) / fps * RFA_TPS))
                rot_keys.append((time_ticks, rf_qx, rf_qy, rf_qz, rf_qw))
        else:
            # No animation: write rest pose as single key at time 0
            lt = local_transforms_rest[bi]
            rh_x, rh_y, rh_z, rh_w = lt['rotation']
            rf_qx, rf_qy, rf_qz, rf_qw = -rh_x, rh_y, rh_z, rh_w  # RH→RF
            rot_keys.append((0, rf_qx, rf_qy, rf_qz, rf_qw))

        # -- Position keys --
        pos_keys = []
        if loc_frames:
            for frame in loc_frames:
                loc = Vector((
                    loc_fcs[0].evaluate(frame) if loc_fcs[0] else 0.0,
                    loc_fcs[1].evaluate(frame) if loc_fcs[1] else 0.0,
                    loc_fcs[2].evaluate(frame) if loc_fcs[2] else 0.0,
                ))
                # Reverse import: delta_parent = rest_rot.matrix @ location
                delta_parent = bl_rest_rot.to_matrix() @ loc
                anim_pos_bl = delta_parent + bl_rest_pos
                # Convert Blender → RF
                rf_pos = bl_to_rf_pos(anim_pos_bl)
                time_ticks = int(round((frame - 1) / fps * RFA_TPS))
                pos_keys.append((time_ticks, rf_pos))
        # (No fallback for position — bones with no pos keys simply get 0 pos keys)

        bone_export.append({'rot_keys': rot_keys, 'pos_keys': pos_keys})

    # ── Recover stored metadata for round-trip fidelity ──
    version = int(action.get('rf_version', 1))
    blend_weights_json = action.get('rf_blend_weights', '[]')
    blend_weights = json.loads(blend_weights_json) if isinstance(blend_weights_json, str) else []

    # ── Build RFA binary ──
    # Header: 80 bytes
    header = bytearray(80)
    struct.pack_into("<I", header, 0, RFA_MAGIC)
    struct.pack_into("<I", header, 4, version)

    # Restore bytes 8-27 from original if available (ramp, timing, etc.)
    header_extra_b64 = action.get('rf_header_extra', '')
    if header_extra_b64:
        try:
            extra = base64.b64decode(header_extra_b64)
            header[8:8+len(extra)] = extra[:20]  # bytes 8-27
        except Exception:
            pass

    # Compute time range
    all_times = []
    for bd in bone_export:
        for rk in bd['rot_keys']:
            all_times.append(rk[0])
        for pk in bd['pos_keys']:
            all_times.append(pk[0])
    start_time = min(all_times) if all_times else 0
    end_time = max(all_times) if all_times else 0

    struct.pack_into("<I", header, 16, start_time)
    struct.pack_into("<I", header, 20, end_time)
    struct.pack_into("<I", header, 24, num_bones)

    # Compute bone data offsets
    bone_data_start = 80 + num_bones * 4
    bone_offsets = []
    bone_blobs = []
    current_off = bone_data_start

    for bi, bd in enumerate(bone_export):
        bone_offsets.append(current_off)
        blob = bytearray()

        # Per-bone header: blend_weight(4) + num_rot(2) + num_pos(2) = 8 bytes
        bw = blend_weights[bi] if bi < len(blend_weights) else 1.0
        blob += struct.pack("<f", bw)
        blob += struct.pack("<2H", len(bd['rot_keys']), len(bd['pos_keys']))

        # Rotation keys: 16 bytes each (time:4 + qxyzw:8 + pad:4)
        for time_t, qx, qy, qz, qw in bd['rot_keys']:
            iqx = max(-32767, min(32767, int(round(qx * 16383.0))))
            iqy = max(-32767, min(32767, int(round(qy * 16383.0))))
            iqz = max(-32767, min(32767, int(round(qz * 16383.0))))
            iqw = max(-32767, min(32767, int(round(qw * 16383.0))))
            blob += struct.pack("<i4h", time_t, iqx, iqy, iqz, iqw)
            blob += b'\x00\x00\x00\x00'  # 4 bytes padding

        # Position keys: 40 bytes each (time:4 + in_tan:12 + value:12 + out_tan:12)
        pk_list = bd['pos_keys']
        for i, (time_t, pos) in enumerate(pk_list):
            x, y, z = pos
            # Catmull-Rom tangent approximation
            if len(pk_list) <= 1:
                in_t = out_t = (0.0, 0.0, 0.0)
            elif i == 0:
                n = pk_list[1]
                dt = max((n[0] - time_t) / RFA_TPS, 0.001)
                in_t = out_t = tuple((n[1][j] - pos[j]) / dt for j in range(3))
            elif i == len(pk_list) - 1:
                p = pk_list[i - 1]
                dt = max((time_t - p[0]) / RFA_TPS, 0.001)
                in_t = out_t = tuple((pos[j] - p[1][j]) / dt for j in range(3))
            else:
                p, n = pk_list[i - 1], pk_list[i + 1]
                dt = max((n[0] - p[0]) / RFA_TPS, 0.001)
                in_t = out_t = tuple((n[1][j] - p[1][j]) / dt for j in range(3))

            blob += struct.pack("<i", time_t)
            blob += struct.pack("<3f", *in_t)
            blob += struct.pack("<3f", x, y, z)
            blob += struct.pack("<3f", *out_t)

        bone_blobs.append(blob)
        current_off += len(blob)

    # Assemble
    output = bytearray(header)
    for off in bone_offsets:
        output += struct.pack("<I", off)
    for blob in bone_blobs:
        output += blob

    with open(filepath, 'wb') as f:
        f.write(output)

    return start_time, end_time


# ═══════════════════════════════════════════════════════════════════════════════
#  ANIMATION DATABASE (stock RF)
# ═══════════════════════════════════════════════════════════════════════════════

# RF Animation Database: model filename → required animation files
# Auto-generated from entity.tbl + weapons.tbl (stock Red Faction)
_RF_ANIM_DB = {
    "admin_fem.v3c": [
        "adfm_cower.rfa", "adfm_crouch.rfa", "adfm_death_blast_backwards.rfa", "adfm_death_blast_forwards.rfa",
        "adfm_death_chest_backwards.rfa", "adfm_death_chest_forwards.rfa", "adfm_death_crouch.rfa", "adfm_death_generic.rfa",
        "adfm_death_head_backwards.rfa", "adfm_death_head_forwards.rfa", "adfm_death_leg_left.rfa", "adfm_flinch_back.rfa",
        "adfm_flinch_chest.rfa", "adfm_flinch_leg_left.rfa", "adfm_flinch_leg_right.rfa", "adfm_freefall.rfa",
        "adfm_hit_alarm.rfa", "adfm_run_flail.rfa", "admin_fem_corpsecarry.rfa", "admin_fem_corpsedrop.rfa",
        "admin_fem_idle01.rfa", "admin_fem_run.rfa", "admin_fem_run_flee.rfa", "admin_fem_stand.rfa",
        "admin_fem_talk.rfa", "admin_fem_talk_short.rfa", "admin_fem_walk.rfa",
    ],
    "admin_male.v3c": [
        "admin_male_talk.rfa", "admin_male_talk_short.rfa", "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa",
        "tech01_cower_loop.rfa", "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa",
        "tech01_death_generic.rfa", "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa", "tech01_death_leg_l.rfa",
        "tech01_death_leg_r.rfa", "tech01_death_spin_fall_l.rfa", "tech01_death_torso_forward.rfa", "tech01_flinch.rfa",
        "tech01_flinch_back.rfa", "tech01_flinch_leg_l.rfa", "tech01_flinch_leg_r.rfa", "tech01_freefall.rfa",
        "tech01_hit_alarm.rfa", "tech01_idle_01.rfa", "tech01_run.rfa", "tech01_run_flail.rfa",
        "tech01_run_flee.rfa", "tech01_stand.rfa", "tech01_walk.rfa",
    ],
    "admin_male2.v3c": [
        "adm2_cower.rfa", "adm2_death_chest_forwards.rfa", "adm2_death_generic.rfa", "adm2_flinch_back.rfa",
        "adm2_flinch_front.rfa", "adm2_flinch_leg_left.rfa", "adm2_flinch_leg_right.rfa", "adm2_idle.rfa",
        "adm2_run.rfa", "adm2_run_flail.rfa", "adm2_run_flee.rfa", "adm2_stand.rfa",
        "adm2_talk.rfa", "adm2_talk_short.rfa", "adm2_walk.rfa", "tech01_blast_back.rfa",
        "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa", "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa",
        "tech01_death_crouch.rfa", "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa", "tech01_death_leg_r.rfa",
        "tech01_death_spin_fall_l.rfa", "tech01_freefall.rfa", "tech01_hit_alarm.rfa",
    ],
    "auto_turret.v3c": ["turt_attack_stand.rfa", "turt_stand.rfa"],
    "baby_reeper.v3c": [
        "brpr_attack.rfa", "brpr_attack_stand.rfa", "brpr_death.rfa", "brpr_flinch.rfa",
        "brpr_run.rfa", "brpr_stand.rfa", "brpr_walk.rfa",
    ],
    "bat1.v3c": ["bat1_fly_fast.rfa", "bat1_fly_slow.rfa", "bat1_idle.rfa"],
    "big_snake.v3c": [
        "bsnk_attack_bite.rfa", "bsnk_attack_spit.rfa", "bsnk_attack_stand.rfa", "bsnk_attack_walk.rfa",
        "bsnk_death.rfa", "bsnk_stand.rfa", "bsnk_walk.rfa",
    ],
    "capek.v3c": [
        "capk_crouch.rfa", "capk_death.rfa", "capk_flinch_back.rfa", "capk_flinch_chest.rfa",
        "capk_flinch_leg_left.rfa", "capk_flinch_leg_right.rfa", "capk_freefall.rfa", "capk_idle_01.rfa",
        "capk_nano_attack.rfa", "capk_stand.rfa", "capk_talk.rfa", "capk_talk_short.rfa",
        "capk_walk.rfa",
    ],
    "combot.v3c": [
        "com1_attack_primary.rfa", "com1_drop_arms.rfa", "com1_flinch_hover.rfa", "com1_hover.rfa",
        "com1_hover_idle.rfa", "com1_ready_arms.rfa",
    ],
    "cutter01.v3c": ["cutter_attack-new.rfa", "cutter_fly.rfa", "cutter_hover.rfa"],
    "drillmissile01.vfx": [
        "fp_rocket_ammocheck.rfa", "fp_rocket_draw.rfa", "fp_rocket_fire.rfa", "fp_rocket_hold.rfa",
        "fp_rocket_holster.rfa", "fp_rocket_jump.rfa", "fp_rocket_reload.rfa", "fp_rocket_run.rfa",
        "fp_rocket_screenwipe.rfa",
    ],
    "drone01.v3c": [
        "drone_attack_armgun.rfa", "drone_attack_melee.rfa", "drone_hover.rfa", "drone_hover_droparms.rfa",
        "drone_hover_flinch.rfa", "drone_hover_idle.rfa", "drone_hover_readyarms.rfa",
    ],
    "edf_ship.v3c": ["cs10_edf1_shot01_15.rfa", "edf1_idle.rfa"],
    "elite_security_guard.v3c": [
        "esgd_attack_crouch.rfa", "esgd_attack_roll_left.rfa", "esgd_attack_roll_right.rfa", "esgd_attack_run.rfa",
        "esgd_attack_stand.rfa", "esgd_attack_walk.rfa", "esgd_firing_stand.rfa", "esgd_mp_reload.rfa",
        "esgd_stand.rfa", "esgd_talk.rfa", "esgd_talk_short.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa",
        "park_riotstick_reload.rfa", "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa",
        "park_run_flail.rfa", "ult2_corpse.rfa", "ult2_corpse2.rfa", "ult2_corpse3.rfa",
        "ult2_corpse_drop.rfa", "ult2_cower.rfa", "ult2_crouch.rfa", "ult2_crouch_death.rfa",
        "ult2_death_blast_forwards.rfa", "ult2_death_carry.rfa", "ult2_death_chest_backward.rfa", "ult2_death_chest_forward.rfa",
        "ult2_death_generic.rfa", "ult2_death_head_backwards.rfa", "ult2_death_head_forwards.rfa", "ult2_death_leg_left.rfa",
        "ult2_death_leg_right.rfa", "ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "ult2_flinch_chestb.rfa",
        "ult2_flinch_chestf.rfa", "ult2_flinch_legl.rfa", "ult2_flinch_legr.rfa", "ult2_flinch_stand.rfa",
        "ult2_hit_alarm.rfa", "ult2_idle_hithead.rfa", "ult2_idle_look.rfa", "ult2_idle_stretch.rfa",
        "ult2_jump.rfa", "ult2_on_turret.rfa", "ult2_run.rfa", "ult2_sidestep_left.rfa",
        "ult2_sidestep_right.rfa", "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "env_scientist.v3c": [
        "esci_cower.rfa", "esci_crouch.rfa", "esci_death_chest_forwards.rfa", "esci_death_generic.rfa",
        "esci_flinch_back.rfa", "esci_flinch_chest.rfa", "esci_flinch_leg_left.rfa", "esci_flinch_leg_right.rfa",
        "esci_idle.rfa", "esci_push_button.rfa", "esci_run.rfa", "esci_run_flee.rfa",
        "esci_stand.rfa", "esci_walk.rfa", "tech01_blast_back.rfa", "tech01_blast_forwards.rfa",
        "tech01_corpse_carry.rfa", "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa", "tech01_death_generic.rfa",
        "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa", "tech01_death_leg_r.rfa", "tech01_death_spin_fall_l.rfa",
        "tech01_freefall.rfa", "tech01_run_flail.rfa",
    ],
    "envirosuit_guard.v3c": [
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_talk.rfa",
        "engd_talk_short.rfa", "esgd_attack_run.rfa", "esgd_attack_stand.rfa", "esgd_attack_walk.rfa",
        "esgd_firing_stand.rfa", "esgd_mp_reload.rfa", "esgd_stand.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa",
        "park_riotstick_reload.rfa", "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa",
        "park_rstick_crouch_walk.rfa", "park_run_flail.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa",
        "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa",
        "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa",
        "park_sniper_attack_stand.rfa", "park_sniper_crouch_fire.rfa", "park_sniper_reload.rfa", "park_sniper_run.rfa",
        "park_sniper_stand.rfa", "park_sniper_stand_fire.rfa", "park_sniper_walk.rfa", "ult2_attack_crouch01.rfa",
        "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa", "ult2_corpse.rfa",
        "ult2_corpse2.rfa", "ult2_corpse3.rfa", "ult2_corpse_drop.rfa", "ult2_cower.rfa",
        "ult2_crouch_death.rfa", "ult2_death_blast_forwards.rfa", "ult2_death_carry.rfa", "ult2_death_chest_backward.rfa",
        "ult2_death_chest_forward.rfa", "ult2_death_generic.rfa", "ult2_death_head_backwards.rfa", "ult2_death_head_forwards.rfa",
        "ult2_death_leg_left.rfa", "ult2_death_leg_right.rfa", "ult2_draw.rfa", "ult2_firing_crouch.rfa",
        "ult2_firing_stand.rfa", "ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "ult2_flinch_chestb.rfa",
        "ult2_flinch_chestf.rfa", "ult2_flinch_legl.rfa", "ult2_flinch_legr.rfa", "ult2_flinch_stand.rfa",
        "ult2_hit_alarm.rfa", "ult2_idle_hithead.rfa", "ult2_idle_look.rfa", "ult2_idle_stretch.rfa",
        "ult2_jump.rfa", "ult2_on_turret.rfa", "ult2_reload.rfa", "ult2_run.rfa",
        "ult2_run_flee_ar.rfa", "ult2_sidestep_left.rfa", "ult2_sidestep_right.rfa", "ult2_stand.rfa",
        "ult2_walk.rfa",
    ],
    "eos.v3c": [
        "adfm_cower.rfa", "adfm_death_blast_backwards.rfa", "adfm_death_blast_forwards.rfa", "adfm_death_chest_backwards.rfa",
        "adfm_death_chest_forwards.rfa", "adfm_death_crouch.rfa", "adfm_death_generic.rfa", "adfm_death_head_backwards.rfa",
        "adfm_death_head_forwards.rfa", "adfm_death_leg_left.rfa", "adfm_flinch_back.rfa", "adfm_flinch_chest.rfa",
        "adfm_flinch_leg_left.rfa", "adfm_flinch_leg_right.rfa", "adfm_freefall.rfa", "adfm_run_flail.rfa",
        "admin_fem_corpsedrop.rfa", "admin_fem_idle01.rfa", "admin_fem_run_flee.rfa", "eos_sidestep_left.rfa",
        "eos_sidestep_right.rfa", "eos_talk.rfa", "eos_talk_short.rfa", "mnrf_mp_attack_crouch.rfa",
        "mnrf_mp_attack_stand.rfa", "mnrf_mp_crouch.rfa", "mnrf_mp_crouch_fire.rfa", "mnrf_mp_reload.rfa",
        "mnrf_mp_run.rfa", "mnrf_mp_stand.rfa", "mnrf_mp_stand_fire.rfa", "mnrf_mp_walk.rfa",
    ],
    "fish.v3c": ["fish_stand.rfa", "fish_swim_fast.rfa", "fish_swim_slow.rfa"],
    "grab.v3c": ["grab_flinch_hover.rfa", "grab_fly.rfa", "grab_hover.rfa", "grab_rock_drop.rfa", "grab_rock_up.rfa"],
    "hendrix.v3c": [
        "hndx_talk.rfa", "hndx_talk_short.rfa", "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa",
        "tech01_cower_loop.rfa", "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa",
        "tech01_death_generic.rfa", "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa", "tech01_death_leg_l.rfa",
        "tech01_death_leg_r.rfa", "tech01_death_spin_fall_l.rfa", "tech01_death_torso_forward.rfa", "tech01_flinch.rfa",
        "tech01_flinch_back.rfa", "tech01_flinch_leg_l.rfa", "tech01_flinch_leg_r.rfa", "tech01_freefall.rfa",
        "tech01_hit_alarm.rfa", "tech01_idle_01.rfa", "tech01_run.rfa", "tech01_run_flail.rfa",
        "tech01_run_flee.rfa", "tech01_stand.rfa", "tech01_walk.rfa",
    ],
    "masako.v3c": [
        "adfm_cower.rfa", "adfm_death_blast_forwards.rfa", "adfm_death_chest_backwards.rfa", "adfm_death_chest_forwards.rfa",
        "adfm_death_crouch.rfa", "adfm_death_generic.rfa", "adfm_death_head_backwards.rfa", "adfm_death_head_forwards.rfa",
        "adfm_death_leg_left.rfa", "adfm_death_leg_right.rfa", "adfm_freefall.rfa", "adfm_run_flail.rfa",
        "admin_fem_corpsecarry.rfa", "admin_fem_corpsedrop.rfa", "admin_fem_idle01.rfa", "admin_fem_run_flee.rfa",
        "admin_fem_walk.rfa", "eos_sidestep_left.rfa", "eos_sidestep_right.rfa", "masa_attack_roll_left.rfa",
        "masa_attack_roll_right.rfa", "masa_sar_attack_crouch.rfa", "masa_sar_attack_stand.rfa", "masa_sar_crouch.rfa",
        "masa_sar_crouch_fire.rfa", "masa_sar_flinch_back.rfa", "masa_sar_flinch_chest.rfa", "masa_sar_reload.rfa",
        "masa_sar_run.rfa", "masa_sar_stand.rfa", "masa_sar_stand_fire.rfa", "masa_sar_walk.rfa",
        "masa_talk.rfa", "masa_talk_short.rfa", "mnr3f_sr_crouch_walk.rfa",
    ],
    "medic01.v3c": [
        "medc_talk.rfa", "medc_talk_short.rfa", "medic01_heal_01.rfa", "tech01_blast_forwards.rfa",
        "tech01_corpse_carry.rfa", "tech01_cower_loop.rfa", "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa",
        "tech01_death_crouch.rfa", "tech01_death_generic.rfa", "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa",
        "tech01_death_leg_l.rfa", "tech01_death_leg_r.rfa", "tech01_death_spin_fall_l.rfa", "tech01_death_torso_forward.rfa",
        "tech01_flinch.rfa", "tech01_flinch_back.rfa", "tech01_flinch_leg_l.rfa", "tech01_flinch_leg_r.rfa",
        "tech01_freefall.rfa", "tech01_hit_alarm.rfa", "tech01_idle_01.rfa", "tech01_run.rfa",
        "tech01_run_flail.rfa", "tech01_run_flee.rfa", "tech01_stand.rfa", "tech01_walk.rfa",
    ],
    "merc_com.v3c": [
        "mcom_swim_stand.rfa", "mcom_swim_walk.rfa", "mrc1_attack_crouch.rfa", "mrc1_attack_crouch_rr.rfa",
        "mrc1_attack_fire.rfa", "mrc1_attack_run.rfa", "mrc1_attack_run_rr.rfa", "mrc1_attack_stand.rfa",
        "mrc1_attack_stand_grenade.rfa", "mrc1_attack_stand_rr.rfa", "mrc1_attack_walk.rfa", "mrc1_attack_walk_rr.rfa",
        "mrc1_corpse_carried.rfa", "mrc1_corpse_drop.rfa", "mrc1_cower.rfa", "mrc1_crouch.rfa",
        "mrc1_death_blast_back.rfa", "mrc1_death_blast_fore.rfa", "mrc1_death_crouch.rfa", "mrc1_death_generic.rfa",
        "mrc1_death_generic_chest.rfa", "mrc1_death_generic_head.rfa", "mrc1_death_head_back.rfa", "mrc1_death_head_fore.rfa",
        "mrc1_death_leg_l.rfa", "mrc1_death_leg_r.rfa", "mrc1_death_torso_back.rfa", "mrc1_death_torso_fore.rfa",
        "mrc1_fire_alt_ft.rfa", "mrc1_fire_grenade.rfa", "mrc1_fire_grenade_alt.rfa", "mrc1_fire_reload_rr.rfa",
        "mrc1_fire_reload_rr_c.rfa", "mrc1_fleerun.rfa", "mrc1_flinch_back.rfa", "mrc1_flinch_chest.rfa",
        "mrc1_flinch_leg_l.rfa", "mrc1_flinch_leg_r.rfa", "mrc1_freefall.rfa", "mrc1_ft_alt_fire_stand.rfa",
        "mrc1_idle.rfa", "mrc1_idle2.rfa", "mrc1_idle2_rr.rfa", "mrc1_reload-ft.rfa",
        "mrc1_reload.rfa", "mrc1_run.rfa", "mrc1_run_flail.rfa", "mrc1_sidestep_left.rfa",
        "mrc1_sidestep_right.rfa", "mrc1_stand-fire-ft.rfa", "mrc1_stand.rfa", "mrc1_stand_ft.rfa",
        "mrc1_walk.rfa", "mrc2_crouch_12mm.rfa", "mrc2_crouch_mp.rfa", "mrc2_crouch_rl.rfa",
        "mrc2_crouch_rs.rfa", "mrc2_crouch_rshield.rfa", "mrc2_crouch_sg.rfa", "mrc2_crouch_sr.rfa",
        "mrc2_crouch_walk_12mm.rfa", "mrc2_crouch_walk_ft.rfa", "mrc2_crouch_walk_grn.rfa", "mrc2_crouch_walk_hmac.rfa",
        "mrc2_crouch_walk_mp.rfa", "mrc2_crouch_walk_rl.rfa", "mrc2_crouch_walk_rr.rfa", "mrc2_crouch_walk_rs.rfa",
        "mrc2_crouch_walk_rshield.rfa", "mrc2_crouch_walk_sg.rfa", "mrc2_crouch_walk_smc.rfa", "mrc2_crouch_walk_sr.rfa",
        "mrc2_fire_12mm.rfa", "mrc2_fire_alt_sg.rfa", "mrc2_fire_mp.rfa", "mrc2_fire_rcharge.rfa",
        "mrc2_fire_rl.rfa", "mrc2_fire_rs.rfa", "mrc2_fire_rshield.rfa", "mrc2_fire_sg.rfa",
        "mrc2_fire_sr.rfa", "mrc2_reload_12mm.rfa", "mrc2_reload_mp.rfa", "mrc2_reload_rl.rfa",
        "mrc2_reload_rs.rfa", "mrc2_reload_sg.rfa", "mrc2_reload_sr.rfa", "mrc2_run_12mm.rfa",
        "mrc2_run_ft.rfa", "mrc2_run_mp.rfa", "mrc2_run_rl.rfa", "mrc2_run_rs.rfa",
        "mrc2_run_rshield.rfa", "mrc2_run_sg.rfa", "mrc2_run_sr.rfa", "mrc2_stand_12mm.rfa",
        "mrc2_stand_mp.rfa", "mrc2_stand_rl.rfa", "mrc2_stand_rs.rfa", "mrc2_stand_rshield.rfa",
        "mrc2_stand_sg.rfa", "mrc2_stand_sr.rfa", "mrc2_walk_12mm.rfa", "mrc2_walk_mp.rfa",
        "mrc2_walk_rl.rfa", "mrc2_walk_rs.rfa", "mrc2_walk_rshield.rfa", "mrc2_walk_sg.rfa",
        "mrc2_walk_sr.rfa", "mrch_attack_crouch.rfa", "mrch_attack_crouch_smw.rfa", "mrch_attack_fire.rfa",
        "mrch_attack_run.rfa", "mrch_attack_stand.rfa", "mrch_attack_stand_smw.rfa", "mrch_attack_walk.rfa",
        "mrch_attack_walk_smw.rfa", "mrch_crouch_fire.rfa", "mrch_fire_crouch_smw.rfa", "mrch_fire_smw.rfa",
        "mrch_flinch_back_smw.rfa", "mrch_flinch_front_smw.rfa", "mrch_idle_smw.rfa", "mrch_reload.rfa",
        "mrch_run_smw.rfa",
    ],
    "merc_grunt.v3c": [
        "mrc1_attack_crouch.rfa", "mrc1_attack_crouch_rr.rfa", "mrc1_attack_fire.rfa", "mrc1_attack_run.rfa",
        "mrc1_attack_run_rr.rfa", "mrc1_attack_stand.rfa", "mrc1_attack_stand_grenade.rfa", "mrc1_attack_stand_rr.rfa",
        "mrc1_attack_walk.rfa", "mrc1_attack_walk_rr.rfa", "mrc1_corpse_carried.rfa", "mrc1_corpse_drop.rfa",
        "mrc1_cower.rfa", "mrc1_crouch.rfa", "mrc1_death_blast_back.rfa", "mrc1_death_blast_fore.rfa",
        "mrc1_death_crouch.rfa", "mrc1_death_generic.rfa", "mrc1_death_generic_chest.rfa", "mrc1_death_generic_head.rfa",
        "mrc1_death_head_back.rfa", "mrc1_death_head_fore.rfa", "mrc1_death_leg_l.rfa", "mrc1_death_leg_r.rfa",
        "mrc1_death_torso_back.rfa", "mrc1_death_torso_fore.rfa", "mrc1_fire_grenade.rfa", "mrc1_fire_grenade_alt.rfa",
        "mrc1_fire_reload_rr.rfa", "mrc1_fire_reload_rr_c.rfa", "mrc1_fleerun.rfa", "mrc1_flinch_back.rfa",
        "mrc1_flinch_chest.rfa", "mrc1_flinch_leg_l.rfa", "mrc1_flinch_leg_r.rfa", "mrc1_freefall.rfa",
        "mrc1_ft_alt_fire_stand.rfa", "mrc1_idle.rfa", "mrc1_idle2.rfa", "mrc1_idle2_rr.rfa",
        "mrc1_reload-ft.rfa", "mrc1_reload.rfa", "mrc1_run.rfa", "mrc1_run_flail.rfa",
        "mrc1_sidestep_left.rfa", "mrc1_sidestep_right.rfa", "mrc1_stand-fire-ft.rfa", "mrc1_stand.rfa",
        "mrc1_stand_ft.rfa", "mrc1_walk.rfa", "mrc2_crouch_12mm.rfa", "mrc2_crouch_mp.rfa",
        "mrc2_crouch_rl.rfa", "mrc2_crouch_rs.rfa", "mrc2_crouch_rshield.rfa", "mrc2_crouch_sg.rfa",
        "mrc2_crouch_sr.rfa", "mrc2_crouch_walk_12mm.rfa", "mrc2_crouch_walk_ft.rfa", "mrc2_crouch_walk_grn.rfa",
        "mrc2_crouch_walk_hmac.rfa", "mrc2_crouch_walk_mp.rfa", "mrc2_crouch_walk_rl.rfa", "mrc2_crouch_walk_rr.rfa",
        "mrc2_crouch_walk_rs.rfa", "mrc2_crouch_walk_rshield.rfa", "mrc2_crouch_walk_sg.rfa", "mrc2_crouch_walk_smc.rfa",
        "mrc2_crouch_walk_sr.rfa", "mrc2_fire_12mm.rfa", "mrc2_fire_alt_sg.rfa", "mrc2_fire_mp.rfa",
        "mrc2_fire_rcharge.rfa", "mrc2_fire_rl.rfa", "mrc2_fire_rs.rfa", "mrc2_fire_rshield.rfa",
        "mrc2_fire_sg.rfa", "mrc2_fire_sr.rfa", "mrc2_reload_12mm.rfa", "mrc2_reload_mp.rfa",
        "mrc2_reload_rl.rfa", "mrc2_reload_rs.rfa", "mrc2_reload_sg.rfa", "mrc2_reload_sr.rfa",
        "mrc2_run_12mm.rfa", "mrc2_run_mp.rfa", "mrc2_run_rl.rfa", "mrc2_run_rs.rfa",
        "mrc2_run_rshield.rfa", "mrc2_run_sg.rfa", "mrc2_run_sr.rfa", "mrc2_stand_12mm.rfa",
        "mrc2_stand_mp.rfa", "mrc2_stand_rl.rfa", "mrc2_stand_rs.rfa", "mrc2_stand_rshield.rfa",
        "mrc2_stand_sg.rfa", "mrc2_stand_sr.rfa", "mrc2_walk_12mm.rfa", "mrc2_walk_mp.rfa",
        "mrc2_walk_rl.rfa", "mrc2_walk_rs.rfa", "mrc2_walk_rshield.rfa", "mrc2_walk_sg.rfa",
        "mrc2_walk_sr.rfa", "mrch_attack_crouch.rfa", "mrch_attack_crouch_smw.rfa", "mrch_attack_fire.rfa",
        "mrch_attack_run.rfa", "mrch_attack_stand.rfa", "mrch_attack_stand_smw.rfa", "mrch_attack_walk.rfa",
        "mrch_attack_walk_smw.rfa", "mrch_crouch_fire.rfa", "mrch_fire_crouch_smw.rfa", "mrch_fire_smw.rfa",
        "mrch_flinch_back_smw.rfa", "mrch_flinch_front_smw.rfa", "mrch_idle_smw.rfa", "mrch_reload.rfa",
        "mrch_run_smw.rfa",
    ],
    "merc_heavy.v3c": [
        "mrc1_attack_crouch.rfa", "mrc1_attack_crouch_rr.rfa", "mrc1_attack_fire.rfa", "mrc1_attack_run.rfa",
        "mrc1_attack_run_rr.rfa", "mrc1_attack_stand.rfa", "mrc1_attack_stand_grenade.rfa", "mrc1_attack_stand_rr.rfa",
        "mrc1_attack_walk.rfa", "mrc1_attack_walk_rr.rfa", "mrc1_corpse_carried.rfa", "mrc1_corpse_drop.rfa",
        "mrc1_cower.rfa", "mrc1_crouch.rfa", "mrc1_death_blast_back.rfa", "mrc1_death_blast_fore.rfa",
        "mrc1_death_crouch.rfa", "mrc1_death_generic.rfa", "mrc1_death_generic_chest.rfa", "mrc1_death_generic_head.rfa",
        "mrc1_death_head_back.rfa", "mrc1_death_head_fore.rfa", "mrc1_death_leg_l.rfa", "mrc1_death_leg_r.rfa",
        "mrc1_death_torso_back.rfa", "mrc1_death_torso_fore.rfa", "mrc1_fire_grenade.rfa", "mrc1_fire_grenade_alt.rfa",
        "mrc1_fire_reload_rr.rfa", "mrc1_fire_reload_rr_c.rfa", "mrc1_fleerun.rfa", "mrc1_flinch_back.rfa",
        "mrc1_flinch_chest.rfa", "mrc1_flinch_leg_l.rfa", "mrc1_flinch_leg_r.rfa", "mrc1_freefall.rfa",
        "mrc1_ft_alt_fire_stand.rfa", "mrc1_idle.rfa", "mrc1_idle2_rr.rfa", "mrc1_reload-ft.rfa",
        "mrc1_reload.rfa", "mrc1_run.rfa", "mrc1_run_flail.rfa", "mrc1_sidestep_left.rfa",
        "mrc1_sidestep_right.rfa", "mrc1_stand-fire-ft.rfa", "mrc1_stand.rfa", "mrc1_walk.rfa",
        "mrc2_crouch_12mm.rfa", "mrc2_crouch_mp.rfa", "mrc2_crouch_rl.rfa", "mrc2_crouch_rs.rfa",
        "mrc2_crouch_rshield.rfa", "mrc2_crouch_sg.rfa", "mrc2_crouch_sr.rfa", "mrc2_crouch_walk_12mm.rfa",
        "mrc2_crouch_walk_ft.rfa", "mrc2_crouch_walk_grn.rfa", "mrc2_crouch_walk_hmac.rfa", "mrc2_crouch_walk_mp.rfa",
        "mrc2_crouch_walk_rl.rfa", "mrc2_crouch_walk_rr.rfa", "mrc2_crouch_walk_rs.rfa", "mrc2_crouch_walk_rshield.rfa",
        "mrc2_crouch_walk_sg.rfa", "mrc2_crouch_walk_smc.rfa", "mrc2_crouch_walk_sr.rfa", "mrc2_fire_12mm.rfa",
        "mrc2_fire_alt_sg.rfa", "mrc2_fire_mp.rfa", "mrc2_fire_rcharge.rfa", "mrc2_fire_rl.rfa",
        "mrc2_fire_rs.rfa", "mrc2_fire_rshield.rfa", "mrc2_fire_sg.rfa", "mrc2_fire_sr.rfa",
        "mrc2_reload_12mm.rfa", "mrc2_reload_mp.rfa", "mrc2_reload_rl.rfa", "mrc2_reload_rs.rfa",
        "mrc2_reload_sg.rfa", "mrc2_reload_sr.rfa", "mrc2_run_12mm.rfa", "mrc2_run_mp.rfa",
        "mrc2_run_rl.rfa", "mrc2_run_rs.rfa", "mrc2_run_rshield.rfa", "mrc2_run_sg.rfa",
        "mrc2_run_sr.rfa", "mrc2_stand_12mm.rfa", "mrc2_stand_mp.rfa", "mrc2_stand_rl.rfa",
        "mrc2_stand_rs.rfa", "mrc2_stand_rshield.rfa", "mrc2_stand_sg.rfa", "mrc2_stand_sr.rfa",
        "mrc2_walk_12mm.rfa", "mrc2_walk_mp.rfa", "mrc2_walk_rl.rfa", "mrc2_walk_rs.rfa",
        "mrc2_walk_rshield.rfa", "mrc2_walk_sg.rfa", "mrc2_walk_sr.rfa", "mrch_attack_crouch.rfa",
        "mrch_attack_crouch_smw.rfa", "mrch_attack_fire.rfa", "mrch_attack_run.rfa", "mrch_attack_stand.rfa",
        "mrch_attack_stand_smw.rfa", "mrch_attack_walk.rfa", "mrch_attack_walk_smw.rfa", "mrch_crouch_fire.rfa",
        "mrch_fire_crouch_smw.rfa", "mrch_fire_smw.rfa", "mrch_flinch_back_smw.rfa", "mrch_flinch_front_smw.rfa",
        "mrch_idle_smw.rfa", "mrch_reload.rfa", "mrch_run_smw.rfa",
    ],
    "mft2.v3c": ["mft2_attack.rfa", "mft2_hover.rfa", "mft2_hover_idle.rfa"],
    "miner.v3c": [
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_sar_attack_crouch.rfa",
        "engd_sar_crouch_fire.rfa", "engd_sar_stand_firing.rfa", "esgd_attack_crouch.rfa", "esgd_attack_run.rfa",
        "esgd_attack_stand.rfa", "esgd_attack_walk.rfa", "esgd_firing_stand.rfa", "esgd_mp_reload.rfa",
        "esgd_stand.rfa", "miner_corpse_carry.rfa", "miner_dead.rfa", "miner_talk.rfa",
        "miner_talk_short.rfa", "park_death_crouch.rfa", "park_death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa",
        "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa", "park_ft_crouch_fire.rfa", "park_ft_reload.rfa",
        "park_ft_run.rfa", "park_ft_stand_fire.rfa", "park_ft_walk.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_hg_crouch_walk.rfa", "park_hmac_attack_crouch.rfa", "park_hmac_attack_stand.rfa",
        "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa", "park_hmac_idle.rfa", "park_hmac_reload.rfa",
        "park_hmac_run.rfa", "park_hmac_stand.rfa", "park_hmac_stand_fire.rfa", "park_hmac_walk.rfa",
        "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_mp_crouch_walk.rfa", "park_remotecharge_throw.rfa",
        "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa", "park_riotstick_reload.rfa",
        "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa", "park_rrif_attack_crouch.rfa",
        "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa", "park_rrif_idle.rfa", "park_rrif_run.rfa",
        "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa", "park_rrif_walk.rfa", "park_rshield_crouch_walk.rfa",
        "park_rstick_crouch_walk.rfa", "park_run.rfa", "park_run_flail.rfa", "park_run_flee_hmac.rfa",
        "park_sc_crouch_walk.rfa", "park_sg_crouch_walk.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa",
        "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa",
        "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa",
        "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa", "park_smw_flinch_back.rfa", "park_smw_flinch_front.rfa",
        "park_smw_idle.rfa", "park_smw_run.rfa", "park_smw_stand.rfa", "park_smw_stand_fire.rfa",
        "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa", "park_sniper_crouch_fire.rfa",
        "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa", "park_sniper_stand_fire.rfa",
        "park_sniper_walk.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa", "rtgd_riotshield_attack.rfa",
        "rtgd_riotshield_attackstand.rfa", "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa",
        "rtgd_riotshield_run.rfa", "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa",
        "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa",
        "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa",
        "ult2_crouch.rfa", "ult2_death_blast_forwards.rfa", "ult2_death_chest_backward.rfa", "ult2_death_chest_forward.rfa",
        "ult2_death_generic.rfa", "ult2_death_head_forwards.rfa", "ult2_death_leg_left.rfa", "ult2_death_leg_right.rfa",
        "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa", "ult2_flee_run.rfa",
        "ult2_flinch_astand.rfa", "ult2_flinch_chestb.rfa", "ult2_flinch_chestf.rfa", "ult2_flinch_legl.rfa",
        "ult2_flinch_legr.rfa", "ult2_flinch_stand.rfa", "ult2_idle_hithead.rfa", "ult2_jump.rfa",
        "ult2_reload.rfa", "ult2_run.rfa", "ult2_run_flee_ar.rfa", "ult2_silencer.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "miner4.v3c": ["mnr4_death.rfa", "mnr4_flinch.rfa", "mnr4_stand.rfa", "mnr4_talk.rfa", "mnr4_talk_short.rfa"],
    "multi_guard2.v3c": [
        "ult3_attack_run.rfa", "ult3_attack_stand.rfa", "ult3_attack_walk.rfa", "ult3_corpse.rfa",
        "ult3_corpse2.rfa", "ult3_corpse3.rfa", "ult3_corpse_drop.rfa", "ult3_cower.rfa",
        "ult3_crouch.rfa", "ult3_death_blast_backwards.rfa", "ult3_death_blast_forwards.rfa", "ult3_death_carry.rfa",
        "ult3_death_chest_backwards.rfa", "ult3_death_chest_forwards.rfa", "ult3_death_crouch.rfa", "ult3_death_generic.rfa",
        "ult3_death_head_backward.rfa", "ult3_death_head_forward.rfa", "ult3_death_leg_left.rfa", "ult3_firing_stand.rfa",
        "ult3_flinch_back.rfa", "ult3_flinch_chest.rfa", "ult3_flinch_leg_left.rfa", "ult3_flinch_leg_right.rfa",
        "ult3_hit_alarm.rfa", "ult3_idle_handstretch.rfa", "ult3_idle_legscratch.rfa", "ult3_idle_radarlook.rfa",
        "ult3_jump.rfa", "ult3_reload.rfa", "ult3_run.rfa", "ult3_run_flail.rfa",
        "ult3_run_flee_ar.rfa", "ult3_sidestep_left.rfa", "ult3_sidestep_right.rfa", "ult3_stand.rfa",
        "ult3_talk.rfa", "ult3_talk_short.rfa", "ult3_walk.rfa",
    ],
    "mutant1.v3c": [
        "mtnt_attack_stand.rfa", "mtnt_corpse_carry.rfa", "mtnt_corpse_drop.rfa", "mtnt_death.rfa",
        "mtnt_flinch_back.rfa", "mtnt_flinch_front.rfa", "mtnt_idle.rfa", "mtnt_move_n_attack.rfa",
        "mtnt_run.rfa", "mtnt_stand.rfa", "mtnt_walk.rfa",
    ],
    "mutant2.v3c": [
        "mnt2_attack.rfa", "mnt2_attack_stand.rfa", "mnt2_corpse_carry.rfa", "mnt2_corpse_drop.rfa",
        "mnt2_death.rfa", "mnt2_flinch_back.rfa", "mnt2_flinch_front.rfa", "mnt2_idle.rfa",
        "mnt2_run.rfa", "mnt2_stand.rfa", "mnt2_walk.rfa",
    ],
    "non_env_miner_fem.v3c": [
        "adfm_cower.rfa", "adfm_crouch.rfa", "adfm_death_blast_forwards.rfa", "adfm_death_chest_backwards.rfa",
        "adfm_death_chest_forwards.rfa", "adfm_death_crouch.rfa", "adfm_death_generic.rfa", "adfm_death_head_backwards.rfa",
        "adfm_death_head_forwards.rfa", "adfm_death_leg_left.rfa", "adfm_death_leg_right.rfa", "adfm_flinch_back.rfa",
        "adfm_flinch_chest.rfa", "adfm_flinch_leg_left.rfa", "adfm_flinch_leg_right.rfa", "adfm_freefall.rfa",
        "adfm_run_flail.rfa", "admin_fem_corpsecarry.rfa", "admin_fem_corpsedrop.rfa", "admin_fem_idle01.rfa",
        "admin_fem_run.rfa", "admin_fem_run_flee.rfa", "admin_fem_stand.rfa", "admin_fem_walk.rfa",
        "eos_sidestep_left.rfa", "eos_sidestep_right.rfa", "mnrf_mp_attack_crouch.rfa", "mnrf_mp_attack_stand.rfa",
        "mnrf_mp_crouch.rfa", "mnrf_mp_crouch_fire.rfa", "mnrf_mp_reload.rfa", "mnrf_mp_run.rfa",
        "mnrf_mp_stand.rfa", "mnrf_mp_stand_fire.rfa", "mnrf_mp_walk.rfa", "mnrf_rs_attack.rfa",
        "mnrf_rs_attack_crouch.rfa", "mnrf_rs_attack_stand.rfa", "mnrf_rs_crouch_attack.rfa", "mnrf_rs_reload.rfa",
        "mnrf_rs_run.rfa", "mnrf_rs_taser.rfa", "mnrf_rs_taser_crouch.rfa", "mnrf_rs_walk.rfa",
        "mnrf_sg_attack_crouch.rfa", "mnrf_sg_attack_stand.rfa", "mnrf_sg_crouch_fire_auto.rfa", "mnrf_sg_crouch_fire_pump.rfa",
        "mnrf_sg_fire_auto.rfa", "mnrf_sg_fire_pump.rfa", "mnrf_sg_reload.rfa", "mnrf_sg_run.rfa",
        "mnrf_sg_stand.rfa", "mnrf_sg_walk.rfa", "mnrf_talk_long.rfa", "mnrf_talk_short.rfa",
        "ult2_run_flee_ar.rfa",
    ],
    "non_env_miner_male.v3c": [
        "esgd_attack_crouch.rfa", "esgd_attack_run.rfa", "esgd_attack_stand.rfa", "esgd_attack_walk.rfa",
        "esgd_firing_stand.rfa", "esgd_mp_reload.rfa", "esgd_stand.rfa", "miner_corpse_carry.rfa",
        "miner_dead.rfa", "mnrm_cower.rfa", "mnrm_death_crouch.rfa", "mnrm_death_generic.rfa",
        "mnrm_talk_long.rfa", "mnrm_talk_short.rfa", "park_death_head_backwards.rfa", "park_jeep_driver.rfa",
        "park_jeep_gunner.rfa", "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa",
        "park_riotstick_reload.rfa", "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa",
        "park_run.rfa", "park_run_flail.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa",
        "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa",
        "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa",
        "park_swim_stand.rfa", "park_swim_walk.rfa", "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa",
        "ult2_attack_stand.rfa", "ult2_attack_walk.rfa", "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa",
        "ult2_corpsecarry_walk.rfa", "ult2_crouch.rfa", "ult2_death_blast_backwards.rfa", "ult2_death_blast_forwards.rfa",
        "ult2_death_chest_backward.rfa", "ult2_death_chest_forward.rfa", "ult2_death_generic.rfa", "ult2_death_head_forwards.rfa",
        "ult2_death_leg_right.rfa", "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa",
        "ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "ult2_flinch_chestb.rfa", "ult2_flinch_chestf.rfa",
        "ult2_flinch_legl.rfa", "ult2_flinch_legr.rfa", "ult2_flinch_stand.rfa", "ult2_idle_look.rfa",
        "ult2_jump.rfa", "ult2_reload.rfa", "ult2_run.rfa", "ult2_run_flee_ar.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa", "ult3_sidestep_left.rfa", "ult3_sidestep_right.rfa",
    ],
    "nurse1.v3c": [
        "adfm_cower.rfa", "adfm_crouch.rfa", "adfm_death_blast_backwards.rfa", "adfm_death_blast_forwards.rfa",
        "adfm_death_chest_backwards.rfa", "adfm_death_chest_forwards.rfa", "adfm_death_crouch.rfa", "adfm_death_generic.rfa",
        "adfm_death_head_backwards.rfa", "adfm_death_head_forwards.rfa", "adfm_death_leg_left.rfa", "adfm_flinch_back.rfa",
        "adfm_flinch_chest.rfa", "adfm_flinch_leg_left.rfa", "adfm_flinch_leg_right.rfa", "adfm_freefall.rfa",
        "adfm_hit_alarm.rfa", "adfm_run_flail.rfa", "admin_fem_corpsecarry.rfa", "admin_fem_corpsedrop.rfa",
        "admin_fem_idle01.rfa", "admin_fem_run.rfa", "admin_fem_run_flee.rfa", "admin_fem_stand.rfa",
        "admin_fem_walk.rfa", "masa_sar_attack_crouch.rfa", "masa_sar_attack_stand.rfa", "masa_sar_crouch_fire.rfa",
        "masa_sar_reload.rfa", "masa_sar_run.rfa", "masa_sar_stand_fire.rfa", "masa_sar_walk.rfa",
        "mnr3f_12mm_crouch.rfa", "mnr3f_12mm_crouch_walk.rfa", "mnr3f_12mm_fire.rfa", "mnr3f_12mm_reload.rfa",
        "mnr3f_12mm_run.rfa", "mnr3f_12mm_stand.rfa", "mnr3f_12mm_walk.rfa", "mnr3f_ar_crouch.rfa",
        "mnr3f_ar_crouch_walk.rfa", "mnr3f_ar_fire.rfa", "mnr3f_ar_reload.rfa", "mnr3f_ar_run.rfa",
        "mnr3f_ar_stand.rfa", "mnr3f_ar_walk.rfa", "mnr3f_ft_crouch.rfa", "mnr3f_ft_crouch_walk.rfa",
        "mnr3f_ft_fire.rfa", "mnr3f_ft_fire_alt.rfa", "mnr3f_ft_reload.rfa", "mnr3f_ft_run.rfa",
        "mnr3f_ft_stand.rfa", "mnr3f_ft_walk.rfa", "mnr3f_grn_fire.rfa", "mnr3f_grn_fire_alt.rfa",
        "mnr3f_hmac_crouch.rfa", "mnr3f_hmac_crouch_walk.rfa", "mnr3f_hmac_fire.rfa", "mnr3f_hmac_reload.rfa",
        "mnr3f_hmac_run.rfa", "mnr3f_hmac_stand.rfa", "mnr3f_hmac_walk.rfa", "mnr3f_rcharge_toss.rfa",
        "mnr3f_rl_crouch.rfa", "mnr3f_rl_crouch_walk.rfa", "mnr3f_rl_fire.rfa", "mnr3f_rl_reload.rfa",
        "mnr3f_rl_run.rfa", "mnr3f_rl_stand.rfa", "mnr3f_rl_walk.rfa", "mnr3f_rr_crouch.rfa",
        "mnr3f_rr_crouch_walk.rfa", "mnr3f_rr_fire.rfa", "mnr3f_rr_run.rfa", "mnr3f_rr_stand.rfa",
        "mnr3f_rr_walk.rfa", "mnr3f_rs_crouch_walk.rfa", "mnr3f_rshield_crouch.rfa", "mnr3f_rshield_crouch_walk.rfa",
        "mnr3f_rshield_fire.rfa", "mnr3f_rshield_run.rfa", "mnr3f_rshield_stand.rfa", "mnr3f_rshield_walk.rfa",
        "mnr3f_sg_crouch_walk.rfa", "mnr3f_smc_crouch.rfa", "mnr3f_smc_crouch_walk.rfa", "mnr3f_smc_fire.rfa",
        "mnr3f_smc_run.rfa", "mnr3f_smc_stand.rfa", "mnr3f_smc_walk.rfa", "mnr3f_sr_crouch.rfa",
        "mnr3f_sr_crouch_walk.rfa", "mnr3f_sr_fire.rfa", "mnr3f_sr_reload.rfa", "mnr3f_sr_run.rfa",
        "mnr3f_sr_stand.rfa", "mnr3f_sr_walk.rfa", "mnr3f_swim_stand.rfa", "mnr3f_swim_walk.rfa",
        "mnrf_mp_attack_crouch.rfa", "mnrf_mp_attack_stand.rfa", "mnrf_mp_crouch_fire.rfa", "mnrf_mp_reload.rfa",
        "mnrf_mp_run.rfa", "mnrf_mp_stand_fire.rfa", "mnrf_mp_walk.rfa", "mnrf_rs_attack.rfa",
        "mnrf_rs_attack_crouch.rfa", "mnrf_rs_attack_stand.rfa", "mnrf_rs_crouch_attack.rfa", "mnrf_rs_reload.rfa",
        "mnrf_rs_run.rfa", "mnrf_rs_taser.rfa", "mnrf_rs_taser_crouch.rfa", "mnrf_rs_walk.rfa",
        "mnrf_sg_attack_crouch.rfa", "mnrf_sg_attack_stand.rfa", "mnrf_sg_crouch_fire_auto.rfa", "mnrf_sg_crouch_fire_pump.rfa",
        "mnrf_sg_fire_auto.rfa", "mnrf_sg_fire_pump.rfa", "mnrf_sg_reload.rfa", "mnrf_sg_run.rfa",
        "mnrf_sg_walk.rfa", "nurs_heal.rfa", "nurs_talk.rfa", "nurs_talk_short.rfa",
    ],
    "parker_sci.v3c": [
        "miner_corpse_carry.rfa", "miner_dead.rfa", "park_death_head_backwards.rfa", "park_hg_crouch_walk.rfa",
        "park_hg_run.rfa", "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_run.rfa",
        "park_run_flail.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa", "prk3_talk.rfa",
        "prk3_talk_short.rfa", "ult2_attack_crouch01.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa",
        "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa",
        "ult2_crouch.rfa", "ult2_crouch_death.rfa", "ult2_death_blast_backwards.rfa", "ult2_death_blast_forwards.rfa",
        "ult2_death_chest_backward.rfa", "ult2_death_chest_forward.rfa", "ult2_death_generic.rfa", "ult2_death_head_forwards.rfa",
        "ult2_death_leg_right.rfa", "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa",
        "ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "ult2_flinch_chestb.rfa", "ult2_flinch_chestf.rfa",
        "ult2_flinch_legl.rfa", "ult2_flinch_legr.rfa", "ult2_flinch_stand.rfa", "ult2_idle_hithead.rfa",
        "ult2_jump.rfa", "ult2_reload.rfa", "ult2_silencer.rfa", "ult2_silencer_off.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "parker_suit.v3c": [
        "miner_corpse_carry.rfa", "miner_dead.rfa", "park_death_head_backwards.rfa", "park_hg_crouch_walk.rfa",
        "park_hg_run.rfa", "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_run.rfa",
        "park_run_flail.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa", "prk2_talk.rfa",
        "prk2_talk_short.rfa", "ult2_attack_crouch01.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa",
        "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa",
        "ult2_crouch.rfa", "ult2_crouch_death.rfa", "ult2_death_blast_forwards.rfa", "ult2_death_chest_backward.rfa",
        "ult2_death_chest_forward.rfa", "ult2_death_generic.rfa", "ult2_death_head_forwards.rfa", "ult2_death_leg_left.rfa",
        "ult2_death_leg_right.rfa", "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa",
        "ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "ult2_flinch_chestb.rfa", "ult2_flinch_chestf.rfa",
        "ult2_flinch_legl.rfa", "ult2_flinch_legr.rfa", "ult2_flinch_stand.rfa", "ult2_idle_hithead.rfa",
        "ult2_jump.rfa", "ult2_reload.rfa", "ult2_silencer.rfa", "ult2_silencer_off.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "powerup_flamecan.v3c": [
        "fp_flame_altfire.rfa", "fp_flame_draw.rfa", "fp_flame_fire.rfa", "fp_flame_holster.rfa",
        "fp_flame_idle.rfa", "fp_flame_idlecheck.rfa", "fp_flame_idletap.rfa", "fp_flame_jump.rfa",
        "fp_flame_reload.rfa", "fp_flame_run.rfa", "fp_flame_switch.rfa",
    ],
    "reeper.v3c": [
        "reep_attack.rfa", "reep_attack_stand.rfa", "reep_death.rfa", "reep_flinch.rfa",
        "reep_idle_scratch.rfa", "reep_run.rfa", "reep_stand.rfa",
    ],
    "riot_guard.v3c": [
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "miner_corpse_carry.rfa",
        "park_death_head_backwards.rfa", "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa",
        "park_riotstick_reload.rfa", "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa",
        "park_run_flail.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa", "park_shotgun_crouchfireauto.rfa",
        "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa", "park_shotgun_stand.rfa",
        "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa", "rtgd_riotshield_attack.rfa",
        "rtgd_riotshield_attackstand.rfa", "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa",
        "rtgd_riotshield_run.rfa", "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa",
        "ult2_attack_walk.rfa", "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa",
        "ult2_cower.rfa", "ult2_crouch.rfa", "ult2_crouch_death.rfa", "ult2_death_blast_backwards.rfa",
        "ult2_death_blast_forwards.rfa", "ult2_death_chest_backward.rfa", "ult2_death_chest_forward.rfa", "ult2_death_generic.rfa",
        "ult2_death_head_forwards.rfa", "ult2_death_leg_right.rfa", "ult2_flee_run.rfa", "ult2_flinch_astand.rfa",
        "ult2_flinch_chestb.rfa", "ult2_flinch_chestf.rfa", "ult2_flinch_legl.rfa", "ult2_flinch_legr.rfa",
        "ult2_flinch_stand.rfa", "ult2_hit_alarm.rfa", "ult2_idle_hithead.rfa", "ult2_jump.rfa",
        "ult2_run.rfa", "ult2_run_flee_ar.rfa", "ult2_sidestep_left.rfa", "ult2_sidestep_right.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "rmt_explosive.v3c": [
        "fp_rmt_chrg_draw.rfa", "fp_rmt_chrg_holster.rfa", "fp_rmt_chrg_idle.rfa", "fp_rmt_chrg_idlerub.rfa",
        "fp_rmt_chrg_idletoss.rfa", "fp_rmt_chrg_jump.rfa", "fp_rmt_chrg_place.rfa", "fp_rmt_chrg_run.rfa",
        "fp_rmt_chrg_throw.rfa",
    ],
    "rock_snake.v3c": [
        "rsnk_attack_bite.rfa", "rsnk_attack_spit.rfa", "rsnk_attack_stand.rfa", "rsnk_attack_walk.rfa",
        "rsnk_death.rfa", "rsnk_stand.rfa", "rsnk_walk.rfa",
    ],
    "sea_creature.v3c": [
        "seac_attack_ram.rfa", "seac_corpse.rfa", "seac_death.rfa", "seac_flinch.rfa",
        "seac_stand.rfa", "seac_swim_fast.rfa", "seac_swim_slow.rfa",
    ],
    "shelltest.vfx": [
        "fp_shol_draw.rfa", "fp_shol_fire.rfa", "fp_shol_holster.rfa", "fp_shol_idle.rfa",
        "fp_shol_idlemonitor.rfa", "fp_shol_idlewipe.rfa", "fp_shol_jump.rfa", "fp_shol_reload.rfa",
        "fp_shol_run.rfa",
    ],
    "tankbot.v3c": [
        "tbot_attack_melee.rfa", "tbot_attack_stand.rfa", "tbot_fire_gun.rfa", "tbot_fire_rocket.rfa",
        "tbot_stand_idle.rfa",
    ],
    "tech01.v3c": [
        "tech01_blast_back.rfa", "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa", "tech01_cower_loop.rfa",
        "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa", "tech01_death_generic.rfa",
        "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa", "tech01_death_leg_l.rfa", "tech01_death_leg_r.rfa",
        "tech01_death_spin_fall_l.rfa", "tech01_death_torso_forward.rfa", "tech01_flinch.rfa", "tech01_flinch_back.rfa",
        "tech01_flinch_leg_l.rfa", "tech01_flinch_leg_r.rfa", "tech01_freefall.rfa", "tech01_hit_alarm.rfa",
        "tech01_idle_01.rfa", "tech01_run.rfa", "tech01_run_flail.rfa", "tech01_run_flee.rfa",
        "tech01_stand.rfa", "tech01_talk.rfa", "tech01_talk_short.rfa", "tech01_walk.rfa",
        "tech_12mm_crouch.rfa", "tech_12mm_crouch_walk.rfa", "tech_12mm_fire.rfa", "tech_12mm_reload.rfa",
        "tech_12mm_run.rfa", "tech_12mm_stand.rfa", "tech_12mm_walk.rfa", "tech_ar_crouch.rfa",
        "tech_ar_crouch_walk.rfa", "tech_ar_fire.rfa", "tech_ar_reload.rfa", "tech_ar_run.rfa",
        "tech_ar_stand.rfa", "tech_ar_walk.rfa", "tech_ft_crouch.rfa", "tech_ft_crouch_walk.rfa",
        "tech_ft_fire.rfa", "tech_ft_fire_alt.rfa", "tech_ft_reload.rfa", "tech_ft_run.rfa",
        "tech_ft_stand.rfa", "tech_ft_walk.rfa", "tech_gren_attack.rfa", "tech_gren_crouch.rfa",
        "tech_gren_crouch_walk.rfa", "tech_gren_run.rfa", "tech_gren_stand.rfa", "tech_gren_throw.rfa",
        "tech_gren_throw_alt.rfa", "tech_gren_walk.rfa", "tech_hmac_crouch.rfa", "tech_hmac_crouch_walk.rfa",
        "tech_hmac_fire.rfa", "tech_hmac_reload.rfa", "tech_hmac_run.rfa", "tech_hmac_stand.rfa",
        "tech_hmac_walk.rfa", "tech_mp_crouch.rfa", "tech_mp_crouch_walk.rfa", "tech_mp_fire.rfa",
        "tech_mp_reload.rfa", "tech_mp_run.rfa", "tech_mp_stand.rfa", "tech_mp_walk.rfa",
        "tech_rc_toss.rfa", "tech_rl_crouch.rfa", "tech_rl_crouch_walk.rfa", "tech_rl_fire.rfa",
        "tech_rl_reload.rfa", "tech_rl_run.rfa", "tech_rl_stand.rfa", "tech_rl_walk.rfa",
        "tech_rr_crouch.rfa", "tech_rr_crouch_walk.rfa", "tech_rr_reload.rfa", "tech_rr_run.rfa",
        "tech_rr_stand.rfa", "tech_rr_walk.rfa", "tech_rs_attack.rfa", "tech_rs_attack_alt.rfa",
        "tech_rs_crouch.rfa", "tech_rs_crouch_walk.rfa", "tech_rs_reload.rfa", "tech_rs_run.rfa",
        "tech_rs_stand.rfa", "tech_rs_walk.rfa", "tech_rshield_crouch.rfa", "tech_rshield_crouch_walk.rfa",
        "tech_rshield_fire.rfa", "tech_rshield_run.rfa", "tech_rshield_stand.rfa", "tech_rshield_walk.rfa",
        "tech_sar_crouch.rfa", "tech_sar_crouch_walk.rfa", "tech_sar_fire.rfa", "tech_sar_reload.rfa",
        "tech_sar_run.rfa", "tech_sar_stand.rfa", "tech_sar_walk.rfa", "tech_sg_crouch.rfa",
        "tech_sg_crouch_walk.rfa", "tech_sg_fire.rfa", "tech_sg_fire_auto.rfa", "tech_sg_reload.rfa",
        "tech_sg_run.rfa", "tech_sg_stand.rfa", "tech_sg_walk.rfa", "tech_smc_crouch.rfa",
        "tech_smc_crouch_walk.rfa", "tech_smc_fire_reload.rfa", "tech_smc_stand.rfa", "tech_smc_walk.rfa",
        "tech_sr_crouch.rfa", "tech_sr_crouch_walk.rfa", "tech_sr_fire.rfa", "tech_sr_reload.rfa",
        "tech_sr_run.rfa", "tech_sr_stand.rfa", "tech_sr_walk.rfa", "tech_swim_stand.rfa",
        "tech_swim_walk.rfa",
    ],
    "transport01.v3c": ["transport_idle.rfa", "transport_takeoff.rfa"],
    "trnsprt_rocket.v3c": ["trnsprt_rockets_fire.rfa", "trnsprt_rockets_idle.rfa"],
    "ult2_guard.v3c": [
        "esgd_attack_crouch.rfa", "esgd_attack_run.rfa", "esgd_attack_stand.rfa", "esgd_attack_walk.rfa",
        "esgd_firing_stand.rfa", "esgd_mp_reload.rfa", "esgd_stand.rfa", "park_riotstick_crouch.rfa",
        "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa", "park_riotstick_reload.rfa", "park_riotstick_stand.rfa",
        "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa", "park_run_flail.rfa", "park_sg_crouch_walk.rfa",
        "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa", "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa",
        "park_shotgun_reload.rfa", "park_shotgun_run.rfa", "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa",
        "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa",
        "park_sniper_crouch_fire.rfa", "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa",
        "park_sniper_stand_fire.rfa", "park_sniper_walk.rfa", "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa",
        "ult2_attack_stand.rfa", "ult2_attack_walk.rfa", "ult2_corpse.rfa", "ult2_corpse2.rfa",
        "ult2_corpse3.rfa", "ult2_corpse_drop.rfa", "ult2_cower.rfa", "ult2_crouch_death.rfa",
        "ult2_death_blast_forwards.rfa", "ult2_death_carry.rfa", "ult2_death_chest_backward.rfa", "ult2_death_chest_forward.rfa",
        "ult2_death_generic.rfa", "ult2_death_head_backwards.rfa", "ult2_death_head_forwards.rfa", "ult2_death_leg_left.rfa",
        "ult2_death_leg_right.rfa", "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa",
        "ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "ult2_flinch_chestb.rfa", "ult2_flinch_chestf.rfa",
        "ult2_flinch_legl.rfa", "ult2_flinch_legr.rfa", "ult2_flinch_stand.rfa", "ult2_hit_alarm.rfa",
        "ult2_idle_look.rfa", "ult2_idle_stretch.rfa", "ult2_jump.rfa", "ult2_on_turret.rfa",
        "ult2_pistol_lean_left.rfa", "ult2_pistol_lean_right.rfa", "ult2_reload.rfa", "ult2_run.rfa",
        "ult2_run_flee_ar.rfa", "ult2_sidestep_left.rfa", "ult2_sidestep_right.rfa", "ult2_stand.rfa",
        "ult2_talk.rfa", "ult2_talk_short.rfa", "ult2_walk.rfa", "ult2_warm_hands.rfa",
    ],
    "ult_scientist.v3c": [
        "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa", "tech01_cower_loop.rfa", "tech01_crouch.rfa",
        "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa", "tech01_death_generic.rfa", "tech01_death_head_back.rfa",
        "tech01_death_head_fore.rfa", "tech01_death_leg_l.rfa", "tech01_death_leg_r.rfa", "tech01_death_spin_fall_l.rfa",
        "tech01_death_torso_forward.rfa", "tech01_flinch.rfa", "tech01_flinch_back.rfa", "tech01_flinch_leg_l.rfa",
        "tech01_flinch_leg_r.rfa", "tech01_freefall.rfa", "tech01_hit_alarm.rfa", "tech01_idle_01.rfa",
        "tech01_run.rfa", "tech01_run_flail.rfa", "tech01_run_flee.rfa", "tech01_stand.rfa",
        "tech01_walk.rfa", "usci_talk.rfa", "usci_talk_short.rfa",
    ],
    "ultorcommand.v3c": [
        "ult3_attack_run.rfa", "ult3_attack_stand.rfa", "ult3_attack_stand_snipe.rfa", "ult3_attack_walk.rfa",
        "ult3_corpse.rfa", "ult3_corpse2.rfa", "ult3_corpse3.rfa", "ult3_corpse_drop.rfa",
        "ult3_cower.rfa", "ult3_crouch.rfa", "ult3_death_blast_backwards.rfa", "ult3_death_blast_forwards.rfa",
        "ult3_death_carry.rfa", "ult3_death_chest_backwards.rfa", "ult3_death_chest_forwards.rfa", "ult3_death_crouch.rfa",
        "ult3_death_generic.rfa", "ult3_death_head_backward.rfa", "ult3_death_head_forward.rfa", "ult3_death_leg_left.rfa",
        "ult3_fire_stand_snipe.rfa", "ult3_firing_stand.rfa", "ult3_flinch_back.rfa", "ult3_flinch_chest.rfa",
        "ult3_flinch_leg_left.rfa", "ult3_flinch_leg_right.rfa", "ult3_hit_alarm.rfa", "ult3_idle_handstretch.rfa",
        "ult3_idle_legscratch.rfa", "ult3_idle_radarlook.rfa", "ult3_jump.rfa", "ult3_reload.rfa",
        "ult3_reload_snipe.rfa", "ult3_run.rfa", "ult3_run_flail.rfa", "ult3_run_flee_ar.rfa",
        "ult3_sidestep_left.rfa", "ult3_sidestep_right.rfa", "ult3_stand.rfa", "ult3_talk.rfa",
        "ult3_talk_short.rfa", "ult3_walk.rfa",
    ],
    "weapon_grenade.v3c": [
        "fp_gren_draw.rfa", "fp_gren_holster.rfa", "fp_gren_idle.rfa", "fp_gren_idlecheck.rfa",
        "fp_gren_jump.rfa", "fp_gren_run.rfa", "fp_gren_spin.rfa", "fp_gren_throw.rfa",
        "fp_gren_twistthrow.rfa",
    ],
}


# ═══ MULTIPLAYER ANIMATION DATABASE ═══
# Generated from pc_multi.tbl + entity.tbl
# Maps V3C filenames to their MP animation sets
_RF_MP_ANIM_DB = {
    "enviro_parker.v3c": [  # MP: enviro_parker / miner1, 150 anims
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_SAR_attack_crouch.rfa",
        "engd_SAR_crouch_fire.rfa", "engd_SAR_stand_firing.rfa", "esgd_attack_crouch.rfa", "esgd_attack_run.rfa",
        "esgd_attack_stand.rfa", "esgd_attack_walk.rfa", "esgd_firing_stand.rfa", "esgd_mp_reload.rfa",
        "esgd_stand.rfa", "miner_corpse_carry.rfa", "miner_dead.rfa", "miner_talk.rfa",
        "miner_talk_short.rfa", "park_Death_crouch.rfa", "park_Death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa",
        "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa", "park_ft_crouch_fire.rfa", "park_ft_reload.rfa",
        "park_ft_run.rfa", "park_ft_stand_fire.rfa", "park_ft_walk.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_HG_crouch_walk.rfa", "park_hmac_attack_crouch.rfa", "park_hmac_attack_stand.rfa",
        "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa", "park_hmac_idle.rfa", "park_hmac_reload.rfa",
        "park_hmac_run.rfa", "park_hmac_stand.rfa", "park_hmac_stand_fire.rfa", "park_hmac_walk.rfa",
        "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_MP_crouch_walk.rfa", "park_remotecharge_throw.rfa",
        "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa", "park_riotstick_reload.rfa",
        "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa", "park_rrif_attack_crouch.rfa",
        "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa", "park_rrif_idle.rfa", "park_rrif_run.rfa",
        "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa", "park_rrif_walk.rfa", "park_Rshield_crouch_walk.rfa",
        "park_Rstick_crouch_walk.rfa", "park_run.rfa", "park_run_flail.rfa", "park_run_flee_HMAC.rfa",
        "park_SC_crouch_walk.rfa", "park_SG_crouch_walk.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa",
        "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa",
        "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa",
        "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa", "park_smw_flinch_back.rfa", "park_smw_flinch_front.rfa",
        "park_smw_idle.rfa", "park_smw_run.rfa", "park_smw_stand.rfa", "park_smw_stand_fire.rfa",
        "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa", "park_sniper_crouch_fire.rfa",
        "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa", "park_sniper_stand_fire.rfa",
        "park_sniper_walk.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa", "rtgd_riotshield_attack.rfa",
        "rtgd_riotshield_attackstand.rfa", "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa",
        "rtgd_riotshield_run.rfa", "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa",
        "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa",
        "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa",
        "ult2_crouch.rfa", "ult2_Death_blast_forwards.rfa", "ult2_Death_chest_backward.rfa", "ult2_Death_chest_forward.rfa",
        "ult2_Death_Generic.rfa", "ult2_Death_head_forwards.rfa", "ult2_Death_leg_Left.rfa", "ult2_Death_leg_Right.rfa",
        "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa", "Ult2_flee_run.rfa",
        "ult2_flinch_astand.rfa", "Ult2_Flinch_ChestB.rfa", "Ult2_Flinch_ChestF.rfa", "Ult2_Flinch_LegL.rfa",
        "Ult2_Flinch_LegR.rfa", "ult2_flinch_stand.rfa", "ult2_idle_HitHead.rfa", "ult2_jump.rfa",
        "ult2_reload.rfa", "ult2_run.rfa", "Ult2_run_flee_AR.rfa", "ult2_silencer.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "parker_sci.v3c": [  # MP: scientist_parker / miner1, 156 anims
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_SAR_attack_crouch.rfa",
        "engd_SAR_crouch_fire.rfa", "engd_SAR_stand_firing.rfa", "esgd_attack_crouch.rfa", "esgd_attack_run.rfa",
        "esgd_attack_stand.rfa", "esgd_attack_walk.rfa", "esgd_firing_stand.rfa", "esgd_mp_reload.rfa",
        "esgd_stand.rfa", "miner_corpse_carry.rfa", "miner_dead.rfa", "miner_talk.rfa",
        "miner_talk_short.rfa", "park_Death_crouch.rfa", "park_Death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa",
        "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa", "park_ft_crouch_fire.rfa", "park_ft_reload.rfa",
        "park_ft_run.rfa", "park_ft_stand_fire.rfa", "park_ft_walk.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_HG_crouch_walk.rfa", "park_hg_run.rfa", "park_hmac_attack_crouch.rfa",
        "park_hmac_attack_stand.rfa", "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa", "park_hmac_idle.rfa",
        "park_hmac_reload.rfa", "park_hmac_run.rfa", "park_hmac_stand.rfa", "park_hmac_stand_fire.rfa",
        "park_hmac_walk.rfa", "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_MP_crouch_walk.rfa",
        "park_remotecharge_throw.rfa", "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa",
        "park_riotstick_reload.rfa", "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa",
        "park_rrif_attack_crouch.rfa", "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa", "park_rrif_idle.rfa",
        "park_rrif_run.rfa", "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa", "park_rrif_walk.rfa",
        "park_Rshield_crouch_walk.rfa", "park_Rstick_crouch_walk.rfa", "park_run.rfa", "park_run_flail.rfa",
        "park_run_flee_HMAC.rfa", "park_SC_crouch_walk.rfa", "park_SG_crouch_walk.rfa", "park_shotgun_attack_stand.rfa",
        "park_shotgun_crouch.rfa", "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa",
        "park_shotgun_run.rfa", "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa",
        "park_shotgun_walk.rfa", "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa", "park_smw_flinch_back.rfa",
        "park_smw_flinch_front.rfa", "park_smw_idle.rfa", "park_smw_run.rfa", "park_smw_stand.rfa",
        "park_smw_stand_fire.rfa", "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa",
        "park_sniper_crouch_fire.rfa", "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa",
        "park_sniper_stand_fire.rfa", "park_sniper_walk.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa",
        "prk3_talk.rfa", "prk3_talk_short.rfa", "rtgd_riotshield_attack.rfa", "rtgd_riotshield_attackstand.rfa",
        "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa", "rtgd_riotshield_run.rfa",
        "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa", "ult2_attack_crouch01.rfa",
        "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa", "ult2_corpse_drop.rfa",
        "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa", "ult2_crouch.rfa",
        "ult2_Crouch_Death.rfa", "ult2_Death_blast_backwards.rfa", "ult2_Death_blast_forwards.rfa", "ult2_Death_chest_backward.rfa",
        "ult2_Death_chest_forward.rfa", "ult2_Death_Generic.rfa", "ult2_Death_head_forwards.rfa", "ult2_Death_leg_Left.rfa",
        "ult2_Death_leg_Right.rfa", "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa",
        "Ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "Ult2_Flinch_ChestB.rfa", "Ult2_Flinch_ChestF.rfa",
        "Ult2_Flinch_LegL.rfa", "Ult2_Flinch_LegR.rfa", "ult2_flinch_stand.rfa", "ult2_idle_HitHead.rfa",
        "ult2_jump.rfa", "ult2_reload.rfa", "ult2_run.rfa", "Ult2_run_flee_AR.rfa",
        "ult2_silencer.rfa", "ult2_silencer_off.rfa", "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "parker_suit.v3c": [  # MP: exec_parker / miner1, 155 anims
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_SAR_attack_crouch.rfa",
        "engd_SAR_crouch_fire.rfa", "engd_SAR_stand_firing.rfa", "esgd_attack_crouch.rfa", "esgd_attack_run.rfa",
        "esgd_attack_stand.rfa", "esgd_attack_walk.rfa", "esgd_firing_stand.rfa", "esgd_mp_reload.rfa",
        "esgd_stand.rfa", "miner_corpse_carry.rfa", "miner_dead.rfa", "miner_talk.rfa",
        "miner_talk_short.rfa", "park_Death_crouch.rfa", "park_Death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa",
        "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa", "park_ft_crouch_fire.rfa", "park_ft_reload.rfa",
        "park_ft_run.rfa", "park_ft_stand_fire.rfa", "park_ft_walk.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_HG_crouch_walk.rfa", "park_hg_run.rfa", "park_hmac_attack_crouch.rfa",
        "park_hmac_attack_stand.rfa", "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa", "park_hmac_idle.rfa",
        "park_hmac_reload.rfa", "park_hmac_run.rfa", "park_hmac_stand.rfa", "park_hmac_stand_fire.rfa",
        "park_hmac_walk.rfa", "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_MP_crouch_walk.rfa",
        "park_remotecharge_throw.rfa", "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa",
        "park_riotstick_reload.rfa", "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa",
        "park_rrif_attack_crouch.rfa", "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa", "park_rrif_idle.rfa",
        "park_rrif_run.rfa", "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa", "park_rrif_walk.rfa",
        "park_Rshield_crouch_walk.rfa", "park_Rstick_crouch_walk.rfa", "park_run.rfa", "park_run_flail.rfa",
        "park_run_flee_HMAC.rfa", "park_SC_crouch_walk.rfa", "park_SG_crouch_walk.rfa", "park_shotgun_attack_stand.rfa",
        "park_shotgun_crouch.rfa", "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa",
        "park_shotgun_run.rfa", "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa",
        "park_shotgun_walk.rfa", "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa", "park_smw_flinch_back.rfa",
        "park_smw_flinch_front.rfa", "park_smw_idle.rfa", "park_smw_run.rfa", "park_smw_stand.rfa",
        "park_smw_stand_fire.rfa", "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa",
        "park_sniper_crouch_fire.rfa", "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa",
        "park_sniper_stand_fire.rfa", "park_sniper_walk.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa",
        "prk2_talk.rfa", "prk2_talk_short.rfa", "rtgd_riotshield_attack.rfa", "rtgd_riotshield_attackstand.rfa",
        "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa", "rtgd_riotshield_run.rfa",
        "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa", "ult2_attack_crouch01.rfa",
        "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa", "ult2_corpse_drop.rfa",
        "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa", "ult2_crouch.rfa",
        "ult2_Crouch_Death.rfa", "ult2_Death_blast_forwards.rfa", "ult2_Death_chest_backward.rfa", "ult2_Death_chest_forward.rfa",
        "ult2_Death_Generic.rfa", "ult2_Death_head_forwards.rfa", "ult2_Death_leg_Left.rfa", "ult2_Death_leg_Right.rfa",
        "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa", "Ult2_flee_run.rfa",
        "ult2_flinch_astand.rfa", "Ult2_Flinch_ChestB.rfa", "Ult2_Flinch_ChestF.rfa", "Ult2_Flinch_LegL.rfa",
        "Ult2_Flinch_LegR.rfa", "ult2_flinch_stand.rfa", "ult2_idle_HitHead.rfa", "ult2_jump.rfa",
        "ult2_reload.rfa", "ult2_run.rfa", "Ult2_run_flee_AR.rfa", "ult2_silencer.rfa",
        "ult2_silencer_off.rfa", "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "guard2.v3c": [  # MP: guard2 / miner1, 189 anims
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_SAR_attack_crouch.rfa",
        "engd_SAR_crouch_fire.rfa", "engd_SAR_stand_firing.rfa", "esgd_attack_crouch.rfa", "esgd_attack_run.rfa",
        "esgd_attack_stand.rfa", "esgd_attack_walk.rfa", "esgd_firing_stand.rfa", "esgd_mp_reload.rfa",
        "esgd_stand.rfa", "miner_corpse_carry.rfa", "miner_dead.rfa", "miner_talk.rfa",
        "miner_talk_short.rfa", "park_Death_crouch.rfa", "park_Death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa",
        "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa", "park_ft_crouch_fire.rfa", "park_ft_reload.rfa",
        "park_ft_run.rfa", "park_ft_stand_fire.rfa", "park_ft_walk.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_HG_crouch_walk.rfa", "park_hmac_attack_crouch.rfa", "park_hmac_attack_stand.rfa",
        "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa", "park_hmac_idle.rfa", "park_hmac_reload.rfa",
        "park_hmac_run.rfa", "park_hmac_stand.rfa", "park_hmac_stand_fire.rfa", "park_hmac_walk.rfa",
        "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_MP_crouch_walk.rfa", "park_remotecharge_throw.rfa",
        "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa", "park_riotstick_reload.rfa",
        "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa", "park_rrif_attack_crouch.rfa",
        "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa", "park_rrif_idle.rfa", "park_rrif_run.rfa",
        "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa", "park_rrif_walk.rfa", "park_Rshield_crouch_walk.rfa",
        "park_Rstick_crouch_walk.rfa", "park_run.rfa", "park_run_flail.rfa", "park_run_flee_HMAC.rfa",
        "park_SC_crouch_walk.rfa", "park_SG_crouch_walk.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa",
        "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa",
        "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa",
        "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa", "park_smw_flinch_back.rfa", "park_smw_flinch_front.rfa",
        "park_smw_idle.rfa", "park_smw_run.rfa", "park_smw_stand.rfa", "park_smw_stand_fire.rfa",
        "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa", "park_sniper_crouch_fire.rfa",
        "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa", "park_sniper_stand_fire.rfa",
        "park_sniper_walk.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa", "rtgd_riotshield_attack.rfa",
        "rtgd_riotshield_attackstand.rfa", "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa",
        "rtgd_riotshield_run.rfa", "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa",
        "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa",
        "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa",
        "ult2_crouch.rfa", "ult2_Death_blast_forwards.rfa", "ult2_Death_chest_backward.rfa", "ult2_Death_chest_forward.rfa",
        "ult2_Death_Generic.rfa", "ult2_Death_head_forwards.rfa", "ult2_Death_leg_Left.rfa", "ult2_Death_leg_Right.rfa",
        "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa", "Ult2_flee_run.rfa",
        "ult2_flinch_astand.rfa", "Ult2_Flinch_ChestB.rfa", "Ult2_Flinch_ChestF.rfa", "Ult2_Flinch_LegL.rfa",
        "Ult2_Flinch_LegR.rfa", "ult2_flinch_stand.rfa", "ult2_idle_HitHead.rfa", "ult2_jump.rfa",
        "ult2_reload.rfa", "ult2_run.rfa", "Ult2_run_flee_AR.rfa", "ult2_silencer.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa", "ult3_attack_run.rfa", "ult3_attack_stand.rfa",
        "ult3_attack_walk.rfa", "ult3_corpse.rfa", "ult3_corpse2.rfa", "ult3_corpse3.rfa",
        "ult3_corpse_drop.rfa", "ult3_cower.rfa", "ult3_crouch.rfa", "ult3_death_blast_backwards.rfa",
        "ult3_death_blast_forwards.rfa", "Ult3_Death_Carry.rfa", "ult3_death_chest_backwards.rfa", "ult3_death_chest_forwards.rfa",
        "ult3_death_crouch.rfa", "ult3_death_generic.rfa", "ult3_death_head_backward.rfa", "ult3_death_head_forward.rfa",
        "ult3_death_leg_left.rfa", "ult3_firing_stand.rfa", "ult3_flinch_back.rfa", "ult3_flinch_chest.rfa",
        "ult3_flinch_leg_left.rfa", "ult3_flinch_leg_right.rfa", "ult3_hit_alarm.rfa", "ult3_idle_handstretch.rfa",
        "ult3_idle_legscratch.rfa", "ult3_idle_radarlook.rfa", "ult3_jump.rfa", "ult3_reload.rfa",
        "ult3_run.rfa", "ult3_run_flail.rfa", "Ult3_run_flee_AR.rfa", "Ult3_sidestep_left.rfa",
        "Ult3_sidestep_right.rfa", "ult3_stand.rfa", "ult3_talk.rfa", "ult3_talk_short.rfa",
        "ult3_walk.rfa",
    ],
    "multi_guard2.v3c": [  # MP: guard2 / miner1, 189 anims
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_SAR_attack_crouch.rfa",
        "engd_SAR_crouch_fire.rfa", "engd_SAR_stand_firing.rfa", "esgd_attack_crouch.rfa", "esgd_attack_run.rfa",
        "esgd_attack_stand.rfa", "esgd_attack_walk.rfa", "esgd_firing_stand.rfa", "esgd_mp_reload.rfa",
        "esgd_stand.rfa", "miner_corpse_carry.rfa", "miner_dead.rfa", "miner_talk.rfa",
        "miner_talk_short.rfa", "park_Death_crouch.rfa", "park_Death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa",
        "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa", "park_ft_crouch_fire.rfa", "park_ft_reload.rfa",
        "park_ft_run.rfa", "park_ft_stand_fire.rfa", "park_ft_walk.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_HG_crouch_walk.rfa", "park_hmac_attack_crouch.rfa", "park_hmac_attack_stand.rfa",
        "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa", "park_hmac_idle.rfa", "park_hmac_reload.rfa",
        "park_hmac_run.rfa", "park_hmac_stand.rfa", "park_hmac_stand_fire.rfa", "park_hmac_walk.rfa",
        "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_MP_crouch_walk.rfa", "park_remotecharge_throw.rfa",
        "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa", "park_riotstick_reload.rfa",
        "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa", "park_rrif_attack_crouch.rfa",
        "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa", "park_rrif_idle.rfa", "park_rrif_run.rfa",
        "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa", "park_rrif_walk.rfa", "park_Rshield_crouch_walk.rfa",
        "park_Rstick_crouch_walk.rfa", "park_run.rfa", "park_run_flail.rfa", "park_run_flee_HMAC.rfa",
        "park_SC_crouch_walk.rfa", "park_SG_crouch_walk.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa",
        "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa",
        "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa",
        "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa", "park_smw_flinch_back.rfa", "park_smw_flinch_front.rfa",
        "park_smw_idle.rfa", "park_smw_run.rfa", "park_smw_stand.rfa", "park_smw_stand_fire.rfa",
        "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa", "park_sniper_crouch_fire.rfa",
        "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa", "park_sniper_stand_fire.rfa",
        "park_sniper_walk.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa", "rtgd_riotshield_attack.rfa",
        "rtgd_riotshield_attackstand.rfa", "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa",
        "rtgd_riotshield_run.rfa", "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa",
        "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa",
        "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa",
        "ult2_crouch.rfa", "ult2_Death_blast_forwards.rfa", "ult2_Death_chest_backward.rfa", "ult2_Death_chest_forward.rfa",
        "ult2_Death_Generic.rfa", "ult2_Death_head_forwards.rfa", "ult2_Death_leg_Left.rfa", "ult2_Death_leg_Right.rfa",
        "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa", "Ult2_flee_run.rfa",
        "ult2_flinch_astand.rfa", "Ult2_Flinch_ChestB.rfa", "Ult2_Flinch_ChestF.rfa", "Ult2_Flinch_LegL.rfa",
        "Ult2_Flinch_LegR.rfa", "ult2_flinch_stand.rfa", "ult2_idle_HitHead.rfa", "ult2_jump.rfa",
        "ult2_reload.rfa", "ult2_run.rfa", "Ult2_run_flee_AR.rfa", "ult2_silencer.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa", "ult3_attack_run.rfa", "ult3_attack_stand.rfa",
        "ult3_attack_walk.rfa", "ult3_corpse.rfa", "ult3_corpse2.rfa", "ult3_corpse3.rfa",
        "ult3_corpse_drop.rfa", "ult3_cower.rfa", "ult3_crouch.rfa", "ult3_death_blast_backwards.rfa",
        "ult3_death_blast_forwards.rfa", "Ult3_Death_Carry.rfa", "ult3_death_chest_backwards.rfa", "ult3_death_chest_forwards.rfa",
        "ult3_death_crouch.rfa", "ult3_death_generic.rfa", "ult3_death_head_backward.rfa", "ult3_death_head_forward.rfa",
        "ult3_death_leg_left.rfa", "ult3_firing_stand.rfa", "ult3_flinch_back.rfa", "ult3_flinch_chest.rfa",
        "ult3_flinch_leg_left.rfa", "ult3_flinch_leg_right.rfa", "ult3_hit_alarm.rfa", "ult3_idle_handstretch.rfa",
        "ult3_idle_legscratch.rfa", "ult3_idle_radarlook.rfa", "ult3_jump.rfa", "ult3_reload.rfa",
        "ult3_run.rfa", "ult3_run_flail.rfa", "Ult3_run_flee_AR.rfa", "Ult3_sidestep_left.rfa",
        "Ult3_sidestep_right.rfa", "ult3_stand.rfa", "ult3_talk.rfa", "ult3_talk_short.rfa",
        "ult3_walk.rfa",
    ],
    "env_guard.v3c": [  # MP: env_guard / miner1, 168 anims
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_SAR_attack_crouch.rfa",
        "engd_SAR_crouch_fire.rfa", "engd_SAR_stand_firing.rfa", "engd_talk.rfa", "engd_talk_short.rfa",
        "esgd_attack_crouch.rfa", "esgd_attack_run.rfa", "esgd_attack_stand.rfa", "esgd_attack_walk.rfa",
        "esgd_firing_stand.rfa", "esgd_mp_reload.rfa", "esgd_stand.rfa", "miner_corpse_carry.rfa",
        "miner_dead.rfa", "miner_talk.rfa", "miner_talk_short.rfa", "park_Death_crouch.rfa",
        "park_Death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa", "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa",
        "park_ft_crouch_fire.rfa", "park_ft_reload.rfa", "park_ft_run.rfa", "park_ft_stand_fire.rfa",
        "park_ft_walk.rfa", "park_grenade_throw.rfa", "park_grenade_throw_alt.rfa", "park_HG_crouch_walk.rfa",
        "park_hmac_attack_crouch.rfa", "park_hmac_attack_stand.rfa", "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa",
        "park_hmac_idle.rfa", "park_hmac_reload.rfa", "park_hmac_run.rfa", "park_hmac_stand.rfa",
        "park_hmac_stand_fire.rfa", "park_hmac_walk.rfa", "park_jeep_driver.rfa", "park_jeep_gunner.rfa",
        "park_MP_crouch_walk.rfa", "park_remotecharge_throw.rfa", "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa",
        "park_riotstick_crouch_taser.rfa", "park_riotstick_reload.rfa", "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa",
        "park_riotstick_taser.rfa", "park_rrif_attack_crouch.rfa", "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa",
        "park_rrif_idle.rfa", "park_rrif_run.rfa", "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa",
        "park_rrif_walk.rfa", "park_Rshield_crouch_walk.rfa", "park_Rstick_crouch_walk.rfa", "park_run.rfa",
        "park_run_flail.rfa", "park_run_flee_HMAC.rfa", "park_SC_crouch_walk.rfa", "park_SG_crouch_walk.rfa",
        "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa", "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa",
        "park_shotgun_reload.rfa", "park_shotgun_run.rfa", "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa",
        "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa", "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa",
        "park_smw_flinch_back.rfa", "park_smw_flinch_front.rfa", "park_smw_idle.rfa", "park_smw_run.rfa",
        "park_smw_stand.rfa", "park_smw_stand_fire.rfa", "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa",
        "park_sniper_attack_stand.rfa", "park_sniper_crouch_fire.rfa", "park_sniper_reload.rfa", "park_sniper_run.rfa",
        "park_sniper_stand.rfa", "park_sniper_stand_fire.rfa", "park_sniper_walk.rfa", "park_swim_stand.rfa",
        "park_swim_walk.rfa", "rtgd_riotshield_attack.rfa", "rtgd_riotshield_attackstand.rfa", "rtgd_riotshield_crouch.rfa",
        "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa", "rtgd_riotshield_run.rfa", "rtgd_riotshield_stand.rfa",
        "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa", "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa",
        "ult2_attack_stand.rfa", "ult2_attack_walk.rfa", "ult2_corpse.rfa", "ult2_corpse2.rfa",
        "ult2_corpse3.rfa", "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa",
        "ult2_cower.rfa", "ult2_crouch.rfa", "ult2_Crouch_Death.rfa", "ult2_Death_blast_forwards.rfa",
        "Ult2_Death_Carry.rfa", "ult2_Death_chest_backward.rfa", "ult2_Death_chest_forward.rfa", "ult2_Death_Generic.rfa",
        "ult2_Death_head_backwards.rfa", "ult2_Death_head_forwards.rfa", "ult2_Death_leg_Left.rfa", "ult2_Death_leg_Right.rfa",
        "ult2_draw.rfa", "ult2_firing_crouch.rfa", "ult2_firing_stand.rfa", "Ult2_flee_run.rfa",
        "ult2_flinch_astand.rfa", "Ult2_Flinch_ChestB.rfa", "ult2_Flinch_ChestB.rfa", "ult2_Flinch_ChestF.rfa",
        "Ult2_Flinch_ChestF.rfa", "Ult2_Flinch_LegL.rfa", "ult2_Flinch_LegL.rfa", "ult2_Flinch_LegR.rfa",
        "Ult2_Flinch_LegR.rfa", "ult2_flinch_stand.rfa", "ult2_hit_alarm.rfa", "ult2_idle_HitHead.rfa",
        "ult2_idle_look.rfa", "ult2_idle_stretch.rfa", "ult2_jump.rfa", "ult2_on_turret.rfa",
        "ult2_reload.rfa", "ult2_run.rfa", "Ult2_run_flee_AR.rfa", "ult2_sidestep_left.rfa",
        "ult2_sidestep_right.rfa", "ult2_silencer.rfa", "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "elite.v3c": [  # MP: elite / miner1, 170 anims
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_SAR_attack_crouch.rfa",
        "engd_SAR_crouch_fire.rfa", "engd_SAR_stand_firing.rfa", "esgd_attack_crouch.rfa", "esgd_attack_roll_left.rfa",
        "esgd_attack_roll_right.rfa", "esgd_attack_run.rfa", "esgd_attack_stand.rfa", "esgd_attack_walk.rfa",
        "esgd_firing_stand.rfa", "esgd_mp_reload.rfa", "esgd_stand.rfa", "esgd_talk.rfa",
        "esgd_talk_short.rfa", "miner_corpse_carry.rfa", "miner_dead.rfa", "miner_talk.rfa",
        "miner_talk_short.rfa", "park_Death_crouch.rfa", "park_Death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa",
        "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa", "park_ft_crouch_fire.rfa", "park_ft_reload.rfa",
        "park_ft_run.rfa", "park_ft_stand_fire.rfa", "park_ft_walk.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_HG_crouch_walk.rfa", "park_hmac_attack_crouch.rfa", "park_hmac_attack_stand.rfa",
        "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa", "park_hmac_idle.rfa", "park_hmac_reload.rfa",
        "park_hmac_run.rfa", "park_hmac_stand.rfa", "park_hmac_stand_fire.rfa", "park_hmac_walk.rfa",
        "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_MP_crouch_walk.rfa", "park_remotecharge_throw.rfa",
        "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa", "park_riotstick_reload.rfa",
        "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa", "park_rrif_attack_crouch.rfa",
        "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa", "park_rrif_idle.rfa", "park_rrif_run.rfa",
        "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa", "park_rrif_walk.rfa", "park_Rshield_crouch_walk.rfa",
        "park_Rstick_crouch_walk.rfa", "park_run.rfa", "park_run_flail.rfa", "park_run_flee_HMAC.rfa",
        "park_SC_crouch_walk.rfa", "park_SG_crouch_walk.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa",
        "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa",
        "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa",
        "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa", "park_smw_flinch_back.rfa", "park_smw_flinch_front.rfa",
        "park_smw_idle.rfa", "park_smw_run.rfa", "park_smw_stand.rfa", "park_smw_stand_fire.rfa",
        "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa", "park_sniper_crouch_fire.rfa",
        "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa", "park_sniper_stand_fire.rfa",
        "park_sniper_walk.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa", "rtgd_riotshield_attack.rfa",
        "rtgd_riotshield_attackstand.rfa", "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa",
        "rtgd_riotshield_run.rfa", "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa",
        "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa",
        "ult2_corpse.rfa", "ult2_corpse2.rfa", "ult2_corpse3.rfa", "ult2_corpse_drop.rfa",
        "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa", "ult2_crouch.rfa",
        "ult2_Crouch_Death.rfa", "ult2_Death_blast_forwards.rfa", "Ult2_Death_Carry.rfa", "ult2_Death_chest_backward.rfa",
        "ult2_Death_chest_forward.rfa", "ult2_Death_Generic.rfa", "ult2_Death_head_backwards.rfa", "ult2_Death_head_forwards.rfa",
        "ult2_Death_leg_Left.rfa", "ult2_Death_leg_Right.rfa", "ult2_draw.rfa", "ult2_firing_crouch.rfa",
        "ult2_firing_stand.rfa", "Ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "Ult2_Flinch_ChestB.rfa",
        "ult2_Flinch_ChestB.rfa", "ult2_Flinch_ChestF.rfa", "Ult2_Flinch_ChestF.rfa", "Ult2_Flinch_LegL.rfa",
        "ult2_Flinch_LegL.rfa", "ult2_Flinch_LegR.rfa", "Ult2_Flinch_LegR.rfa", "ult2_flinch_stand.rfa",
        "ult2_hit_alarm.rfa", "ult2_idle_HitHead.rfa", "ult2_idle_look.rfa", "ult2_idle_stretch.rfa",
        "ult2_jump.rfa", "ult2_on_turret.rfa", "ult2_reload.rfa", "ult2_run.rfa",
        "Ult2_run_flee_AR.rfa", "ult2_sidestep_left.rfa", "ult2_sidestep_right.rfa", "ult2_silencer.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "ult2_guard.v3c": [  # MP: elite / miner1, 170 anims
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_SAR_attack_crouch.rfa",
        "engd_SAR_crouch_fire.rfa", "engd_SAR_stand_firing.rfa", "esgd_attack_crouch.rfa", "esgd_attack_roll_left.rfa",
        "esgd_attack_roll_right.rfa", "esgd_attack_run.rfa", "esgd_attack_stand.rfa", "esgd_attack_walk.rfa",
        "esgd_firing_stand.rfa", "esgd_mp_reload.rfa", "esgd_stand.rfa", "esgd_talk.rfa",
        "esgd_talk_short.rfa", "miner_corpse_carry.rfa", "miner_dead.rfa", "miner_talk.rfa",
        "miner_talk_short.rfa", "park_Death_crouch.rfa", "park_Death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa",
        "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa", "park_ft_crouch_fire.rfa", "park_ft_reload.rfa",
        "park_ft_run.rfa", "park_ft_stand_fire.rfa", "park_ft_walk.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_HG_crouch_walk.rfa", "park_hmac_attack_crouch.rfa", "park_hmac_attack_stand.rfa",
        "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa", "park_hmac_idle.rfa", "park_hmac_reload.rfa",
        "park_hmac_run.rfa", "park_hmac_stand.rfa", "park_hmac_stand_fire.rfa", "park_hmac_walk.rfa",
        "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_MP_crouch_walk.rfa", "park_remotecharge_throw.rfa",
        "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa", "park_riotstick_reload.rfa",
        "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa", "park_rrif_attack_crouch.rfa",
        "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa", "park_rrif_idle.rfa", "park_rrif_run.rfa",
        "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa", "park_rrif_walk.rfa", "park_Rshield_crouch_walk.rfa",
        "park_Rstick_crouch_walk.rfa", "park_run.rfa", "park_run_flail.rfa", "park_run_flee_HMAC.rfa",
        "park_SC_crouch_walk.rfa", "park_SG_crouch_walk.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa",
        "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa",
        "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa",
        "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa", "park_smw_flinch_back.rfa", "park_smw_flinch_front.rfa",
        "park_smw_idle.rfa", "park_smw_run.rfa", "park_smw_stand.rfa", "park_smw_stand_fire.rfa",
        "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa", "park_sniper_crouch_fire.rfa",
        "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa", "park_sniper_stand_fire.rfa",
        "park_sniper_walk.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa", "rtgd_riotshield_attack.rfa",
        "rtgd_riotshield_attackstand.rfa", "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa",
        "rtgd_riotshield_run.rfa", "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa",
        "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa",
        "ult2_corpse.rfa", "ult2_corpse2.rfa", "ult2_corpse3.rfa", "ult2_corpse_drop.rfa",
        "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa", "ult2_crouch.rfa",
        "ult2_Crouch_Death.rfa", "ult2_Death_blast_forwards.rfa", "Ult2_Death_Carry.rfa", "ult2_Death_chest_backward.rfa",
        "ult2_Death_chest_forward.rfa", "ult2_Death_Generic.rfa", "ult2_Death_head_backwards.rfa", "ult2_Death_head_forwards.rfa",
        "ult2_Death_leg_Left.rfa", "ult2_Death_leg_Right.rfa", "ult2_draw.rfa", "ult2_firing_crouch.rfa",
        "ult2_firing_stand.rfa", "Ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "Ult2_Flinch_ChestB.rfa",
        "ult2_Flinch_ChestB.rfa", "ult2_Flinch_ChestF.rfa", "Ult2_Flinch_ChestF.rfa", "Ult2_Flinch_LegL.rfa",
        "ult2_Flinch_LegL.rfa", "ult2_Flinch_LegR.rfa", "Ult2_Flinch_LegR.rfa", "ult2_flinch_stand.rfa",
        "ult2_hit_alarm.rfa", "ult2_idle_HitHead.rfa", "ult2_idle_look.rfa", "ult2_idle_stretch.rfa",
        "ult2_jump.rfa", "ult2_on_turret.rfa", "ult2_reload.rfa", "ult2_run.rfa",
        "Ult2_run_flee_AR.rfa", "ult2_sidestep_left.rfa", "ult2_sidestep_right.rfa", "ult2_silencer.rfa",
        "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "riot_guard.v3c": [  # MP: riot_guard / miner1, 155 anims
        "engd_arifle_attack_crouch.rfa", "engd_arifle_attack_run.rfa", "engd_arifle_attack_stand.rfa", "engd_arifle_attack_walk.rfa",
        "engd_arifle_reload.rfa", "engd_arifle_stand.rfa", "engd_arifle_stand_firing.rfa", "engd_arifle_stand_firing_2.rfa",
        "engd_rl_attack_crouch.rfa", "engd_rl_attack_stand.rfa", "engd_rl_crouch_firing.rfa", "engd_rl_run.rfa",
        "engd_rl_stand.rfa", "engd_rl_stand_firing.rfa", "engd_rl_walk.rfa", "engd_SAR_attack_crouch.rfa",
        "engd_SAR_crouch_fire.rfa", "engd_SAR_stand_firing.rfa", "esgd_attack_crouch.rfa", "esgd_attack_run.rfa",
        "esgd_attack_stand.rfa", "esgd_attack_walk.rfa", "esgd_firing_stand.rfa", "esgd_mp_reload.rfa",
        "esgd_stand.rfa", "miner_corpse_carry.rfa", "miner_dead.rfa", "miner_talk.rfa",
        "miner_talk_short.rfa", "park_Death_crouch.rfa", "park_Death_head_backwards.rfa", "park_ft_alt_fire_stand.rfa",
        "park_ft_attack_crouch.rfa", "park_ft_attack_stand.rfa", "park_ft_crouch_fire.rfa", "park_ft_reload.rfa",
        "park_ft_run.rfa", "park_ft_stand_fire.rfa", "park_ft_walk.rfa", "park_grenade_throw.rfa",
        "park_grenade_throw_alt.rfa", "park_HG_crouch_walk.rfa", "park_hmac_attack_crouch.rfa", "park_hmac_attack_stand.rfa",
        "park_hmac_crouch_fire.rfa", "park_hmac_crouch_walk.rfa", "park_hmac_idle.rfa", "park_hmac_reload.rfa",
        "park_hmac_run.rfa", "park_hmac_stand.rfa", "park_hmac_stand_fire.rfa", "park_hmac_walk.rfa",
        "park_jeep_driver.rfa", "park_jeep_gunner.rfa", "park_MP_crouch_walk.rfa", "park_remotecharge_throw.rfa",
        "park_riotstick_crouch.rfa", "park_riotstick_crouch_swing.rfa", "park_riotstick_crouch_taser.rfa", "park_riotstick_reload.rfa",
        "park_riotstick_stand.rfa", "park_riotstick_swing1.rfa", "park_riotstick_taser.rfa", "park_rrif_attack_crouch.rfa",
        "park_rrif_attack_stand.rfa", "park_rrif_crouch_fire.rfa", "park_rrif_idle.rfa", "park_rrif_run.rfa",
        "park_rrif_stand.rfa", "park_rrif_stand_fire.rfa", "park_rrif_walk.rfa", "park_Rshield_crouch_walk.rfa",
        "park_Rstick_crouch_walk.rfa", "park_run.rfa", "park_run_flail.rfa", "park_run_flee_HMAC.rfa",
        "park_SC_crouch_walk.rfa", "park_SG_crouch_walk.rfa", "park_shotgun_attack_stand.rfa", "park_shotgun_crouch.rfa",
        "park_shotgun_crouchfireauto.rfa", "park_shotgun_crouchfirepump.rfa", "park_shotgun_reload.rfa", "park_shotgun_run.rfa",
        "park_shotgun_stand.rfa", "park_shotgun_stand_fireauto.rfa", "park_shotgun_stand_firepump.rfa", "park_shotgun_walk.rfa",
        "park_smw_crouch.rfa", "park_smw_crouch_fire.rfa", "park_smw_flinch_back.rfa", "park_smw_flinch_front.rfa",
        "park_smw_idle.rfa", "park_smw_run.rfa", "park_smw_stand.rfa", "park_smw_stand_fire.rfa",
        "park_smw_walk.rfa", "park_sniper_attack_crouch.rfa", "park_sniper_attack_stand.rfa", "park_sniper_crouch_fire.rfa",
        "park_sniper_reload.rfa", "park_sniper_run.rfa", "park_sniper_stand.rfa", "park_sniper_stand_fire.rfa",
        "park_sniper_walk.rfa", "park_swim_stand.rfa", "park_swim_walk.rfa", "rtgd_riotshield_attack.rfa",
        "rtgd_riotshield_attackstand.rfa", "rtgd_riotshield_crouch.rfa", "rtgd_riotshield_flinch_back.rfa", "rtgd_riotshield_flinchfront.rfa",
        "rtgd_riotshield_run.rfa", "rtgd_riotshield_stand.rfa", "rtgd_riotshield_walk.rfa", "rtgd_riotshieldattackcrouch.rfa",
        "ult2_attack_crouch01.rfa", "ult2_attack_run.rfa", "ult2_attack_stand.rfa", "ult2_attack_walk.rfa",
        "ult2_corpse_drop.rfa", "ult2_corpsecarry_stand.rfa", "ult2_corpsecarry_walk.rfa", "ult2_cower.rfa",
        "ult2_crouch.rfa", "ult2_Crouch_Death.rfa", "ult2_Death_blast_backwards.rfa", "ult2_Death_blast_forwards.rfa",
        "ult2_Death_chest_backward.rfa", "ult2_Death_chest_forward.rfa", "ult2_Death_Generic.rfa", "ult2_Death_head_forwards.rfa",
        "ult2_Death_leg_Left.rfa", "ult2_Death_leg_Right.rfa", "ult2_draw.rfa", "ult2_firing_crouch.rfa",
        "ult2_firing_stand.rfa", "Ult2_flee_run.rfa", "ult2_flinch_astand.rfa", "Ult2_Flinch_ChestB.rfa",
        "Ult2_Flinch_ChestF.rfa", "Ult2_Flinch_LegL.rfa", "Ult2_Flinch_LegR.rfa", "ult2_flinch_stand.rfa",
        "ult2_hit_alarm.rfa", "ult2_idle_HitHead.rfa", "ult2_jump.rfa", "ult2_reload.rfa",
        "ult2_run.rfa", "Ult2_run_flee_AR.rfa", "ult2_sidestep_left.rfa", "ult2_sidestep_right.rfa",
        "ult2_silencer.rfa", "ult2_stand.rfa", "ult2_walk.rfa",
    ],
    "nurse1.v3c": [  # MP: nurse / multi_female, 129 anims
        "ADFM_cower.rfa", "ADFM_crouch.rfa", "ADFM_death_blast_backwards.rfa", "ADFM_death_blast_forwards.rfa",
        "ADFM_death_chest_backwards.rfa", "ADFM_death_chest_forwards.rfa", "ADFM_death_crouch.rfa", "ADFM_death_generic.rfa",
        "ADFM_death_head_backwards.rfa", "ADFM_death_head_forwards.rfa", "ADFM_death_leg_left.rfa", "ADFM_death_leg_right.rfa",
        "ADFM_flinch_back.rfa", "ADFM_flinch_chest.rfa", "ADFM_flinch_leg_left.rfa", "ADFM_flinch_leg_right.rfa",
        "ADFM_freefall.rfa", "ADFM_hit_alarm.rfa", "adfm_run_flail.rfa", "admin_fem_corpsecarry.rfa",
        "admin_fem_corpsedrop.rfa", "admin_fem_idle01.rfa", "admin_fem_run.rfa", "admin_fem_run_flee.rfa",
        "admin_fem_stand.rfa", "admin_fem_walk.rfa", "masa_sar_attack_crouch.rfa", "masa_sar_attack_stand.rfa",
        "masa_sar_crouch_fire.rfa", "masa_sar_reload.rfa", "masa_sar_run.rfa", "masa_sar_stand_fire.rfa",
        "masa_sar_walk.rfa", "mnr3f_12mm_crouch.rfa", "mnr3f_12mm_crouch_walk.rfa", "mnr3f_12mm_fire.rfa",
        "mnr3f_12mm_reload.rfa", "mnr3f_12mm_run.rfa", "mnr3f_12mm_stand.rfa", "mnr3f_12mm_walk.rfa",
        "mnr3f_AR_crouch.rfa", "mnr3f_AR_crouch_walk.rfa", "mnr3f_AR_fire.rfa", "mnr3f_AR_reload.rfa",
        "mnr3f_AR_run.rfa", "mnr3f_AR_stand.rfa", "mnr3f_AR_walk.rfa", "mnr3f_FT_crouch.rfa",
        "mnr3f_FT_crouch_walk.rfa", "mnr3f_FT_fire.rfa", "mnr3f_FT_fire_ALT.rfa", "mnr3f_FT_reload.rfa",
        "mnr3f_FT_run.rfa", "mnr3f_FT_stand.rfa", "mnr3f_FT_walk.rfa", "mnr3f_GRN_fire.rfa",
        "mnr3f_GRN_fire_ALT.rfa", "mnr3f_Hmac_crouch.rfa", "mnr3f_Hmac_crouch_walk.rfa", "mnr3f_Hmac_fire.rfa",
        "mnr3f_Hmac_reload.rfa", "mnr3f_Hmac_run.rfa", "mnr3f_Hmac_stand.rfa", "mnr3f_Hmac_walk.rfa",
        "mnr3f_Rcharge_toss.rfa", "mnr3f_RL_crouch.rfa", "mnr3f_RL_crouch_walk.rfa", "mnr3f_RL_fire.rfa",
        "mnr3f_RL_reload.rfa", "mnr3f_RL_run.rfa", "mnr3f_RL_stand.rfa", "mnr3f_RL_walk.rfa",
        "mnr3f_RR_crouch.rfa", "mnr3f_RR_crouch_walk.rfa", "mnr3f_RR_fire.rfa", "mnr3f_RR_run.rfa",
        "mnr3f_RR_stand.rfa", "mnr3f_RR_walk.rfa", "mnr3f_rs_crouch_walk.rfa", "mnr3f_Rshield_crouch.rfa",
        "mnr3f_Rshield_crouch_walk.rfa", "mnr3f_Rshield_fire.rfa", "mnr3f_Rshield_run.rfa", "mnr3f_Rshield_stand.rfa",
        "mnr3f_Rshield_walk.rfa", "mnr3f_SG_crouch_walk.rfa", "mnr3f_SMC_crouch.rfa", "mnr3f_SMC_crouch_walk.rfa",
        "mnr3f_SMC_fire.rfa", "mnr3f_SMC_run.rfa", "mnr3f_SMC_stand.rfa", "mnr3f_SMC_walk.rfa",
        "mnr3f_SR_crouch.rfa", "mnr3f_SR_crouch_walk.rfa", "mnr3f_SR_fire.rfa", "mnr3f_SR_reload.rfa",
        "mnr3f_SR_run.rfa", "mnr3f_SR_stand.rfa", "mnr3f_SR_walk.rfa", "mnr3f_swim_stand.rfa",
        "mnr3f_swim_walk.rfa", "mnrf_mp_attack_crouch.rfa", "mnrf_mp_attack_stand.rfa", "mnrf_mp_crouch_fire.rfa",
        "mnrf_mp_reload.rfa", "mnrf_mp_run.rfa", "mnrf_mp_stand_fire.rfa", "mnrf_mp_walk.rfa",
        "mnrf_rs_attack.rfa", "mnrf_rs_attack_crouch.rfa", "mnrf_rs_attack_stand.rfa", "mnrf_rs_crouch_attack.rfa",
        "mnrf_rs_reload.rfa", "mnrf_rs_run.rfa", "mnrf_rs_taser.rfa", "mnrf_rs_taser_crouch.rfa",
        "mnrf_rs_walk.rfa", "mnrf_sg_attack_crouch.rfa", "mnrf_sg_attack_stand.rfa", "mnrf_sg_crouch_fire_auto.rfa",
        "mnrf_sg_crouch_fire_pump.rfa", "mnrf_sg_fire_auto.rfa", "mnrf_sg_fire_pump.rfa", "mnrf_sg_reload.rfa",
        "mnrf_sg_run.rfa", "mnrf_sg_walk.rfa", "NURS_heal.rfa", "NURS_talk.rfa",
        "NURS_talk_short.rfa",
    ],
    "miner3.v3c": [  # MP: female_miner / multi_female, 133 anims
        "ADFM_cower.rfa", "ADFM_crouch.rfa", "ADFM_death_blast_backwards.rfa", "ADFM_death_blast_forwards.rfa",
        "ADFM_death_chest_backwards.rfa", "ADFM_death_chest_forwards.rfa", "ADFM_death_crouch.rfa", "ADFM_death_generic.rfa",
        "ADFM_death_head_backwards.rfa", "ADFM_death_head_forwards.rfa", "ADFM_death_leg_left.rfa", "ADFM_death_leg_right.rfa",
        "ADFM_flinch_back.rfa", "ADFM_flinch_chest.rfa", "ADFM_flinch_leg_left.rfa", "ADFM_flinch_leg_right.rfa",
        "ADFM_freefall.rfa", "adfm_run_flail.rfa", "admin_fem_corpsecarry.rfa", "admin_fem_corpsedrop.rfa",
        "admin_fem_idle01.rfa", "admin_fem_run.rfa", "admin_fem_run_flee.rfa", "admin_fem_stand.rfa",
        "admin_fem_walk.rfa", "eos_sidestep_left.rfa", "eos_sidestep_right.rfa", "masa_sar_attack_crouch.rfa",
        "masa_sar_attack_stand.rfa", "masa_sar_crouch_fire.rfa", "masa_sar_reload.rfa", "masa_sar_run.rfa",
        "masa_sar_stand_fire.rfa", "masa_sar_walk.rfa", "mnr3f_12mm_crouch.rfa", "mnr3f_12mm_crouch_walk.rfa",
        "mnr3f_12mm_fire.rfa", "mnr3f_12mm_reload.rfa", "mnr3f_12mm_run.rfa", "mnr3f_12mm_stand.rfa",
        "mnr3f_12mm_walk.rfa", "mnr3f_AR_crouch.rfa", "mnr3f_AR_crouch_walk.rfa", "mnr3f_AR_fire.rfa",
        "mnr3f_AR_reload.rfa", "mnr3f_AR_run.rfa", "mnr3f_AR_stand.rfa", "mnr3f_AR_walk.rfa",
        "mnr3f_FT_crouch.rfa", "mnr3f_FT_crouch_walk.rfa", "mnr3f_FT_fire.rfa", "mnr3f_FT_fire_ALT.rfa",
        "mnr3f_FT_reload.rfa", "mnr3f_FT_run.rfa", "mnr3f_FT_stand.rfa", "mnr3f_FT_walk.rfa",
        "mnr3f_GRN_fire.rfa", "mnr3f_GRN_fire_ALT.rfa", "mnr3f_Hmac_crouch.rfa", "mnr3f_Hmac_crouch_walk.rfa",
        "mnr3f_Hmac_fire.rfa", "mnr3f_Hmac_reload.rfa", "mnr3f_Hmac_run.rfa", "mnr3f_Hmac_stand.rfa",
        "mnr3f_Hmac_walk.rfa", "mnr3f_Rcharge_toss.rfa", "mnr3f_RL_crouch.rfa", "mnr3f_RL_crouch_walk.rfa",
        "mnr3f_RL_fire.rfa", "mnr3f_RL_reload.rfa", "mnr3f_RL_run.rfa", "mnr3f_RL_stand.rfa",
        "mnr3f_RL_walk.rfa", "mnr3f_RR_crouch.rfa", "mnr3f_RR_crouch_walk.rfa", "mnr3f_RR_fire.rfa",
        "mnr3f_RR_run.rfa", "mnr3f_RR_stand.rfa", "mnr3f_RR_walk.rfa", "mnr3f_rs_crouch_walk.rfa",
        "mnr3f_Rshield_crouch.rfa", "mnr3f_Rshield_crouch_walk.rfa", "mnr3f_Rshield_fire.rfa", "mnr3f_Rshield_run.rfa",
        "mnr3f_Rshield_stand.rfa", "mnr3f_Rshield_walk.rfa", "mnr3f_SG_crouch_walk.rfa", "mnr3f_SMC_crouch.rfa",
        "mnr3f_SMC_crouch_walk.rfa", "mnr3f_SMC_fire.rfa", "mnr3f_SMC_run.rfa", "mnr3f_SMC_stand.rfa",
        "mnr3f_SMC_walk.rfa", "mnr3f_SR_crouch.rfa", "mnr3f_SR_crouch_walk.rfa", "mnr3f_SR_fire.rfa",
        "mnr3f_SR_reload.rfa", "mnr3f_SR_run.rfa", "mnr3f_SR_stand.rfa", "mnr3f_SR_walk.rfa",
        "mnr3f_swim_stand.rfa", "mnr3f_swim_walk.rfa", "mnrf_mp_attack_crouch.rfa", "mnrf_mp_attack_stand.rfa",
        "mnrf_mp_crouch.rfa", "mnrf_mp_crouch_fire.rfa", "mnrf_mp_reload.rfa", "mnrf_mp_run.rfa",
        "mnrf_mp_stand.rfa", "mnrf_mp_stand_fire.rfa", "mnrf_mp_walk.rfa", "mnrf_rs_attack.rfa",
        "mnrf_rs_attack_crouch.rfa", "mnrf_rs_attack_stand.rfa", "mnrf_rs_crouch_attack.rfa", "mnrf_rs_reload.rfa",
        "mnrf_rs_run.rfa", "mnrf_rs_taser.rfa", "mnrf_rs_taser_crouch.rfa", "mnrf_rs_walk.rfa",
        "mnrf_sg_attack_crouch.rfa", "mnrf_sg_attack_stand.rfa", "mnrf_sg_crouch_fire_auto.rfa", "mnrf_sg_crouch_fire_pump.rfa",
        "mnrf_sg_fire_auto.rfa", "mnrf_sg_fire_pump.rfa", "mnrf_sg_reload.rfa", "mnrf_sg_run.rfa",
        "mnrf_sg_stand.rfa", "mnrf_sg_walk.rfa", "mnrf_talk_long.rfa", "mnrf_talk_short.rfa",
        "Ult2_run_flee_AR.rfa",
    ],
    "admin_fem.v3c": [  # MP: female_admin / multi_female, 128 anims
        "ADFM_cower.rfa", "ADFM_crouch.rfa", "ADFM_death_blast_backwards.rfa", "ADFM_death_blast_forwards.rfa",
        "ADFM_death_chest_backwards.rfa", "ADFM_death_chest_forwards.rfa", "ADFM_death_crouch.rfa", "ADFM_death_generic.rfa",
        "ADFM_death_head_backwards.rfa", "ADFM_death_head_forwards.rfa", "ADFM_death_leg_left.rfa", "ADFM_death_leg_right.rfa",
        "ADFM_flinch_back.rfa", "ADFM_flinch_chest.rfa", "ADFM_flinch_leg_left.rfa", "ADFM_flinch_leg_right.rfa",
        "ADFM_freefall.rfa", "ADFM_hit_alarm.rfa", "adfm_run_flail.rfa", "admin_fem_corpsecarry.rfa",
        "admin_fem_corpsedrop.rfa", "admin_fem_idle01.rfa", "admin_fem_run.rfa", "admin_fem_run_flee.rfa",
        "admin_fem_stand.rfa", "admin_fem_talk.rfa", "admin_fem_talk_short.rfa", "admin_fem_walk.rfa",
        "masa_sar_attack_crouch.rfa", "masa_sar_attack_stand.rfa", "masa_sar_crouch_fire.rfa", "masa_sar_reload.rfa",
        "masa_sar_run.rfa", "masa_sar_stand_fire.rfa", "masa_sar_walk.rfa", "mnr3f_12mm_crouch.rfa",
        "mnr3f_12mm_crouch_walk.rfa", "mnr3f_12mm_fire.rfa", "mnr3f_12mm_reload.rfa", "mnr3f_12mm_run.rfa",
        "mnr3f_12mm_stand.rfa", "mnr3f_12mm_walk.rfa", "mnr3f_AR_crouch.rfa", "mnr3f_AR_crouch_walk.rfa",
        "mnr3f_AR_fire.rfa", "mnr3f_AR_reload.rfa", "mnr3f_AR_run.rfa", "mnr3f_AR_stand.rfa",
        "mnr3f_AR_walk.rfa", "mnr3f_FT_crouch.rfa", "mnr3f_FT_crouch_walk.rfa", "mnr3f_FT_fire.rfa",
        "mnr3f_FT_fire_ALT.rfa", "mnr3f_FT_reload.rfa", "mnr3f_FT_run.rfa", "mnr3f_FT_stand.rfa",
        "mnr3f_FT_walk.rfa", "mnr3f_GRN_fire.rfa", "mnr3f_GRN_fire_ALT.rfa", "mnr3f_Hmac_crouch.rfa",
        "mnr3f_Hmac_crouch_walk.rfa", "mnr3f_Hmac_fire.rfa", "mnr3f_Hmac_reload.rfa", "mnr3f_Hmac_run.rfa",
        "mnr3f_Hmac_stand.rfa", "mnr3f_Hmac_walk.rfa", "mnr3f_Rcharge_toss.rfa", "mnr3f_RL_crouch.rfa",
        "mnr3f_RL_crouch_walk.rfa", "mnr3f_RL_fire.rfa", "mnr3f_RL_reload.rfa", "mnr3f_RL_run.rfa",
        "mnr3f_RL_stand.rfa", "mnr3f_RL_walk.rfa", "mnr3f_RR_crouch.rfa", "mnr3f_RR_crouch_walk.rfa",
        "mnr3f_RR_fire.rfa", "mnr3f_RR_run.rfa", "mnr3f_RR_stand.rfa", "mnr3f_RR_walk.rfa",
        "mnr3f_rs_crouch_walk.rfa", "mnr3f_Rshield_crouch.rfa", "mnr3f_Rshield_crouch_walk.rfa", "mnr3f_Rshield_fire.rfa",
        "mnr3f_Rshield_run.rfa", "mnr3f_Rshield_stand.rfa", "mnr3f_Rshield_walk.rfa", "mnr3f_SG_crouch_walk.rfa",
        "mnr3f_SMC_crouch.rfa", "mnr3f_SMC_crouch_walk.rfa", "mnr3f_SMC_fire.rfa", "mnr3f_SMC_run.rfa",
        "mnr3f_SMC_stand.rfa", "mnr3f_SMC_walk.rfa", "mnr3f_SR_crouch.rfa", "mnr3f_SR_crouch_walk.rfa",
        "mnr3f_SR_fire.rfa", "mnr3f_SR_reload.rfa", "mnr3f_SR_run.rfa", "mnr3f_SR_stand.rfa",
        "mnr3f_SR_walk.rfa", "mnr3f_swim_stand.rfa", "mnr3f_swim_walk.rfa", "mnrf_mp_attack_crouch.rfa",
        "mnrf_mp_attack_stand.rfa", "mnrf_mp_crouch_fire.rfa", "mnrf_mp_reload.rfa", "mnrf_mp_run.rfa",
        "mnrf_mp_stand_fire.rfa", "mnrf_mp_walk.rfa", "mnrf_rs_attack.rfa", "mnrf_rs_attack_crouch.rfa",
        "mnrf_rs_attack_stand.rfa", "mnrf_rs_crouch_attack.rfa", "mnrf_rs_reload.rfa", "mnrf_rs_run.rfa",
        "mnrf_rs_taser.rfa", "mnrf_rs_taser_crouch.rfa", "mnrf_rs_walk.rfa", "mnrf_sg_attack_crouch.rfa",
        "mnrf_sg_attack_stand.rfa", "mnrf_sg_crouch_fire_auto.rfa", "mnrf_sg_crouch_fire_pump.rfa", "mnrf_sg_fire_auto.rfa",
        "mnrf_sg_fire_pump.rfa", "mnrf_sg_reload.rfa", "mnrf_sg_run.rfa", "mnrf_sg_walk.rfa",
    ],
    "masako.v3c": [  # MP: masako / multi_female, 134 anims
        "ADFM_cower.rfa", "ADFM_crouch.rfa", "ADFM_death_blast_backwards.rfa", "ADFM_death_blast_forwards.rfa",
        "ADFM_death_chest_backwards.rfa", "ADFM_death_chest_forwards.rfa", "ADFM_death_crouch.rfa", "ADFM_death_generic.rfa",
        "ADFM_death_head_backwards.rfa", "ADFM_death_head_forwards.rfa", "ADFM_death_leg_left.rfa", "ADFM_death_leg_right.rfa",
        "ADFM_flinch_back.rfa", "ADFM_flinch_chest.rfa", "ADFM_flinch_leg_left.rfa", "ADFM_flinch_leg_right.rfa",
        "ADFM_freefall.rfa", "adfm_run_flail.rfa", "admin_fem_corpsecarry.rfa", "admin_fem_corpsedrop.rfa",
        "admin_fem_idle01.rfa", "admin_fem_run.rfa", "admin_fem_run_flee.rfa", "admin_fem_walk.rfa",
        "eos_sidestep_left.rfa", "eos_sidestep_right.rfa", "masa_attack_roll_left.rfa", "masa_attack_roll_right.rfa",
        "masa_sar_attack_crouch.rfa", "masa_sar_attack_stand.rfa", "masa_sar_crouch.rfa", "masa_sar_crouch_fire.rfa",
        "masa_sar_flinch_back.rfa", "masa_sar_flinch_chest.rfa", "masa_sar_reload.rfa", "masa_sar_run.rfa",
        "masa_sar_stand.rfa", "masa_sar_stand_fire.rfa", "masa_sar_walk.rfa", "masa_talk.rfa",
        "masa_talk_short.rfa", "mnr3f_12mm_crouch.rfa", "mnr3f_12mm_crouch_walk.rfa", "mnr3f_12mm_fire.rfa",
        "mnr3f_12mm_reload.rfa", "mnr3f_12mm_run.rfa", "mnr3f_12mm_stand.rfa", "mnr3f_12mm_walk.rfa",
        "mnr3f_AR_crouch.rfa", "mnr3f_AR_crouch_walk.rfa", "mnr3f_AR_fire.rfa", "mnr3f_AR_reload.rfa",
        "mnr3f_AR_run.rfa", "mnr3f_AR_stand.rfa", "mnr3f_AR_walk.rfa", "mnr3f_FT_crouch.rfa",
        "mnr3f_FT_crouch_walk.rfa", "mnr3f_FT_fire.rfa", "mnr3f_FT_fire_ALT.rfa", "mnr3f_FT_reload.rfa",
        "mnr3f_FT_run.rfa", "mnr3f_FT_stand.rfa", "mnr3f_FT_walk.rfa", "mnr3f_GRN_fire.rfa",
        "mnr3f_GRN_fire_ALT.rfa", "mnr3f_Hmac_crouch.rfa", "mnr3f_Hmac_crouch_walk.rfa", "mnr3f_Hmac_fire.rfa",
        "mnr3f_Hmac_reload.rfa", "mnr3f_Hmac_run.rfa", "mnr3f_Hmac_stand.rfa", "mnr3f_Hmac_walk.rfa",
        "mnr3f_Rcharge_toss.rfa", "mnr3f_RL_crouch.rfa", "mnr3f_RL_crouch_walk.rfa", "mnr3f_RL_fire.rfa",
        "mnr3f_RL_reload.rfa", "mnr3f_RL_run.rfa", "mnr3f_RL_stand.rfa", "mnr3f_RL_walk.rfa",
        "mnr3f_RR_crouch.rfa", "mnr3f_RR_crouch_walk.rfa", "mnr3f_RR_fire.rfa", "mnr3f_RR_run.rfa",
        "mnr3f_RR_stand.rfa", "mnr3f_RR_walk.rfa", "mnr3f_rs_crouch_walk.rfa", "mnr3f_Rshield_crouch.rfa",
        "mnr3f_Rshield_crouch_walk.rfa", "mnr3f_Rshield_fire.rfa", "mnr3f_Rshield_run.rfa", "mnr3f_Rshield_stand.rfa",
        "mnr3f_Rshield_walk.rfa", "mnr3f_SG_crouch_walk.rfa", "mnr3f_SMC_crouch.rfa", "mnr3f_SMC_crouch_walk.rfa",
        "mnr3f_SMC_fire.rfa", "mnr3f_SMC_run.rfa", "mnr3f_SMC_stand.rfa", "mnr3f_SMC_walk.rfa",
        "mnr3f_SR_crouch.rfa", "mnr3f_SR_crouch_walk.rfa", "mnr3f_SR_fire.rfa", "mnr3f_SR_reload.rfa",
        "mnr3f_SR_run.rfa", "mnr3f_SR_stand.rfa", "mnr3f_SR_walk.rfa", "mnr3f_swim_stand.rfa",
        "mnr3f_swim_walk.rfa", "mnrf_mp_attack_crouch.rfa", "mnrf_mp_attack_stand.rfa", "mnrf_mp_crouch_fire.rfa",
        "mnrf_mp_reload.rfa", "mnrf_mp_run.rfa", "mnrf_mp_stand_fire.rfa", "mnrf_mp_walk.rfa",
        "mnrf_rs_attack.rfa", "mnrf_rs_attack_crouch.rfa", "mnrf_rs_attack_stand.rfa", "mnrf_rs_crouch_attack.rfa",
        "mnrf_rs_reload.rfa", "mnrf_rs_run.rfa", "mnrf_rs_taser.rfa", "mnrf_rs_taser_crouch.rfa",
        "mnrf_rs_walk.rfa", "mnrf_sg_attack_crouch.rfa", "mnrf_sg_attack_stand.rfa", "mnrf_sg_crouch_fire_auto.rfa",
        "mnrf_sg_crouch_fire_pump.rfa", "mnrf_sg_fire_auto.rfa", "mnrf_sg_fire_pump.rfa", "mnrf_sg_reload.rfa",
        "mnrf_sg_run.rfa", "mnrf_sg_walk.rfa",
    ],
    "eos.v3c": [  # MP: eos / multi_female, 128 anims
        "ADFM_cower.rfa", "ADFM_crouch.rfa", "ADFM_death_blast_backwards.rfa", "ADFM_death_blast_forwards.rfa",
        "ADFM_death_chest_backwards.rfa", "ADFM_death_chest_forwards.rfa", "ADFM_death_crouch.rfa", "ADFM_death_generic.rfa",
        "ADFM_death_head_backwards.rfa", "ADFM_death_head_forwards.rfa", "ADFM_death_leg_left.rfa", "ADFM_flinch_back.rfa",
        "ADFM_flinch_chest.rfa", "ADFM_flinch_leg_left.rfa", "ADFM_flinch_leg_right.rfa", "ADFM_freefall.rfa",
        "adfm_run_flail.rfa", "admin_fem_corpsedrop.rfa", "admin_fem_idle01.rfa", "admin_fem_run.rfa",
        "admin_fem_run_flee.rfa", "admin_fem_walk.rfa", "eos_sidestep_left.rfa", "eos_sidestep_right.rfa",
        "eos_talk.rfa", "eos_talk_short.rfa", "masa_sar_attack_crouch.rfa", "masa_sar_attack_stand.rfa",
        "masa_sar_crouch_fire.rfa", "masa_sar_reload.rfa", "masa_sar_run.rfa", "masa_sar_stand_fire.rfa",
        "masa_sar_walk.rfa", "mnr3f_12mm_crouch.rfa", "mnr3f_12mm_crouch_walk.rfa", "mnr3f_12mm_fire.rfa",
        "mnr3f_12mm_reload.rfa", "mnr3f_12mm_run.rfa", "mnr3f_12mm_stand.rfa", "mnr3f_12mm_walk.rfa",
        "mnr3f_AR_crouch.rfa", "mnr3f_AR_crouch_walk.rfa", "mnr3f_AR_fire.rfa", "mnr3f_AR_reload.rfa",
        "mnr3f_AR_run.rfa", "mnr3f_AR_stand.rfa", "mnr3f_AR_walk.rfa", "mnr3f_FT_crouch.rfa",
        "mnr3f_FT_crouch_walk.rfa", "mnr3f_FT_fire.rfa", "mnr3f_FT_fire_ALT.rfa", "mnr3f_FT_reload.rfa",
        "mnr3f_FT_run.rfa", "mnr3f_FT_stand.rfa", "mnr3f_FT_walk.rfa", "mnr3f_GRN_fire.rfa",
        "mnr3f_GRN_fire_ALT.rfa", "mnr3f_Hmac_crouch.rfa", "mnr3f_Hmac_crouch_walk.rfa", "mnr3f_Hmac_fire.rfa",
        "mnr3f_Hmac_reload.rfa", "mnr3f_Hmac_run.rfa", "mnr3f_Hmac_stand.rfa", "mnr3f_Hmac_walk.rfa",
        "mnr3f_Rcharge_toss.rfa", "mnr3f_RL_crouch.rfa", "mnr3f_RL_crouch_walk.rfa", "mnr3f_RL_fire.rfa",
        "mnr3f_RL_reload.rfa", "mnr3f_RL_run.rfa", "mnr3f_RL_stand.rfa", "mnr3f_RL_walk.rfa",
        "mnr3f_RR_crouch.rfa", "mnr3f_RR_crouch_walk.rfa", "mnr3f_RR_fire.rfa", "mnr3f_RR_run.rfa",
        "mnr3f_RR_stand.rfa", "mnr3f_RR_walk.rfa", "mnr3f_rs_crouch_walk.rfa", "mnr3f_Rshield_crouch.rfa",
        "mnr3f_Rshield_crouch_walk.rfa", "mnr3f_Rshield_fire.rfa", "mnr3f_Rshield_run.rfa", "mnr3f_Rshield_stand.rfa",
        "mnr3f_Rshield_walk.rfa", "mnr3f_SG_crouch_walk.rfa", "mnr3f_SMC_crouch.rfa", "mnr3f_SMC_crouch_walk.rfa",
        "mnr3f_SMC_fire.rfa", "mnr3f_SMC_run.rfa", "mnr3f_SMC_stand.rfa", "mnr3f_SMC_walk.rfa",
        "mnr3f_SR_crouch.rfa", "mnr3f_SR_crouch_walk.rfa", "mnr3f_SR_fire.rfa", "mnr3f_SR_reload.rfa",
        "mnr3f_SR_run.rfa", "mnr3f_SR_stand.rfa", "mnr3f_SR_walk.rfa", "mnr3f_swim_stand.rfa",
        "mnr3f_swim_walk.rfa", "mnrf_mp_attack_crouch.rfa", "mnrf_mp_attack_stand.rfa", "mnrf_mp_crouch.rfa",
        "mnrf_mp_crouch_fire.rfa", "mnrf_mp_reload.rfa", "mnrf_mp_run.rfa", "mnrf_mp_stand.rfa",
        "mnrf_mp_stand_fire.rfa", "mnrf_mp_walk.rfa", "mnrf_rs_attack.rfa", "mnrf_rs_attack_crouch.rfa",
        "mnrf_rs_attack_stand.rfa", "mnrf_rs_crouch_attack.rfa", "mnrf_rs_reload.rfa", "mnrf_rs_run.rfa",
        "mnrf_rs_taser.rfa", "mnrf_rs_taser_crouch.rfa", "mnrf_rs_walk.rfa", "mnrf_sg_attack_crouch.rfa",
        "mnrf_sg_attack_stand.rfa", "mnrf_sg_crouch_fire_auto.rfa", "mnrf_sg_crouch_fire_pump.rfa", "mnrf_sg_fire_auto.rfa",
        "mnrf_sg_fire_pump.rfa", "mnrf_sg_reload.rfa", "mnrf_sg_run.rfa", "mnrf_sg_walk.rfa",
    ],
    "merc_grunt.v3c": [  # MP: merc_grunt / multi_merc, 125 anims
        "mcom_swim_stand.rfa", "mcom_swim_walk.rfa", "mrc1_attack_crouch.rfa", "mrc1_attack_crouch_RR.rfa",
        "mrc1_attack_fire.rfa", "mrc1_attack_run.rfa", "mrc1_attack_run_RR.rfa", "mrc1_attack_stand.rfa",
        "mrc1_attack_stand_grenade.rfa", "mrc1_attack_stand_RR.rfa", "mrc1_attack_walk.rfa", "mrc1_attack_walk_RR.rfa",
        "mrc1_corpse_carried.rfa", "mrc1_corpse_drop.rfa", "mrc1_cower.rfa", "mrc1_crouch.rfa",
        "mrc1_death_blast_back.rfa", "mrc1_death_blast_fore.rfa", "mrc1_death_crouch.rfa", "mrc1_death_generic.rfa",
        "mrc1_death_generic_chest.rfa", "mrc1_death_generic_head.rfa", "mrc1_death_head_back.rfa", "mrc1_death_head_fore.rfa",
        "mrc1_death_leg_L.rfa", "mrc1_death_leg_R.rfa", "mrc1_death_torso_back.rfa", "mrc1_death_torso_fore.rfa",
        "mrc1_fire_ALT_FT.rfa", "mrc1_fire_grenade.rfa", "mrc1_fire_grenade_ALT.rfa", "mrc1_fire_reload_RR.rfa",
        "mrc1_fire_reload_RR_C.rfa", "mrc1_Fleerun.rfa", "mrc1_flinch_back.rfa", "mrc1_flinch_chest.rfa",
        "mrc1_flinch_leg_L.rfa", "mrc1_flinch_leg_R.rfa", "mrc1_freefall.rfa", "mrc1_ft_alt_fire_stand.rfa",
        "mrc1_idle.rfa", "mrc1_idle2.rfa", "mrc1_idle2_RR.rfa", "mrc1_reload-FT.rfa",
        "mrc1_reload.rfa", "mrc1_run.rfa", "mrc1_run_flail.rfa", "mrc1_sidestep_left.rfa",
        "mrc1_sidestep_right.rfa", "mrc1_stand-fire-FT.rfa", "mrc1_stand.rfa", "mrc1_stand_FT.rfa",
        "mrc1_walk.rfa", "mrc2_crouch_12mm.rfa", "mrc2_crouch_MP.rfa", "mrc2_crouch_RL.rfa",
        "mrc2_crouch_RS.rfa", "mrc2_crouch_Rshield.rfa", "mrc2_crouch_SG.rfa", "mrc2_crouch_SR.rfa",
        "mrc2_crouch_walk_12mm.rfa", "mrc2_crouch_walk_FT.rfa", "mrc2_crouch_walk_GRN.rfa", "mrc2_crouch_walk_Hmac.rfa",
        "mrc2_crouch_walk_MP.rfa", "mrc2_crouch_walk_RL.rfa", "mrc2_crouch_walk_RR.rfa", "mrc2_crouch_walk_RS.rfa",
        "mrc2_crouch_walk_Rshield.rfa", "mrc2_crouch_walk_SG.rfa", "mrc2_crouch_walk_SMC.rfa", "mrc2_crouch_walk_SR.rfa",
        "mrc2_fire_12mm.rfa", "mrc2_fire_ALT_SG.rfa", "mrc2_fire_MP.rfa", "mrc2_fire_Rcharge.rfa",
        "mrc2_fire_RL.rfa", "mrc2_fire_RS.rfa", "mrc2_fire_Rshield.rfa", "mrc2_fire_SG.rfa",
        "mrc2_fire_SR.rfa", "mrc2_reload_12mm.rfa", "mrc2_reload_MP.rfa", "mrc2_reload_RL.rfa",
        "mrc2_reload_RS.rfa", "mrc2_reload_SG.rfa", "mrc2_reload_SR.rfa", "mrc2_run_12mm.rfa",
        "mrc2_run_FT.rfa", "mrc2_run_MP.rfa", "mrc2_run_RL.rfa", "mrc2_run_RS.rfa",
        "mrc2_run_Rshield.rfa", "mrc2_run_SG.rfa", "mrc2_run_SR.rfa", "mrc2_stand_12mm.rfa",
        "mrc2_stand_MP.rfa", "mrc2_stand_RL.rfa", "mrc2_stand_RS.rfa", "mrc2_stand_Rshield.rfa",
        "mrc2_stand_SG.rfa", "mrc2_stand_SR.rfa", "mrc2_walk_12mm.rfa", "mrc2_walk_MP.rfa",
        "mrc2_walk_RL.rfa", "mrc2_walk_RS.rfa", "mrc2_walk_Rshield.rfa", "mrc2_walk_SG.rfa",
        "mrc2_walk_SR.rfa", "mrch_attack_crouch.rfa", "mrch_attack_crouch_smw.rfa", "mrch_attack_fire.rfa",
        "mrch_attack_run.rfa", "mrch_attack_stand.rfa", "mrch_attack_stand_smw.rfa", "mrch_attack_walk.rfa",
        "mrch_attack_walk_smw.rfa", "mrch_crouch_fire.rfa", "mrch_fire_crouch_smw.rfa", "mrch_fire_smw.rfa",
        "mrch_flinch_back_smw.rfa", "mrch_flinch_front_smw.rfa", "mrch_idle_smw.rfa", "mrch_reload.rfa",
        "mrch_run_smw.rfa",
    ],
    "tech1.v3c": [  # MP: tech / multi_civilian, 129 anims
        "tech01_blast_back.rfa", "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa", "tech01_cower_loop.rfa",
        "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa", "tech01_death_generic.rfa",
        "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa", "tech01_death_leg_L.rfa", "tech01_death_leg_R.rfa",
        "tech01_death_spin_fall_L.rfa", "tech01_death_torso_forward.rfa", "tech01_flinch.rfa", "tech01_flinch_back.rfa",
        "tech01_flinch_leg_L.rfa", "tech01_flinch_leg_R.rfa", "tech01_freefall.rfa", "tech01_hit_alarm.rfa",
        "tech01_idle_01.rfa", "tech01_run.rfa", "tech01_run_flail.rfa", "tech01_run_flee.rfa",
        "tech01_stand.rfa", "tech01_talk.rfa", "tech01_talk_short.rfa", "tech01_walk.rfa",
        "tech_12mm_crouch.rfa", "tech_12mm_crouch_walk.rfa", "tech_12mm_fire.rfa", "tech_12mm_reload.rfa",
        "tech_12mm_run.rfa", "tech_12mm_stand.rfa", "tech_12mm_walk.rfa", "tech_ar_crouch.rfa",
        "tech_ar_crouch_walk.rfa", "tech_ar_fire.rfa", "tech_ar_reload.rfa", "tech_ar_run.rfa",
        "tech_ar_stand.rfa", "tech_ar_walk.rfa", "tech_ft_crouch.rfa", "tech_ft_crouch_walk.rfa",
        "tech_ft_fire.rfa", "Tech_FT_fire_ALT.rfa", "tech_ft_reload.rfa", "tech_ft_run.rfa",
        "tech_ft_stand.rfa", "tech_ft_walk.rfa", "tech_gren_attack.rfa", "tech_gren_crouch.rfa",
        "tech_gren_crouch_walk.rfa", "tech_gren_run.rfa", "tech_gren_stand.rfa", "tech_gren_throw.rfa",
        "tech_gren_throw_alt.rfa", "tech_gren_walk.rfa", "tech_hmac_crouch.rfa", "tech_hmac_crouch_walk.rfa",
        "tech_hmac_fire.rfa", "tech_hmac_reload.rfa", "tech_hmac_run.rfa", "tech_hmac_stand.rfa",
        "tech_hmac_walk.rfa", "tech_mp_crouch.rfa", "tech_mp_crouch_walk.rfa", "tech_mp_fire.rfa",
        "tech_mp_reload.rfa", "tech_mp_run.rfa", "tech_mp_stand.rfa", "tech_mp_walk.rfa",
        "tech_rc_toss.rfa", "tech_RL_crouch.rfa", "tech_RL_crouch_walk.rfa", "tech_RL_fire.rfa",
        "tech_RL_reload.rfa", "tech_RL_run.rfa", "tech_RL_stand.rfa", "tech_RL_walk.rfa",
        "tech_rr_crouch.rfa", "tech_rr_crouch_walk.rfa", "tech_rr_reload.rfa", "tech_rr_run.rfa",
        "tech_rr_stand.rfa", "tech_rr_walk.rfa", "tech_rs_attack.rfa", "tech_rs_attack_alt.rfa",
        "tech_rs_crouch.rfa", "tech_rs_crouch_walk.rfa", "tech_rs_reload.rfa", "tech_rs_run.rfa",
        "tech_rs_stand.rfa", "tech_rs_walk.rfa", "tech_rshield_crouch.rfa", "tech_rshield_crouch_walk.rfa",
        "tech_rshield_fire.rfa", "tech_rshield_run.rfa", "tech_rshield_stand.rfa", "tech_rshield_walk.rfa",
        "tech_sar_crouch.rfa", "tech_sar_crouch_walk.rfa", "tech_sar_fire.rfa", "tech_sar_reload.rfa",
        "tech_sar_run.rfa", "tech_sar_stand.rfa", "tech_sar_walk.rfa", "tech_sg_crouch.rfa",
        "tech_sg_crouch_walk.rfa", "tech_sg_fire.rfa", "tech_sg_fire_auto.rfa", "tech_sg_reload.rfa",
        "tech_sg_run.rfa", "tech_sg_stand.rfa", "tech_sg_walk.rfa", "tech_smc_crouch.rfa",
        "tech_smc_crouch_walk.rfa", "tech_smc_fire_reload.rfa", "tech_smc_stand.rfa", "tech_smc_walk.rfa",
        "tech_SR_crouch.rfa", "tech_SR_crouch_walk.rfa", "tech_SR_fire.rfa", "tech_SR_reload.rfa",
        "tech_SR_run.rfa", "tech_SR_stand.rfa", "tech_SR_walk.rfa", "Tech_swim_stand.rfa",
        "Tech_swim_walk.rfa",
    ],
    "ult_scientist.v3c": [  # MP: scientist / multi_civilian, 128 anims
        "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa", "tech01_cower_loop.rfa", "tech01_crouch.rfa",
        "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa", "tech01_death_generic.rfa", "tech01_death_head_back.rfa",
        "tech01_death_head_fore.rfa", "tech01_death_leg_L.rfa", "tech01_death_leg_R.rfa", "tech01_death_spin_fall_L.rfa",
        "tech01_death_torso_forward.rfa", "tech01_flinch.rfa", "tech01_flinch_back.rfa", "tech01_flinch_leg_L.rfa",
        "tech01_flinch_leg_R.rfa", "tech01_freefall.rfa", "tech01_hit_alarm.rfa", "tech01_idle_01.rfa",
        "tech01_run.rfa", "tech01_run_flail.rfa", "tech01_run_flee.rfa", "tech01_stand.rfa",
        "tech01_walk.rfa", "tech_12mm_crouch.rfa", "tech_12mm_crouch_walk.rfa", "tech_12mm_fire.rfa",
        "tech_12mm_reload.rfa", "tech_12mm_run.rfa", "tech_12mm_stand.rfa", "tech_12mm_walk.rfa",
        "tech_ar_crouch.rfa", "tech_ar_crouch_walk.rfa", "tech_ar_fire.rfa", "tech_ar_reload.rfa",
        "tech_ar_run.rfa", "tech_ar_stand.rfa", "tech_ar_walk.rfa", "tech_ft_crouch.rfa",
        "tech_ft_crouch_walk.rfa", "tech_ft_fire.rfa", "Tech_FT_fire_ALT.rfa", "tech_ft_reload.rfa",
        "tech_ft_run.rfa", "tech_ft_stand.rfa", "tech_ft_walk.rfa", "tech_gren_attack.rfa",
        "tech_gren_crouch.rfa", "tech_gren_crouch_walk.rfa", "tech_gren_run.rfa", "tech_gren_stand.rfa",
        "tech_gren_throw.rfa", "tech_gren_throw_alt.rfa", "tech_gren_walk.rfa", "tech_hmac_crouch.rfa",
        "tech_hmac_crouch_walk.rfa", "tech_hmac_fire.rfa", "tech_hmac_reload.rfa", "tech_hmac_run.rfa",
        "tech_hmac_stand.rfa", "tech_hmac_walk.rfa", "tech_mp_crouch.rfa", "tech_mp_crouch_walk.rfa",
        "tech_mp_fire.rfa", "tech_mp_reload.rfa", "tech_mp_run.rfa", "tech_mp_stand.rfa",
        "tech_mp_walk.rfa", "tech_rc_toss.rfa", "tech_RL_crouch.rfa", "tech_RL_crouch_walk.rfa",
        "tech_RL_fire.rfa", "tech_RL_reload.rfa", "tech_RL_run.rfa", "tech_RL_stand.rfa",
        "tech_RL_walk.rfa", "tech_rr_crouch.rfa", "tech_rr_crouch_walk.rfa", "tech_rr_reload.rfa",
        "tech_rr_run.rfa", "tech_rr_stand.rfa", "tech_rr_walk.rfa", "tech_rs_attack.rfa",
        "tech_rs_attack_alt.rfa", "tech_rs_crouch.rfa", "tech_rs_crouch_walk.rfa", "tech_rs_reload.rfa",
        "tech_rs_run.rfa", "tech_rs_stand.rfa", "tech_rs_walk.rfa", "tech_rshield_crouch.rfa",
        "tech_rshield_crouch_walk.rfa", "tech_rshield_fire.rfa", "tech_rshield_run.rfa", "tech_rshield_stand.rfa",
        "tech_rshield_walk.rfa", "tech_sar_crouch.rfa", "tech_sar_crouch_walk.rfa", "tech_sar_fire.rfa",
        "tech_sar_reload.rfa", "tech_sar_run.rfa", "tech_sar_stand.rfa", "tech_sar_walk.rfa",
        "tech_sg_crouch.rfa", "tech_sg_crouch_walk.rfa", "tech_sg_fire.rfa", "tech_sg_fire_auto.rfa",
        "tech_sg_reload.rfa", "tech_sg_run.rfa", "tech_sg_stand.rfa", "tech_sg_walk.rfa",
        "tech_smc_crouch.rfa", "tech_smc_crouch_walk.rfa", "tech_smc_fire_reload.rfa", "tech_smc_stand.rfa",
        "tech_smc_walk.rfa", "tech_SR_crouch.rfa", "tech_SR_crouch_walk.rfa", "tech_SR_fire.rfa",
        "tech_SR_reload.rfa", "tech_SR_run.rfa", "tech_SR_stand.rfa", "tech_SR_walk.rfa",
        "Tech_swim_stand.rfa", "Tech_swim_walk.rfa", "usci_talk.rfa", "usci_talk_short.rfa",
    ],
    "medic1.v3c": [  # MP: medic1 / multi_civilian, 129 anims
        "medc_talk.rfa", "medc_talk_short.rfa", "medic01_heal_01.rfa", "tech01_blast_forwards.rfa",
        "tech01_corpse_carry.rfa", "tech01_cower_loop.rfa", "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa",
        "tech01_death_crouch.rfa", "tech01_death_generic.rfa", "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa",
        "tech01_death_leg_L.rfa", "tech01_death_leg_R.rfa", "tech01_death_spin_fall_L.rfa", "tech01_death_torso_forward.rfa",
        "tech01_flinch.rfa", "tech01_flinch_back.rfa", "tech01_flinch_leg_L.rfa", "tech01_flinch_leg_R.rfa",
        "tech01_freefall.rfa", "tech01_hit_alarm.rfa", "tech01_idle_01.rfa", "tech01_run.rfa",
        "tech01_run_flail.rfa", "tech01_run_flee.rfa", "tech01_stand.rfa", "tech01_walk.rfa",
        "tech_12mm_crouch.rfa", "tech_12mm_crouch_walk.rfa", "tech_12mm_fire.rfa", "tech_12mm_reload.rfa",
        "tech_12mm_run.rfa", "tech_12mm_stand.rfa", "tech_12mm_walk.rfa", "tech_ar_crouch.rfa",
        "tech_ar_crouch_walk.rfa", "tech_ar_fire.rfa", "tech_ar_reload.rfa", "tech_ar_run.rfa",
        "tech_ar_stand.rfa", "tech_ar_walk.rfa", "tech_ft_crouch.rfa", "tech_ft_crouch_walk.rfa",
        "tech_ft_fire.rfa", "Tech_FT_fire_ALT.rfa", "tech_ft_reload.rfa", "tech_ft_run.rfa",
        "tech_ft_stand.rfa", "tech_ft_walk.rfa", "tech_gren_attack.rfa", "tech_gren_crouch.rfa",
        "tech_gren_crouch_walk.rfa", "tech_gren_run.rfa", "tech_gren_stand.rfa", "tech_gren_throw.rfa",
        "tech_gren_throw_alt.rfa", "tech_gren_walk.rfa", "tech_hmac_crouch.rfa", "tech_hmac_crouch_walk.rfa",
        "tech_hmac_fire.rfa", "tech_hmac_reload.rfa", "tech_hmac_run.rfa", "tech_hmac_stand.rfa",
        "tech_hmac_walk.rfa", "tech_mp_crouch.rfa", "tech_mp_crouch_walk.rfa", "tech_mp_fire.rfa",
        "tech_mp_reload.rfa", "tech_mp_run.rfa", "tech_mp_stand.rfa", "tech_mp_walk.rfa",
        "tech_rc_toss.rfa", "tech_RL_crouch.rfa", "tech_RL_crouch_walk.rfa", "tech_RL_fire.rfa",
        "tech_RL_reload.rfa", "tech_RL_run.rfa", "tech_RL_stand.rfa", "tech_RL_walk.rfa",
        "tech_rr_crouch.rfa", "tech_rr_crouch_walk.rfa", "tech_rr_reload.rfa", "tech_rr_run.rfa",
        "tech_rr_stand.rfa", "tech_rr_walk.rfa", "tech_rs_attack.rfa", "tech_rs_attack_alt.rfa",
        "tech_rs_crouch.rfa", "tech_rs_crouch_walk.rfa", "tech_rs_reload.rfa", "tech_rs_run.rfa",
        "tech_rs_stand.rfa", "tech_rs_walk.rfa", "tech_rshield_crouch.rfa", "tech_rshield_crouch_walk.rfa",
        "tech_rshield_fire.rfa", "tech_rshield_run.rfa", "tech_rshield_stand.rfa", "tech_rshield_walk.rfa",
        "tech_sar_crouch.rfa", "tech_sar_crouch_walk.rfa", "tech_sar_fire.rfa", "tech_sar_reload.rfa",
        "tech_sar_run.rfa", "tech_sar_stand.rfa", "tech_sar_walk.rfa", "tech_sg_crouch.rfa",
        "tech_sg_crouch_walk.rfa", "tech_sg_fire.rfa", "tech_sg_fire_auto.rfa", "tech_sg_reload.rfa",
        "tech_sg_run.rfa", "tech_sg_stand.rfa", "tech_sg_walk.rfa", "tech_smc_crouch.rfa",
        "tech_smc_crouch_walk.rfa", "tech_smc_fire_reload.rfa", "tech_smc_stand.rfa", "tech_smc_walk.rfa",
        "tech_SR_crouch.rfa", "tech_SR_crouch_walk.rfa", "tech_SR_fire.rfa", "tech_SR_reload.rfa",
        "tech_SR_run.rfa", "tech_SR_stand.rfa", "tech_SR_walk.rfa", "Tech_swim_stand.rfa",
        "Tech_swim_walk.rfa",
    ],
    "env_scientist.v3c": [  # MP: env_scientist1 / multi_civilian, 133 anims
        "esci_cower.rfa", "esci_crouch.rfa", "esci_death_chest_forwards.rfa", "esci_death_generic.rfa",
        "esci_flinch_back.rfa", "esci_flinch_chest.rfa", "esci_flinch_leg_left.rfa", "esci_flinch_leg_right.rfa",
        "esci_idle.rfa", "esci_push_button.rfa", "esci_run.rfa", "esci_run_flee.rfa",
        "esci_stand.rfa", "esci_walk.rfa", "tech01_blast_back.rfa", "tech01_blast_forwards.rfa",
        "tech01_corpse_carry.rfa", "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa", "tech01_death_generic.rfa",
        "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa", "tech01_death_leg_L.rfa", "tech01_death_leg_R.rfa",
        "tech01_death_spin_fall_L.rfa", "tech01_death_torso_forward.rfa", "tech01_flinch.rfa", "tech01_flinch_back.rfa",
        "tech01_flinch_leg_L.rfa", "tech01_flinch_leg_R.rfa", "tech01_freefall.rfa", "tech01_run_flail.rfa",
        "tech_12mm_crouch.rfa", "tech_12mm_crouch_walk.rfa", "tech_12mm_fire.rfa", "tech_12mm_reload.rfa",
        "tech_12mm_run.rfa", "tech_12mm_stand.rfa", "tech_12mm_walk.rfa", "tech_ar_crouch.rfa",
        "tech_ar_crouch_walk.rfa", "tech_ar_fire.rfa", "tech_ar_reload.rfa", "tech_ar_run.rfa",
        "tech_ar_stand.rfa", "tech_ar_walk.rfa", "tech_ft_crouch.rfa", "tech_ft_crouch_walk.rfa",
        "tech_ft_fire.rfa", "Tech_FT_fire_ALT.rfa", "tech_ft_reload.rfa", "tech_ft_run.rfa",
        "tech_ft_stand.rfa", "tech_ft_walk.rfa", "tech_gren_attack.rfa", "tech_gren_crouch.rfa",
        "tech_gren_crouch_walk.rfa", "tech_gren_run.rfa", "tech_gren_stand.rfa", "tech_gren_throw.rfa",
        "tech_gren_throw_alt.rfa", "tech_gren_walk.rfa", "tech_hmac_crouch.rfa", "tech_hmac_crouch_walk.rfa",
        "tech_hmac_fire.rfa", "tech_hmac_reload.rfa", "tech_hmac_run.rfa", "tech_hmac_stand.rfa",
        "tech_hmac_walk.rfa", "tech_mp_crouch.rfa", "tech_mp_crouch_walk.rfa", "tech_mp_fire.rfa",
        "tech_mp_reload.rfa", "tech_mp_run.rfa", "tech_mp_stand.rfa", "tech_mp_walk.rfa",
        "tech_rc_toss.rfa", "tech_RL_crouch.rfa", "tech_RL_crouch_walk.rfa", "tech_RL_fire.rfa",
        "tech_RL_reload.rfa", "tech_RL_run.rfa", "tech_RL_stand.rfa", "tech_RL_walk.rfa",
        "tech_rr_crouch.rfa", "tech_rr_crouch_walk.rfa", "tech_rr_reload.rfa", "tech_rr_run.rfa",
        "tech_rr_stand.rfa", "tech_rr_walk.rfa", "tech_rs_attack.rfa", "tech_rs_attack_alt.rfa",
        "tech_rs_crouch.rfa", "tech_rs_crouch_walk.rfa", "tech_rs_reload.rfa", "tech_rs_run.rfa",
        "tech_rs_stand.rfa", "tech_rs_walk.rfa", "tech_rshield_crouch.rfa", "tech_rshield_crouch_walk.rfa",
        "tech_rshield_fire.rfa", "tech_rshield_run.rfa", "tech_rshield_stand.rfa", "tech_rshield_walk.rfa",
        "tech_sar_crouch.rfa", "tech_sar_crouch_walk.rfa", "tech_sar_fire.rfa", "tech_sar_reload.rfa",
        "tech_sar_run.rfa", "tech_sar_stand.rfa", "tech_sar_walk.rfa", "tech_sg_crouch.rfa",
        "tech_sg_crouch_walk.rfa", "tech_sg_fire.rfa", "tech_sg_fire_auto.rfa", "tech_sg_reload.rfa",
        "tech_sg_run.rfa", "tech_sg_stand.rfa", "tech_sg_walk.rfa", "tech_smc_crouch.rfa",
        "tech_smc_crouch_walk.rfa", "tech_smc_fire_reload.rfa", "tech_smc_stand.rfa", "tech_smc_walk.rfa",
        "tech_SR_crouch.rfa", "tech_SR_crouch_walk.rfa", "tech_SR_fire.rfa", "tech_SR_reload.rfa",
        "tech_SR_run.rfa", "tech_SR_stand.rfa", "tech_SR_walk.rfa", "Tech_swim_stand.rfa",
        "Tech_swim_walk.rfa",
    ],
    "comp_tech.v3c": [  # MP: hendrix / multi_civilian, 128 anims
        "hndx_talk.rfa", "hndx_talk_short.rfa", "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa",
        "tech01_cower_loop.rfa", "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa",
        "tech01_death_generic.rfa", "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa", "tech01_death_leg_L.rfa",
        "tech01_death_leg_R.rfa", "tech01_death_spin_fall_L.rfa", "tech01_death_torso_forward.rfa", "tech01_flinch.rfa",
        "tech01_flinch_back.rfa", "tech01_flinch_leg_L.rfa", "tech01_flinch_leg_R.rfa", "tech01_freefall.rfa",
        "tech01_hit_alarm.rfa", "tech01_idle_01.rfa", "tech01_run.rfa", "tech01_run_flail.rfa",
        "tech01_run_flee.rfa", "tech01_stand.rfa", "tech01_walk.rfa", "tech_12mm_crouch.rfa",
        "tech_12mm_crouch_walk.rfa", "tech_12mm_fire.rfa", "tech_12mm_reload.rfa", "tech_12mm_run.rfa",
        "tech_12mm_stand.rfa", "tech_12mm_walk.rfa", "tech_ar_crouch.rfa", "tech_ar_crouch_walk.rfa",
        "tech_ar_fire.rfa", "tech_ar_reload.rfa", "tech_ar_run.rfa", "tech_ar_stand.rfa",
        "tech_ar_walk.rfa", "tech_ft_crouch.rfa", "tech_ft_crouch_walk.rfa", "tech_ft_fire.rfa",
        "Tech_FT_fire_ALT.rfa", "tech_ft_reload.rfa", "tech_ft_run.rfa", "tech_ft_stand.rfa",
        "tech_ft_walk.rfa", "tech_gren_attack.rfa", "tech_gren_crouch.rfa", "tech_gren_crouch_walk.rfa",
        "tech_gren_run.rfa", "tech_gren_stand.rfa", "tech_gren_throw.rfa", "tech_gren_throw_alt.rfa",
        "tech_gren_walk.rfa", "tech_hmac_crouch.rfa", "tech_hmac_crouch_walk.rfa", "tech_hmac_fire.rfa",
        "tech_hmac_reload.rfa", "tech_hmac_run.rfa", "tech_hmac_stand.rfa", "tech_hmac_walk.rfa",
        "tech_mp_crouch.rfa", "tech_mp_crouch_walk.rfa", "tech_mp_fire.rfa", "tech_mp_reload.rfa",
        "tech_mp_run.rfa", "tech_mp_stand.rfa", "tech_mp_walk.rfa", "tech_rc_toss.rfa",
        "tech_RL_crouch.rfa", "tech_RL_crouch_walk.rfa", "tech_RL_fire.rfa", "tech_RL_reload.rfa",
        "tech_RL_run.rfa", "tech_RL_stand.rfa", "tech_RL_walk.rfa", "tech_rr_crouch.rfa",
        "tech_rr_crouch_walk.rfa", "tech_rr_reload.rfa", "tech_rr_run.rfa", "tech_rr_stand.rfa",
        "tech_rr_walk.rfa", "tech_rs_attack.rfa", "tech_rs_attack_alt.rfa", "tech_rs_crouch.rfa",
        "tech_rs_crouch_walk.rfa", "tech_rs_reload.rfa", "tech_rs_run.rfa", "tech_rs_stand.rfa",
        "tech_rs_walk.rfa", "tech_rshield_crouch.rfa", "tech_rshield_crouch_walk.rfa", "tech_rshield_fire.rfa",
        "tech_rshield_run.rfa", "tech_rshield_stand.rfa", "tech_rshield_walk.rfa", "tech_sar_crouch.rfa",
        "tech_sar_crouch_walk.rfa", "tech_sar_fire.rfa", "tech_sar_reload.rfa", "tech_sar_run.rfa",
        "tech_sar_stand.rfa", "tech_sar_walk.rfa", "tech_sg_crouch.rfa", "tech_sg_crouch_walk.rfa",
        "tech_sg_fire.rfa", "tech_sg_fire_auto.rfa", "tech_sg_reload.rfa", "tech_sg_run.rfa",
        "tech_sg_stand.rfa", "tech_sg_walk.rfa", "tech_smc_crouch.rfa", "tech_smc_crouch_walk.rfa",
        "tech_smc_fire_reload.rfa", "tech_smc_stand.rfa", "tech_smc_walk.rfa", "tech_SR_crouch.rfa",
        "tech_SR_crouch_walk.rfa", "tech_SR_fire.rfa", "tech_SR_reload.rfa", "tech_SR_run.rfa",
        "tech_SR_stand.rfa", "tech_SR_walk.rfa", "Tech_swim_stand.rfa", "Tech_swim_walk.rfa",
    ],
    "admin_male.v3c": [  # MP: gryphon / multi_civilian, 128 anims
        "admin_male_talk.rfa", "admin_male_talk_short.rfa", "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa",
        "tech01_cower_loop.rfa", "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa", "tech01_death_crouch.rfa",
        "tech01_death_generic.rfa", "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa", "tech01_death_leg_L.rfa",
        "tech01_death_leg_R.rfa", "tech01_death_spin_fall_L.rfa", "tech01_death_torso_forward.rfa", "tech01_flinch.rfa",
        "tech01_flinch_back.rfa", "tech01_flinch_leg_L.rfa", "tech01_flinch_leg_R.rfa", "tech01_freefall.rfa",
        "tech01_hit_alarm.rfa", "tech01_idle_01.rfa", "tech01_run.rfa", "tech01_run_flail.rfa",
        "tech01_run_flee.rfa", "tech01_stand.rfa", "tech01_walk.rfa", "tech_12mm_crouch.rfa",
        "tech_12mm_crouch_walk.rfa", "tech_12mm_fire.rfa", "tech_12mm_reload.rfa", "tech_12mm_run.rfa",
        "tech_12mm_stand.rfa", "tech_12mm_walk.rfa", "tech_ar_crouch.rfa", "tech_ar_crouch_walk.rfa",
        "tech_ar_fire.rfa", "tech_ar_reload.rfa", "tech_ar_run.rfa", "tech_ar_stand.rfa",
        "tech_ar_walk.rfa", "tech_ft_crouch.rfa", "tech_ft_crouch_walk.rfa", "tech_ft_fire.rfa",
        "Tech_FT_fire_ALT.rfa", "tech_ft_reload.rfa", "tech_ft_run.rfa", "tech_ft_stand.rfa",
        "tech_ft_walk.rfa", "tech_gren_attack.rfa", "tech_gren_crouch.rfa", "tech_gren_crouch_walk.rfa",
        "tech_gren_run.rfa", "tech_gren_stand.rfa", "tech_gren_throw.rfa", "tech_gren_throw_alt.rfa",
        "tech_gren_walk.rfa", "tech_hmac_crouch.rfa", "tech_hmac_crouch_walk.rfa", "tech_hmac_fire.rfa",
        "tech_hmac_reload.rfa", "tech_hmac_run.rfa", "tech_hmac_stand.rfa", "tech_hmac_walk.rfa",
        "tech_mp_crouch.rfa", "tech_mp_crouch_walk.rfa", "tech_mp_fire.rfa", "tech_mp_reload.rfa",
        "tech_mp_run.rfa", "tech_mp_stand.rfa", "tech_mp_walk.rfa", "tech_rc_toss.rfa",
        "tech_RL_crouch.rfa", "tech_RL_crouch_walk.rfa", "tech_RL_fire.rfa", "tech_RL_reload.rfa",
        "tech_RL_run.rfa", "tech_RL_stand.rfa", "tech_RL_walk.rfa", "tech_rr_crouch.rfa",
        "tech_rr_crouch_walk.rfa", "tech_rr_reload.rfa", "tech_rr_run.rfa", "tech_rr_stand.rfa",
        "tech_rr_walk.rfa", "tech_rs_attack.rfa", "tech_rs_attack_alt.rfa", "tech_rs_crouch.rfa",
        "tech_rs_crouch_walk.rfa", "tech_rs_reload.rfa", "tech_rs_run.rfa", "tech_rs_stand.rfa",
        "tech_rs_walk.rfa", "tech_rshield_crouch.rfa", "tech_rshield_crouch_walk.rfa", "tech_rshield_fire.rfa",
        "tech_rshield_run.rfa", "tech_rshield_stand.rfa", "tech_rshield_walk.rfa", "tech_sar_crouch.rfa",
        "tech_sar_crouch_walk.rfa", "tech_sar_fire.rfa", "tech_sar_reload.rfa", "tech_sar_run.rfa",
        "tech_sar_stand.rfa", "tech_sar_walk.rfa", "tech_sg_crouch.rfa", "tech_sg_crouch_walk.rfa",
        "tech_sg_fire.rfa", "tech_sg_fire_auto.rfa", "tech_sg_reload.rfa", "tech_sg_run.rfa",
        "tech_sg_stand.rfa", "tech_sg_walk.rfa", "tech_smc_crouch.rfa", "tech_smc_crouch_walk.rfa",
        "tech_smc_fire_reload.rfa", "tech_smc_stand.rfa", "tech_smc_walk.rfa", "tech_SR_crouch.rfa",
        "tech_SR_crouch_walk.rfa", "tech_SR_fire.rfa", "tech_SR_reload.rfa", "tech_SR_run.rfa",
        "tech_SR_stand.rfa", "tech_SR_walk.rfa", "Tech_swim_stand.rfa", "Tech_swim_walk.rfa",
    ],
    "admin_male2.v3c": [  # MP: fat_admin / multi_civilian, 135 anims
        "adm2_cower.rfa", "adm2_death_chest_forwards.rfa", "adm2_death_generic.rfa", "adm2_flinch_back.rfa",
        "adm2_flinch_front.rfa", "adm2_flinch_leg_left.rfa", "adm2_flinch_leg_right.rfa", "adm2_idle.rfa",
        "adm2_run.rfa", "adm2_run_flail.rfa", "adm2_run_flee.rfa", "adm2_stand.rfa",
        "adm2_talk.rfa", "adm2_talk_short.rfa", "adm2_walk.rfa", "tech01_blast_back.rfa",
        "tech01_blast_forwards.rfa", "tech01_corpse_carry.rfa", "tech01_crouch.rfa", "tech01_death_corpse_drop.rfa",
        "tech01_death_crouch.rfa", "tech01_death_generic.rfa", "tech01_death_head_back.rfa", "tech01_death_head_fore.rfa",
        "tech01_death_leg_L.rfa", "tech01_death_leg_R.rfa", "tech01_death_spin_fall_L.rfa", "tech01_death_torso_forward.rfa",
        "tech01_flinch.rfa", "tech01_flinch_back.rfa", "tech01_flinch_leg_L.rfa", "tech01_flinch_leg_R.rfa",
        "tech01_freefall.rfa", "tech01_hit_alarm.rfa", "tech_12mm_crouch.rfa", "tech_12mm_crouch_walk.rfa",
        "tech_12mm_fire.rfa", "tech_12mm_reload.rfa", "tech_12mm_run.rfa", "tech_12mm_stand.rfa",
        "tech_12mm_walk.rfa", "tech_ar_crouch.rfa", "tech_ar_crouch_walk.rfa", "tech_ar_fire.rfa",
        "tech_ar_reload.rfa", "tech_ar_run.rfa", "tech_ar_stand.rfa", "tech_ar_walk.rfa",
        "tech_ft_crouch.rfa", "tech_ft_crouch_walk.rfa", "tech_ft_fire.rfa", "Tech_FT_fire_ALT.rfa",
        "tech_ft_reload.rfa", "tech_ft_run.rfa", "tech_ft_stand.rfa", "tech_ft_walk.rfa",
        "tech_gren_attack.rfa", "tech_gren_crouch.rfa", "tech_gren_crouch_walk.rfa", "tech_gren_run.rfa",
        "tech_gren_stand.rfa", "tech_gren_throw.rfa", "tech_gren_throw_alt.rfa", "tech_gren_walk.rfa",
        "tech_hmac_crouch.rfa", "tech_hmac_crouch_walk.rfa", "tech_hmac_fire.rfa", "tech_hmac_reload.rfa",
        "tech_hmac_run.rfa", "tech_hmac_stand.rfa", "tech_hmac_walk.rfa", "tech_mp_crouch.rfa",
        "tech_mp_crouch_walk.rfa", "tech_mp_fire.rfa", "tech_mp_reload.rfa", "tech_mp_run.rfa",
        "tech_mp_stand.rfa", "tech_mp_walk.rfa", "tech_rc_toss.rfa", "tech_RL_crouch.rfa",
        "tech_RL_crouch_walk.rfa", "tech_RL_fire.rfa", "tech_RL_reload.rfa", "tech_RL_run.rfa",
        "tech_RL_stand.rfa", "tech_RL_walk.rfa", "tech_rr_crouch.rfa", "tech_rr_crouch_walk.rfa",
        "tech_rr_reload.rfa", "tech_rr_run.rfa", "tech_rr_stand.rfa", "tech_rr_walk.rfa",
        "tech_rs_attack.rfa", "tech_rs_attack_alt.rfa", "tech_rs_crouch.rfa", "tech_rs_crouch_walk.rfa",
        "tech_rs_reload.rfa", "tech_rs_run.rfa", "tech_rs_stand.rfa", "tech_rs_walk.rfa",
        "tech_rshield_crouch.rfa", "tech_rshield_crouch_walk.rfa", "tech_rshield_fire.rfa", "tech_rshield_run.rfa",
        "tech_rshield_stand.rfa", "tech_rshield_walk.rfa", "tech_sar_crouch.rfa", "tech_sar_crouch_walk.rfa",
        "tech_sar_fire.rfa", "tech_sar_reload.rfa", "tech_sar_run.rfa", "tech_sar_stand.rfa",
        "tech_sar_walk.rfa", "tech_sg_crouch.rfa", "tech_sg_crouch_walk.rfa", "tech_sg_fire.rfa",
        "tech_sg_fire_auto.rfa", "tech_sg_reload.rfa", "tech_sg_run.rfa", "tech_sg_stand.rfa",
        "tech_sg_walk.rfa", "tech_smc_crouch.rfa", "tech_smc_crouch_walk.rfa", "tech_smc_fire_reload.rfa",
        "tech_smc_stand.rfa", "tech_smc_walk.rfa", "tech_SR_crouch.rfa", "tech_SR_crouch_walk.rfa",
        "tech_SR_fire.rfa", "tech_SR_reload.rfa", "tech_SR_run.rfa", "tech_SR_stand.rfa",
        "tech_SR_walk.rfa", "Tech_swim_stand.rfa", "Tech_swim_walk.rfa",
    ],
}


def _lookup_required_anims(model_filename):
    """Look up required animations for a model from the stock RF database.
    Merges SP (_RF_ANIM_DB) and MP (_RF_MP_ANIM_DB) animations for characters
    that appear in both. Returns deduplicated list of .rfa filenames."""
    key = os.path.basename(model_filename).lower()

    collected = []
    seen = set()

    def add_from(db, k):
        for anim in db.get(k, []):
            lc = anim.lower()
            if lc not in seen:
                seen.add(lc)
                collected.append(anim)

    # Exact match in both DBs
    add_from(_RF_ANIM_DB, key)
    add_from(_RF_MP_ANIM_DB, key)

    # Try .vcm/.v3d → .v3c variants
    if not collected:
        for ext in ['.vcm', '.v3d']:
            alt = key.replace(ext, '.v3c')
            add_from(_RF_ANIM_DB, alt)
            add_from(_RF_MP_ANIM_DB, alt)
            if collected:
                break

    return collected


# ═══════════════════════════════════════════════════════════════════════════════
#  V3C EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _pad16(buf):
    """Pad bytearray to 16-byte alignment."""
    rem = len(buf) % 16
    if rem:
        buf += b'\x00' * (16 - rem)
    return buf


def _gather_chunks(mesh_obj, bone_names):
    """Split a Blender mesh into per-material chunks with RF-space data.
    Returns list of dicts with positions, normals, uvs, triangles, bone_links."""
    import bmesh

    mesh = mesh_obj.data
    num_bones = len(bone_names)
    bone_idx = {name: i for i, name in enumerate(bone_names)}

    # Use bmesh for safe triangulated access (avoids native crashes in Blender 5.0)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    uv_layer = bm.loops.layers.uv.active

    # Group triangulated faces by material index
    mat_tris = {}
    for face in bm.faces:
        mi = face.material_index
        mat_tris.setdefault(mi, []).append(face)

    chunks = []
    for mi in sorted(mat_tris.keys()):
        faces = mat_tris[mi]

        # Check if material is double-sided (backface culling disabled)
        face_flags = 0
        if mi < len(mesh_obj.data.materials) and mesh_obj.data.materials[mi]:
            mat = mesh_obj.data.materials[mi]
            if hasattr(mat, 'use_backface_culling') and not mat.use_backface_culling:
                face_flags = 0x20  # DOUBLE_SIDED

        vert_map = {}
        positions = []
        normals = []
        uvs = []
        bone_links = []
        triangles = []

        for face in faces:
            tri_indices = []
            for loop in face.loops:
                vi = loop.vert.index

                # Position in RF space
                rf_pos = bl_to_rf_pos(loop.vert.co)

                # Normal in RF space
                rf_norm = bl_to_rf_pos(loop.vert.normal)

                # UV with V flip
                if uv_layer:
                    uv = loop[uv_layer].uv
                    rf_uv = (uv[0], 1.0 - uv[1])
                else:
                    rf_uv = (0.0, 0.0)

                key = (vi, round(rf_uv[0], 6), round(rf_uv[1], 6))

                if key not in vert_map:
                    local_idx = len(positions)
                    vert_map[key] = local_idx
                    positions.append(rf_pos)
                    normals.append(rf_norm)
                    uvs.append(rf_uv)

                    # Bone weights from vertex groups (original mesh, not bmesh)
                    vg_weights = []
                    if vi < len(mesh.vertices):
                        for g in mesh.vertices[vi].groups:
                            if g.group >= len(mesh_obj.vertex_groups):
                                continue
                            grp = mesh_obj.vertex_groups[g.group]
                            bi = bone_idx.get(grp.name, -1)
                            if bi >= 0 and g.weight > 0.001:
                                vg_weights.append((bi, g.weight))

                    vg_weights.sort(key=lambda x: -x[1])
                    vg_weights = vg_weights[:4]

                    # Convert to 0-255 byte range, normalized to sum=255
                    # Redux normalizes: divide by sum, multiply by 255, adjust slot 0
                    byte_bones = [0xFF, 0xFF, 0xFF, 0xFF]  # 0xFF = unused (Redux/stock convention)
                    byte_weights = [0, 0, 0, 0]
                    if vg_weights:
                        w_sum = sum(w for _, w in vg_weights) or 1.0
                        for wi, (bi, w) in enumerate(vg_weights):
                            byte_bones[wi] = min(bi, 255)
                            byte_weights[wi] = max(1, min(255, int(round(w / w_sum * 255.0))))
                        # Adjust slot 0 so sum is exactly 255
                        bw_sum = sum(byte_weights)
                        if bw_sum > 0 and bw_sum != 255:
                            byte_weights[0] = max(1, min(255, byte_weights[0] + 255 - bw_sum))

                    bone_links.append((tuple(byte_weights), tuple(byte_bones)))

                tri_indices.append(vert_map[key])

            # Flip winding order: Blender CCW → RF CW (swap indices 1↔2)
            triangles.append((tri_indices[0], tri_indices[2], tri_indices[1]))

        chunks.append({
            'material_index': mi,
            'positions': positions,
            'normals': normals,
            'uvs': uvs,
            'triangles': triangles,
            'bone_links': bone_links,
            'face_flags': face_flags,
        })

    bm.free()
    return chunks


def _build_data_block(chunks, is_character, prop_points=None):
    """Build the V3C data block containing chunk headers + per-chunk vertex data.
    Returns (data_block bytes, chunk_infos list)."""
    if prop_points is None:
        prop_points = []

    db = bytearray()
    chunk_infos = []

    # ── Chunk headers (0x38 = 56 bytes each) ──
    for ci, chunk in enumerate(chunks):
        header = bytearray(0x38)
        # Offset 0x20: texture index into tex_refs list (sequential, not Blender material slot)
        struct.pack_into("<i", header, 0x20, ci)
        db += header
    db = bytearray(_pad16(db))

    # ── Per-chunk vertex data ──
    for ci, chunk in enumerate(chunks):
        npos = len(chunk['positions'])
        nuvs = len(chunk['uvs'])
        ntris = len(chunk['triangles'])
        nweights = len(chunk['bone_links']) if is_character else 0

        # Positions
        for x, y, z in chunk['positions']:
            db += struct.pack("<3f", x, y, z)
        db = bytearray(_pad16(db))

        # Normals (same count as positions)
        for nx, ny, nz in chunk['normals']:
            db += struct.pack("<3f", nx, ny, nz)
        db = bytearray(_pad16(db))

        # UVs
        for u, v in chunk['uvs']:
            db += struct.pack("<2f", u, v)
        db = bytearray(_pad16(db))

        # Triangles (3 indices + 1 flags uint16 = 8 bytes each)
        face_flags = chunk.get('face_flags', 0)
        for i0, i1, i2 in chunk['triangles']:
            db += struct.pack("<4H", i0, i1, i2, face_flags)
        db = bytearray(_pad16(db))

        # Samepos (cv * 2 bytes of zeros — Redux writes this, engine expects it)
        db += b'\x00' * (npos * 2)
        db = bytearray(_pad16(db))

        # Bone weights (4 bytes weights + 4 bytes bone indices = 8 per vert)
        if is_character and nweights > 0:
            for weights, bones in chunk['bone_links']:
                db += struct.pack("<4B", *weights)
                db += struct.pack("<4B", *bones)
            db = bytearray(_pad16(db))

        # Build chunk info: 7*uint16 + uint32 = 18 bytes
        # Byte size fields must include 16-byte alignment padding
        ci_vecs = _a16(npos * 12) if npos > 0 else 0
        ci_uvs = _a16(nuvs * 8) if nuvs > 0 else 0
        ci_facesalloc = _a16(ntris * 8) if ntris > 0 else 0
        ci_samepos = _a16(npos * 2) if npos > 0 else 0  # MUST be padded like all other alloc fields
        ci_wi = _a16(nweights * 8) if nweights > 0 else 0
        ci_rflags = 0x00518C41  # standard RF render flags (matches stock + Redux)
        chunk_infos.append(struct.pack("<7HI",
            npos,           # ci_verts (authoritative count)
            ntris,          # ci_faces
            ci_vecs,        # position byte size (padded)
            ci_facesalloc,  # triangle byte size (padded)
            ci_samepos,     # samepos byte size
            ci_wi,          # weight byte size (padded)
            ci_uvs,         # uv byte size (padded)
            ci_rflags       # render flags
        ))

    # ── Prop points (in data block) ──
    for pp in prop_points:
        # name: 0x44 bytes (68 bytes, null-padded)
        name_bytes = pp['name'].encode('ascii', errors='replace')[:0x43]
        db += name_bytes + b'\x00' * (0x44 - len(name_bytes))
        # quaternion (x,y,z,w)
        db += struct.pack("<4f", *pp['quat'])
        # position
        db += struct.pack("<3f", *pp['pos'])
        # parent bone
        db += struct.pack("<i", pp['parent_bone'])

    return bytes(db), chunk_infos


def _gather_scene_cspheres(arm_obj, bone_names):
    """Read collision sphere empties from the scene back to RF format."""
    cspheres = []
    for obj in arm_obj.children:
        if obj.type != 'EMPTY' or not obj.name.startswith('CS_'):
            continue
        name = obj.name[3:]  # strip 'CS_' prefix

        # Get parent bone index
        parent_bi = -1
        if obj.parent_type == 'BONE' and obj.parent_bone:
            if obj.parent_bone in bone_names:
                parent_bi = bone_names.index(obj.parent_bone)
            # Reverse the bone-tail offset compensation from import
            bone = arm_obj.data.bones.get(obj.parent_bone)
            bone_len = bone.length if bone else 0
            loc = (obj.location[0], obj.location[1] + bone_len, obj.location[2])
            rf_pos = bl_to_rf_pos(Vector(loc))
        else:
            rf_pos = bl_to_rf_pos(obj.location)

        cspheres.append({
            'name': name, 'parent_bone': parent_bi,
            'pos': list(rf_pos), 'radius': obj.empty_display_size
        })
    return cspheres


def _gather_scene_prop_points(arm_obj, bone_names):
    """Read prop point empties from the scene back to RF format."""
    prop_points = []
    for obj in arm_obj.children:
        if obj.type != 'EMPTY' or not obj.get('rf_prop_point'):
            continue
        name = obj.name[3:] if obj.name.startswith('PP_') else obj.name

        # Get parent bone index
        parent_bi = -1
        if obj.parent_type == 'BONE' and obj.parent_bone:
            if obj.parent_bone in bone_names:
                parent_bi = bone_names.index(obj.parent_bone)
            bone = arm_obj.data.bones.get(obj.parent_bone)
            bone_len = bone.length if bone else 0
            loc = (obj.location[0], obj.location[1] + bone_len, obj.location[2])
            rf_pos = bl_to_rf_pos(Vector(loc))
        else:
            rf_pos = bl_to_rf_pos(obj.location)

        # Convert orientation back to RF
        rf_quat = bl_to_rf_quat(obj.rotation_quaternion) if obj.rotation_mode == 'QUATERNION' \
            else bl_to_rf_quat(obj.rotation_euler.to_quaternion())

        prop_points.append({
            'name': name, 'pos': list(rf_pos),
            'quat': list(rf_quat), 'parent_bone': parent_bi
        })
    return prop_points


def _export_v3c(mesh_obj, arm_obj, filepath, bones_data, cspheres_data=None, prop_points_data=None):
    """Export Blender mesh + armature as a V3C/V3M binary file."""
    is_character = bool(bones_data)
    bone_names = [b['name'] for b in bones_data] if bones_data else []
    if cspheres_data is None:
        cspheres_data = []
    if prop_points_data is None:
        prop_points_data = []

    # Gather mesh chunks
    chunks = _gather_chunks(mesh_obj, bone_names)
    if not chunks:
        raise ValueError("No mesh data to export")

    # Split chunks that exceed V3C limits
    # Alloc fields are uint16, storing byte sizes (nv*12 for positions)
    # Max verts per chunk: 65535 / 12 = 5461
    MAX_CHUNK_VERTS = 5400  # conservative limit for uint16 alloc fields
    split_chunks = []
    for chunk in chunks:
        if len(chunk['positions']) <= MAX_CHUNK_VERTS:
            split_chunks.append(chunk)
        else:
            # Split by iterating triangles and building sub-chunks
            src = chunk
            vert_remap = {}
            cur = {'material_index': src['material_index'], 'positions': [], 'normals': [],
                   'uvs': [], 'triangles': [], 'bone_links': [], 'face_flags': src.get('face_flags', 0)}

            for tri in src['triangles']:
                # Check if adding this triangle would overflow
                new_verts = sum(1 for idx in tri if idx not in vert_remap)
                if len(cur['positions']) + new_verts > MAX_CHUNK_VERTS and cur['triangles']:
                    split_chunks.append(cur)
                    vert_remap = {}
                    cur = {'material_index': src['material_index'], 'positions': [], 'normals': [],
                           'uvs': [], 'triangles': [], 'bone_links': [], 'face_flags': src.get('face_flags', 0)}

                new_tri = []
                for idx in tri:
                    if idx not in vert_remap:
                        new_idx = len(cur['positions'])
                        vert_remap[idx] = new_idx
                        cur['positions'].append(src['positions'][idx])
                        cur['normals'].append(src['normals'][idx])
                        cur['uvs'].append(src['uvs'][idx])
                        if src['bone_links']:
                            cur['bone_links'].append(src['bone_links'][idx])
                    new_tri.append(vert_remap[idx])
                cur['triangles'].append(tuple(new_tri))

            if cur['triangles']:
                split_chunks.append(cur)
    chunks = split_chunks

    # Build data block
    data_block, chunk_infos = _build_data_block(chunks, is_character, prop_points_data)

    # Compute totals
    total_verts = sum(len(c['positions']) for c in chunks)
    total_tris = sum(len(c['triangles']) for c in chunks)
    num_chunks = len(chunks)

    # Build material list from mesh materials
    mesh = mesh_obj.data
    materials = []
    mat_flags = []
    for mat in mesh.materials:
        if mat:
            # Get texture name — prefer the image connected to Base Color (if any)
            tex_name = mat.name + '.tga'
            img = None
            if mat.use_nodes and mat.node_tree:
                bsdf = mat.node_tree.nodes.get('Principled BSDF')
                if bsdf and 'Base Color' in bsdf.inputs:
                    for link in bsdf.inputs['Base Color'].links:
                        if link.from_node.type == 'TEX_IMAGE' and link.from_node.image:
                            img = link.from_node.image
                            break
                if not img:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            img = node.image
                            break
            if img:
                img_name = img.name
                if not img_name.lower().endswith('.tga'):
                    img_name = os.path.splitext(img_name)[0] + '.tga'
                tex_name = img_name
            materials.append(tex_name)
            # Material flags: base = 0x01 (stock/Redux standard), alpha adds 0x08
            flag = 0x01
            if hasattr(mat, 'blend_method') and mat.blend_method in ('CLIP', 'HASHED', 'BLEND'):
                flag |= 0x08
            elif hasattr(mat, 'surface_render_method') and mat.surface_render_method == 'DITHERED':
                flag |= 0x08
            elif mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.outputs.get('Alpha'):
                        if node.outputs['Alpha'].is_linked:
                            flag |= 0x08
                            break
            mat_flags.append(flag)
        else:
            materials.append('default.tga')
            mat_flags.append(0x01)

    # Build texture reference list — one per chunk, material index as ID
    tex_refs = []
    for ci, chunk in enumerate(chunks):
        mi = chunk['material_index']
        if mi < len(materials):
            tex_refs.append((mi, materials[mi]))
        else:
            tex_refs.append((0, 'default.tga'))

    # ── Compute bounding box ──
    all_pos = []
    for c in chunks:
        all_pos.extend(c['positions'])
    if all_pos:
        mins = [min(p[i] for p in all_pos) for i in range(3)]
        maxs = [max(p[i] for p in all_pos) for i in range(3)]
        radius = max(math.sqrt(sum(p[i]**2 for i in range(3))) for p in all_pos)
    else:
        mins = maxs = [0, 0, 0]
        radius = 0

    # ═══ ASSEMBLE FILE ═══
    output = bytearray()

    # ── File header (40 bytes) ──
    sig = _V3C_SIG if is_character else _V3M_SIG
    output += struct.pack("<I", sig)
    output += struct.pack("<I", _V3D_VER)
    output += struct.pack("<I", 1)          # num_submeshes
    output += struct.pack("<I", total_verts)
    output += struct.pack("<I", total_tris)
    output += struct.pack("<I", 0)          # unk0
    output += struct.pack("<I", len(materials))
    output += struct.pack("<I", 0)          # unk1
    output += struct.pack("<I", 0)          # unk2
    output += struct.pack("<i", len(cspheres_data))

    # ── SUBM section header (size=0, parsed inline) ──
    output += struct.pack("<I", _SEC_SUBM)
    output += struct.pack("<I", 0)

    # Submesh header: name(24) + parent(24) + version(4) + num_lods(4)
    sm_name = (mesh_obj.name[:23]).encode('ascii', errors='replace')
    output += sm_name + b'\x00' * (24 - len(sm_name))
    parent_name = b'\x00' * 24  # empty string (stock convention)
    output += parent_name
    output += struct.pack("<i", 7)  # submesh version
    output += struct.pack("<i", 1)  # num_lods (single LOD)

    # LOD distance
    output += struct.pack("<f", 0.0)

    # Bounding: offset(12) + radius(4) + bboxmin(12) + bboxmax(12)
    output += struct.pack("<3f", 0.0, 0.0, 0.0)  # offset = zero (stock/Redux convention)
    output += struct.pack("<f", radius)
    output += struct.pack("<3f", *mins)
    output += struct.pack("<3f", *maxs)

    # ── LOD 0 ──
    lod_flags = LOD_CHAR if is_character else 0
    output += struct.pack("<I", lod_flags)
    output += struct.pack("<i", total_verts)
    output += struct.pack("<H", num_chunks)
    output += struct.pack("<i", len(data_block))

    # Data block
    output += data_block

    # unk1 after data block (Redux and stock files write -1)
    output += struct.pack("<i", -1)

    # Chunk infos
    for ci_bytes in chunk_infos:
        output += ci_bytes

    # Prop points count + texture count
    output += struct.pack("<i", len(prop_points_data))
    output += struct.pack("<i", len(tex_refs))

    # Texture references: id(1 byte) + null-terminated name
    for tex_id, tex_name in tex_refs:
        output += struct.pack("<B", tex_id)
        output += tex_name.encode('ascii', errors='replace') + b'\x00'

    # ── Materials ──
    output += struct.pack("<i", len(materials))
    for mi, mat_name in enumerate(materials):
        # diffuse texture (32 bytes)
        mat_bytes = mat_name.encode('ascii', errors='replace')[:31]
        output += mat_bytes + b'\x00' * (32 - len(mat_bytes))
        # emissive(4) + specular(4) + glossiness(4) + reflection(4)
        output += struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)
        # reflection map (32 bytes)
        output += b'\x00' * 32
        # material flags
        output += struct.pack("<I", mat_flags[mi] if mi < len(mat_flags) else 0)

    # Unknown section after materials (Redux writes 1 entry: name + float)
    output += struct.pack("<i", 1)
    sm_name_unk = (mesh_obj.name[:23]).encode('ascii', errors='replace')
    output += sm_name_unk + b'\x00' * (24 - len(sm_name_unk))
    output += struct.pack("<f", 0.0)

    # ── CSPH sections (one per sphere) ──
    for cs in cspheres_data:
        output += struct.pack("<I", _SEC_CSPH)
        output += struct.pack("<I", 44)  # section size = one sphere
        cs_name = cs['name'].encode('ascii', errors='replace')[:23]
        output += cs_name + b'\x00' * (24 - len(cs_name))
        output += struct.pack("<i", cs['parent_bone'])
        output += struct.pack("<3f", *cs['pos'])
        output += struct.pack("<f", cs['radius'])

    # ── BONE section ──
    # Write the original V3C bone data (stored during import)
    # These MUST match what the RFA animations expect
    if bones_data and arm_obj:
        bone_data_size = 4 + len(bones_data) * (24 + 16 + 12 + 4)
        output += struct.pack("<I", _SEC_BONE)
        output += struct.pack("<I", bone_data_size)
        output += struct.pack("<I", len(bones_data))
        for b in bones_data:
            b_name = b['name'].encode('ascii', errors='replace')[:23]
            output += b_name + b'\x00' * (24 - len(b_name))
            output += struct.pack("<4f", *b['inv_bind_quat'])
            output += struct.pack("<3f", *b['inv_bind_pos'])
            output += struct.pack("<i", b['parent_index'])

    # ── End section marker ──
    output += struct.pack("<I", 0)  # section type = 0 (end)
    output += struct.pack("<I", 0)  # section size = 0

    with open(filepath, 'wb') as f:
        f.write(output)

    return total_verts, total_tris


# ═══════════════════════════════════════════════════════════════════════════════
#  BLENDER OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class RFCHAR_OT_ImportV3C(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.rf_v3c"
    bl_label = "Import RF Character Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".v3c"
    filter_glob: StringProperty(default="*.v3c;*.v3m", options={'HIDDEN'})

    import_armature: BoolProperty(name="Import Armature", default=True)
    import_cspheres: BoolProperty(name="Import Collision Spheres", default=True,
        description="Import hitbox collision spheres as empty objects parented to bones")
    import_prop_points: BoolProperty(name="Import Prop Points", default=True,
        description="Import attachment points (weapon, eye, flag, etc.) as empties")
    import_lod: EnumProperty(
        name="LOD",
        items=[('0', "LOD 0 (Highest)", ""), ('1', "LOD 1", ""), ('2', "LOD 2 (Lowest)", "")],
        default='0'
    )

    def execute(self, context):
        # Ensure we're in Object mode before importing (prevents crashes)
        try:
            if context.active_object and context.active_object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        # Deselect all to prevent context conflicts
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            pass

        try:
            v3c = parse_v3c(self.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"V3C parse failed: {e}")
            return {'CANCELLED'}

        arm_obj = None
        bone_names = []
        if self.import_armature and v3c['is_character'] and v3c['bones']:
            arm_obj, bone_names = _import_armature(v3c)
            # Store bone data on armature for RFA import
            arm_obj['rf_bones'] = json.dumps([{
                'name': b['name'],
                'inv_bind_quat': list(b['inv_bind_quat']),
                'inv_bind_pos': list(b['inv_bind_pos']),
                'parent_index': b['parent_index']
            } for b in v3c['bones']])

        # Import LOD 0 mesh only (user works with this)
        _import_mesh(v3c, arm_obj, bone_names, lod_index=0)

        # Store lower LOD data on armature for round-trip export
        sm = v3c['submeshes'][0]
        num_lods = len(sm['lods'])
        if arm_obj and num_lods > 1:
            arm_obj['rf_lod_distances'] = json.dumps([l['distance'] for l in sm['lods']])
            # Store lower LOD raw chunk data for verbatim re-export
            lod_data = []
            for li in range(1, num_lods):
                lod = sm['lods'][li]
                lod_data.append({
                    'distance': lod['distance'],
                    'chunks': [{
                        'positions': c['positions'],
                        'normals': c['normals'],
                        'uvs': c['uvs'],
                        'triangles': c['triangles'],
                        'bone_links': [list(w) + list(b) for w, b in c['bone_links']],
                        'texture_index': c['texture_index'],
                    } for c in lod['chunks']],
                })
            arm_obj['rf_lower_lods'] = json.dumps(lod_data)

        # Collision spheres
        cs_count = 0
        if self.import_cspheres and arm_obj and v3c.get('cspheres'):
            cs_objs = _import_cspheres(v3c, arm_obj, bone_names)
            cs_count = len(cs_objs)
            # Store for future V3C export
            arm_obj['rf_cspheres'] = json.dumps([{
                'name': cs['name'], 'parent_bone': cs['parent_bone'],
                'pos': list(cs['pos']), 'radius': cs['radius']
            } for cs in v3c['cspheres']])

        # Prop points (dummies)
        pp_count = 0
        if self.import_prop_points and arm_obj:
            pp_objs = _import_prop_points(v3c, arm_obj, bone_names)
            pp_count = len(pp_objs)
            # Store for V3C export
            sm = v3c['submeshes'][0]
            lod = sm['lods'][0]
            pps = lod.get('prop_points', [])
            if pps:
                arm_obj['rf_prop_points'] = json.dumps([{
                    'name': pp['name'], 'pos': list(pp['pos']),
                    'quat': list(pp['quat']), 'parent_bone': pp['parent_bone']
                } for pp in pps])

        sm = v3c['submeshes'][0]
        lod = sm['lods'][0]
        tv = sum(len(c['positions']) for c in lod['chunks'])
        tt = sum(len(c['triangles']) for c in lod['chunks'])

        # Look up required animations from stock database
        auto_loaded = 0
        if arm_obj:
            req_anims = _lookup_required_anims(self.filepath)
            if req_anims:
                arm_obj['rf_required_anims'] = json.dumps(req_anims)
            # Store source filename for export default
            arm_obj['rf_source_file'] = os.path.basename(self.filepath)

            # Auto-load required animations from master folder (if configured)
            try:
                if _get_auto_load_enabled(context) and req_anims:
                    master_abs = _get_master_folder(context)
                    if master_abs and os.path.isdir(master_abs):
                        auto_loaded = _auto_load_anims_from_folder(
                            arm_obj, req_anims, master_abs)
                        if auto_loaded > 0:
                            arm_obj['rf_anim_folder'] = master_abs
                            context.scene['rf_last_anim_folder'] = master_abs
                            # Leave armature in rest pose — no active animation
                            if arm_obj.animation_data:
                                arm_obj.animation_data.action = None
            except Exception as e:
                print(f"[RF Character] Auto-load skipped: {e}")

        info = f"Imported '{sm['name']}': {tv} verts, {tt} tris, {len(v3c['bones'])} bones, {num_lods} LOD{'s' if num_lods > 1 else ''}"
        if cs_count:
            info += f", {cs_count} hitboxes"
        if pp_count:
            info += f", {pp_count} props"
        if arm_obj and arm_obj.get('rf_required_anims'):
            req = json.loads(arm_obj['rf_required_anims'])
            info += f", {len(req)} required anims"
        if auto_loaded:
            info += f"  ·  auto-loaded {auto_loaded} anims"
        self.report({'INFO'}, info)

        # Track recent files (scene-level, up to 10)
        try:
            recent = list(context.scene.get('rf_recent_v3c', []))
            abs_path = os.path.abspath(self.filepath)
            if abs_path in recent:
                recent.remove(abs_path)
            recent.insert(0, abs_path)
            context.scene['rf_recent_v3c'] = recent[:10]
        except Exception:
            pass

        return {'FINISHED'}


class RFCHAR_OT_ImportRFA(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.rf_rfa"
    bl_label = "Import RF Animation (.rfa/.mvf)"
    bl_description = "Import one or more RF animation files. Hold Shift/Ctrl to multi-select"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".rfa"
    filter_glob: StringProperty(default="*.rfa;*.mvf", options={'HIDDEN'})

    files: CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: StringProperty(subtype='DIR_PATH')

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature found. Import V3C first.")
            return {'CANCELLED'}

        rf_bones_json = arm_obj.get('rf_bones')
        if not rf_bones_json:
            self.report({'ERROR'}, "No RF bone data found on armature. Import V3C first.")
            return {'CANCELLED'}
        bones_data = json.loads(rf_bones_json)
        for b in bones_data:
            b['inv_bind_quat'] = tuple(b['inv_bind_quat'])
            b['inv_bind_pos'] = tuple(b['inv_bind_pos'])

        # Build file list (single or multi-select)
        if self.files:
            filepaths = [os.path.join(self.directory, f.name) for f in self.files if f.name]
        else:
            filepaths = [self.filepath]

        imported = 0
        failed = 0
        for fp in filepaths:
            anim_name = os.path.splitext(os.path.basename(fp))[0]
            try:
                rfa = parse_rfa(fp)
                _import_rfa(rfa, arm_obj, bones_data, anim_name)
                # Always stash to NLA so animations persist through Rest Pose
                action = arm_obj.animation_data.action
                if action:
                    track = arm_obj.animation_data.nla_tracks.new()
                    track.name = action.name
                    start = int(action.frame_range[0])
                    strip = track.strips.new(action.name, start, action)
                    track.mute = True
                    arm_obj.animation_data.action = None
                imported += 1
            except Exception as e:
                self.report({'WARNING'}, f"Failed '{anim_name}': {e}")
                failed += 1

        # Leave armature in rest pose after loading — don't auto-activate any anim
        if imported > 0 and arm_obj.animation_data:
            try:
                arm_obj.animation_data.action = None
                for track in arm_obj.animation_data.nla_tracks:
                    track.mute = True
                for pb in arm_obj.pose.bones:
                    pb.rotation_quaternion = (1, 0, 0, 0)
                    pb.location = (0, 0, 0)
                    pb.scale = (1, 1, 1)
                context.scene.frame_set(context.scene.frame_current)
            except Exception:
                pass

        if imported == 1:
            self.report({'INFO'}, f"Imported '{os.path.splitext(os.path.basename(filepaths[0]))[0]}'")
        else:
            msg = f"Imported {imported} animation{'s' if imported != 1 else ''}"
            if failed:
                msg += f", {failed} failed"
            self.report({'INFO'}, msg)
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════════
#  PANEL: ANIMATION FILE LIST ITEM
# ═══════════════════════════════════════════════════════════════════════════════

class RFCHAR_AnimFileItem(bpy.types.PropertyGroup):
    name: StringProperty(name="Name")
    filepath: StringProperty(name="Path")
    selected: BoolProperty(name="Import", default=True)


class RFCHAR_UL_AnimList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_property, index):
        row = layout.row(align=True)
        row.prop(item, "selected", text="")
        row.label(text=item.name)


# ═══════════════════════════════════════════════════════════════════════════════
#  PANEL: OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_scan_dir(self, context):
    """Called when animation directory changes — auto-scan for files."""
    scn = context.scene
    anim_dir = bpy.path.abspath(scn.rfchar_anim_dir)
    scn.rfchar_anim_files.clear()
    if not anim_dir or not os.path.isdir(anim_dir):
        return
    for fn in sorted(os.listdir(anim_dir)):
        if fn.lower().endswith(('.rfa', '.mvf')):
            item = scn.rfchar_anim_files.add()
            item.name = os.path.splitext(fn)[0]
            item.filepath = os.path.join(anim_dir, fn)
            item.selected = True


class RFCHAR_OT_BrowseAnimFolder(bpy.types.Operator):
    bl_idname = "rfchar.browse_anim_folder"
    bl_label = "Batch Import Anims"
    bl_description = "Browse for a folder of .rfa/.mvf animation files to batch import"

    directory: StringProperty(subtype='DIR_PATH')

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        scn = context.scene
        scn.rfchar_anim_dir = self.directory
        # Auto-scan is triggered by the property update callback
        return {'FINISHED'}


class RFCHAR_OT_ImportSelectedAnims(bpy.types.Operator):
    bl_idname = "rfchar.import_selected_anims"
    bl_label = "Import Selected"
    bl_description = "Import checked animations onto the active armature"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scn = context.scene
        arm_obj = _get_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "Select an armature or parented mesh")
            return {'CANCELLED'}

        rf_bones_json = arm_obj.get('rf_bones')
        if not rf_bones_json:
            self.report({'ERROR'}, "No RF bone data. Import V3C first.")
            return {'CANCELLED'}
        bones_data = json.loads(rf_bones_json)
        for b in bones_data:
            b['inv_bind_quat'] = tuple(b['inv_bind_quat'])
            b['inv_bind_pos'] = tuple(b['inv_bind_pos'])

        imported = 0
        failed = 0
        for item in scn.rfchar_anim_files:
            if not item.selected:
                continue
            try:
                rfa = parse_rfa(item.filepath)
                _import_rfa(rfa, arm_obj, bones_data, item.name)
                action = arm_obj.animation_data.action
                if action:
                    track = arm_obj.animation_data.nla_tracks.new()
                    track.name = action.name
                    start = int(action.frame_range[0])
                    strip = track.strips.new(action.name, start, action)
                    track.mute = True
                    arm_obj.animation_data.action = None
                imported += 1
            except Exception as e:
                self.report({'WARNING'}, f"Failed '{item.name}': {e}")
                failed += 1

        # Leave armature in rest pose after loading — don't auto-activate any anim
        if imported > 0 and arm_obj.animation_data:
            try:
                arm_obj.animation_data.action = None
                for track in arm_obj.animation_data.nla_tracks:
                    track.mute = True
                for pb in arm_obj.pose.bones:
                    pb.rotation_quaternion = (1, 0, 0, 0)
                    pb.location = (0, 0, 0)
                    pb.scale = (1, 1, 1)
                context.scene.frame_set(context.scene.frame_current)
            except Exception:
                pass

        msg = f"Imported {imported} animation{'s' if imported != 1 else ''}"
        if failed:
            msg += f", {failed} failed"
        self.report({'INFO'}, msg)

        # Remember folder for future auto-loads
        if imported > 0:
            try:
                first_fp = next((i.filepath for i in scn.rfchar_anim_files if i.selected), None)
                if first_fp:
                    folder = os.path.dirname(first_fp)
                    if os.path.isdir(folder):
                        arm_obj['rf_anim_folder'] = folder
                        context.scene['rf_last_anim_folder'] = folder
            except Exception:
                pass

        return {'FINISHED'}


class RFCHAR_OT_SelectAllAnims(bpy.types.Operator):
    bl_idname = "rfchar.select_all_anims"
    bl_label = "All"
    bl_description = "Select all"

    def execute(self, context):
        for item in context.scene.rfchar_anim_files:
            item.selected = True
        return {'FINISHED'}


class RFCHAR_OT_DeselectAllAnims(bpy.types.Operator):
    bl_idname = "rfchar.deselect_all_anims"
    bl_label = "None"
    bl_description = "Deselect all"

    def execute(self, context):
        for item in context.scene.rfchar_anim_files:
            item.selected = False
        return {'FINISHED'}


class RFCHAR_OT_ClearAnimList(bpy.types.Operator):
    bl_idname = "rfchar.clear_anim_list"
    bl_label = "Clear"
    bl_description = "Clear the file list"

    def execute(self, context):
        context.scene.rfchar_anim_files.clear()
        context.scene.rfchar_anim_dir = ""
        return {'FINISHED'}


class RFCHAR_OT_SetActiveAction(bpy.types.Operator):
    bl_idname = "rfchar.set_active_action"
    bl_label = "Set Active Animation"
    bl_description = "Switch to this animation"

    action_name: StringProperty()

    def execute(self, context):
        arm_obj = _get_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "Select an armature")
            return {'CANCELLED'}

        action = bpy.data.actions.get(self.action_name)
        if not action:
            self.report({'ERROR'}, f"Action '{self.action_name}' not found")
            return {'CANCELLED'}

        if not arm_obj.animation_data:
            arm_obj.animation_data_create()
        arm_obj.animation_data.action = action

        if hasattr(arm_obj.animation_data, 'action_slot') and hasattr(action, 'slots') and len(action.slots) > 0:
            arm_obj.animation_data.action_slot = action.slots[0]

        fr = action.frame_range
        context.scene.frame_start = int(fr[0])
        context.scene.frame_end = int(fr[1])
        context.scene.frame_set(int(fr[0]))
        return {'FINISHED'}


def _import_dae(filepath):
    """Import a Collada .dae file (for Blender 5.0+ which removed built-in Collada support)."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Handle namespace
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    def find(elem, tag):
        return elem.find(ns + tag)

    def findall(elem, tag):
        return elem.findall(ns + tag)

    def find_path(elem, path):
        parts = path.split('/')
        current = elem
        for p in parts:
            current = find(current, p)
            if current is None:
                return None
        return current

    # Check up axis
    up_axis_elem = find(root, 'asset')
    up_axis = 'Y_UP'
    if up_axis_elem is not None:
        ua = find(up_axis_elem, 'up_axis')
        if ua is not None and ua.text:
            up_axis = ua.text.strip()

    def apply_up_axis(x, y, z):
        if up_axis == 'Z_UP':
            return (x, -z, y)
        return (x, y, z)

    # Parse sources (float arrays) by id
    lib_geom = find(root, 'library_geometries')
    if lib_geom is None:
        raise ValueError("No geometry found in DAE file")

    created_objects = []

    for geom in findall(lib_geom, 'geometry'):
        geom_name = geom.get('name', geom.get('id', 'DAE_Mesh'))
        mesh_elem = find(geom, 'mesh')
        if mesh_elem is None:
            continue

        # Parse all sources
        sources = {}
        for source in findall(mesh_elem, 'source'):
            sid = source.get('id', '')
            fa = find(source, 'float_array')
            if fa is not None and fa.text:
                floats = [float(v) for v in fa.text.split()]
                # Determine stride from accessor
                tech = find(source, 'technique_common')
                stride = 3
                if tech is not None:
                    acc = find(tech, 'accessor')
                    if acc is not None:
                        stride = int(acc.get('stride', '3'))
                sources[sid] = (floats, stride)

        # Parse vertices element (maps vertex semantic to position source)
        vert_elem = find(mesh_elem, 'vertices')
        vert_id = vert_elem.get('id', '') if vert_elem is not None else ''
        vert_source_id = ''
        if vert_elem is not None:
            for inp in findall(vert_elem, 'input'):
                if inp.get('semantic') == 'POSITION':
                    vert_source_id = inp.get('source', '').lstrip('#')

        # Process triangles and polylist elements
        all_verts = []
        all_faces = []
        all_uvs = []
        all_normals = []

        for prim_tag in ('triangles', 'polylist', 'polygons'):
            for prim in findall(mesh_elem, prim_tag):
                # Parse inputs
                inputs = []
                max_offset = 0
                for inp in findall(prim, 'input'):
                    semantic = inp.get('semantic', '')
                    source_id = inp.get('source', '').lstrip('#')
                    offset = int(inp.get('offset', '0'))
                    # Resolve VERTEX reference
                    if semantic == 'VERTEX':
                        source_id = vert_source_id
                        semantic = 'POSITION'
                    inputs.append((semantic, source_id, offset))
                    max_offset = max(max_offset, offset)
                stride = max_offset + 1

                # Parse index data
                p_elem = find(prim, 'p')
                if p_elem is None or not p_elem.text:
                    continue
                indices = [int(v) for v in p_elem.text.split()]

                # Parse vcount for polylist
                if prim_tag == 'polylist':
                    vc_elem = find(prim, 'vcount')
                    if vc_elem is not None and vc_elem.text:
                        vcounts = [int(v) for v in vc_elem.text.split()]
                    else:
                        continue
                else:
                    # Triangles: all faces have 3 verts
                    nfaces = int(prim.get('count', '0'))
                    vcounts = [3] * nfaces

                # Find source references
                pos_source = norm_source = uv_source = None
                pos_offset = norm_offset = uv_offset = 0
                for sem, sid, off in inputs:
                    if sem == 'POSITION' and sid in sources:
                        pos_source, _ = sources[sid]
                        pos_offset = off
                    elif sem == 'NORMAL' and sid in sources:
                        norm_source, _ = sources[sid]
                        norm_offset = off
                    elif sem == 'TEXCOORD' and sid in sources:
                        uv_source, uv_stride = sources[sid]
                        uv_offset = off

                if pos_source is None:
                    continue

                # Build mesh data
                vert_map = {}
                idx_ptr = 0

                for vcount in vcounts:
                    face_indices = []
                    for vi in range(vcount):
                        # Read indices for this vertex
                        pi = indices[idx_ptr + pos_offset]
                        ni = indices[idx_ptr + norm_offset] if norm_source else 0
                        ti = indices[idx_ptr + uv_offset] if uv_source else 0
                        idx_ptr += stride

                        key = (pi, ni, ti)
                        if key not in vert_map:
                            local_idx = len(all_verts)
                            vert_map[key] = local_idx

                            px = pos_source[pi * 3]
                            py = pos_source[pi * 3 + 1]
                            pz = pos_source[pi * 3 + 2]
                            all_verts.append(apply_up_axis(px, py, pz))

                            if norm_source and ni * 3 + 2 < len(norm_source):
                                nx = norm_source[ni * 3]
                                ny = norm_source[ni * 3 + 1]
                                nz = norm_source[ni * 3 + 2]
                                all_normals.append(apply_up_axis(nx, ny, nz))

                            if uv_source and ti * 2 + 1 < len(uv_source):
                                all_uvs.append((uv_source[ti * 2], uv_source[ti * 2 + 1]))

                        face_indices.append(vert_map[key])

                    # Triangulate n-gons
                    for fi in range(1, len(face_indices) - 1):
                        all_faces.append((face_indices[0], face_indices[fi], face_indices[fi + 1]))

        if not all_verts or not all_faces:
            continue

        # Create Blender mesh
        mesh = bpy.data.meshes.new(geom_name)
        mesh.from_pydata(all_verts, [], all_faces)
        mesh.validate(clean_customdata=False)

        # UVs
        if all_uvs and len(all_uvs) == len(all_verts):
            uv_layer = mesh.uv_layers.new(name="UVMap")
            for li, loop in enumerate(mesh.loops):
                vi = loop.vertex_index
                if vi < len(all_uvs):
                    uv_layer.data[li].uv = all_uvs[vi]

        # Normals
        if all_normals and len(all_normals) == len(all_verts):
            try:
                normals_list = [all_normals[l.vertex_index] if l.vertex_index < len(all_normals)
                                else (0, 0, 1) for l in mesh.loops]
                if hasattr(mesh, 'normals_split_custom_set'):
                    mesh.normals_split_custom_set(normals_list)
            except Exception:
                pass

        mesh.update()

        obj = bpy.data.objects.new(geom_name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        created_objects.append(obj)

    if not created_objects:
        raise ValueError("No mesh geometry found in DAE file")

    return created_objects


def _import_glm(filepath):
    """Import a Ghoul 2 .glm mesh file (Jedi Academy/Jedi Outcast) — mesh only, no skeleton required."""
    with open(filepath, 'rb') as f:
        data = f.read()

    # Header: ident(4) + version(4) + name(64) + animName(64) + animIndex(4) +
    #         numBones(4) + numLODs(4) + ofsLODs(4) + numSurfaces(4) +
    #         ofsSurfHierarchy(4) + ofsEnd(4) = 164 bytes
    HEADER_SIZE = 164

    ident = data[0:4]
    if ident != b'2LGM':
        raise ValueError(f"Not a GLM file (ident={ident})")

    num_bones = struct.unpack_from("<i", data, 140)[0]
    num_lods = struct.unpack_from("<i", data, 144)[0]
    ofs_lods = struct.unpack_from("<i", data, 148)[0]
    num_surfaces = struct.unpack_from("<i", data, 152)[0]
    ofs_end = struct.unpack_from("<i", data, 160)[0]

    # Surface data offsets table (starts right after header, one int per surface)
    # Values are relative to HEADER_SIZE (byte 164)
    sd_offsets = []
    for si in range(num_surfaces):
        sd_offsets.append(struct.unpack_from("<i", data, HEADER_SIZE + si * 4)[0])

    # Read surface hierarchy: name(64) + flags(4) + shader(64) + shaderIndex(4) + parent(4) + numChildren(4) + children(4*n)
    surf_names = []
    surf_shaders = []
    surf_flags = []
    for si in range(num_surfaces):
        pos = HEADER_SIZE + sd_offsets[si]
        s_name = data[pos:pos+64].split(b'\x00')[0].decode('ascii', errors='replace')
        s_flags = struct.unpack_from("<I", data, pos + 64)[0]
        s_shader = data[pos+68:pos+68+64].split(b'\x00')[0].decode('ascii', errors='replace')
        surf_names.append(s_name)
        surf_shaders.append(s_shader)
        surf_flags.append(s_flags)

    # Try to auto-load .skin file for texture mapping (surface_name → texture_path)
    skin_textures = {}
    glm_dir = os.path.dirname(filepath)

    # Search for any .skin file in the directory
    skin_files = []
    try:
        for f in os.listdir(glm_dir):
            if f.lower().endswith('.skin'):
                skin_files.append(f)
    except Exception:
        pass

    # Prefer model_default.skin, then any other
    skin_files.sort(key=lambda f: (0 if 'default' in f.lower() else 1, f))

    for skin_name in skin_files:
        skin_path = os.path.join(glm_dir, skin_name)
        try:
            with open(skin_path, 'r') as sf:
                for line in sf:
                    line = line.strip()
                    if ',' in line and not line.startswith('//'):
                        parts = line.split(',', 1)
                        surf = parts[0].strip()
                        tex = parts[1].strip()
                        if surf and tex:
                            skin_textures[surf] = tex
        except Exception:
            pass
        if skin_textures:
            break

    # Build a map of available texture files in the GLM directory
    dir_textures = {}  # lowercase basename (no ext) → full path
    try:
        for f in os.listdir(glm_dir):
            fl = f.lower()
            if fl.endswith(('.jpg', '.jpeg', '.png', '.tga', '.dds', '.bmp')):
                base = os.path.splitext(f)[0].lower()
                dir_textures[base] = os.path.join(glm_dir, f)
    except Exception:
        pass

    # Read LOD 0 surfaces
    lod_start = ofs_lods
    lod_ofs_end = struct.unpack_from("<i", data, lod_start)[0]

    # Surface offsets table: numSurfaces ints, right after ofsEnd
    # Offsets are relative to (lod_start + 4), i.e. after the ofsEnd field
    surf_offsets = []
    for si in range(num_surfaces):
        soff = struct.unpack_from("<i", data, lod_start + 4 + si * 4)[0]
        surf_offsets.append(soff)

    created_objects = []

    for si in range(num_surfaces):
      try:
        # Surface position: lod_start + 4 + surf_offsets[si]
        surf_start = lod_start + 4 + surf_offsets[si]

        # Bounds check
        if surf_start + 40 > len(data):
            continue

        # Surface header: ident(4) + index(4) + ofsHeader(4) + numVerts(4) + ofsVerts(4) +
        #   numTris(4) + ofsTris(4) + numBoneRefs(4) + ofsBoneRefs(4) + ofsEnd(4) = 40 bytes
        _ident = struct.unpack_from("<i", data, surf_start)[0]
        s_index = struct.unpack_from("<i", data, surf_start + 4)[0]
        s_num_verts = struct.unpack_from("<i", data, surf_start + 12)[0]
        s_ofs_verts = struct.unpack_from("<i", data, surf_start + 16)[0]
        s_num_tris = struct.unpack_from("<i", data, surf_start + 20)[0]
        s_ofs_tris = struct.unpack_from("<i", data, surf_start + 24)[0]
        s_ofs_end = struct.unpack_from("<i", data, surf_start + 36)[0]

        # Skip empty surfaces, tags (*prefix), stubs (3 verts), and placeholders
        s_name = surf_names[si] if si < len(surf_names) else f"surface_{si}"
        if s_num_verts <= 3 or s_num_tris <= 0:
            continue
        if s_name.startswith('*'):
            continue
        if 'stupidtriangle' in s_name.lower():
            continue

        # Vertices: normal(3f) + position(3f) + packed(I) + weights(4B) = 32 bytes each
        verts = []
        norms = []
        v_off = surf_start + s_ofs_verts
        if v_off + s_num_verts * 32 > len(data):
            continue  # vertex data out of bounds
        for vi in range(s_num_verts):
            nx, ny, nz = struct.unpack_from("<3f", data, v_off); v_off += 12
            px, py, pz = struct.unpack_from("<3f", data, v_off); v_off += 12
            v_off += 8  # skip packed bone data + weight bytes
            verts.append((px, py, pz))
            norms.append((nx, ny, nz))

        # UVs follow all vertices: 2f per vert
        uvs = []
        if v_off + s_num_verts * 8 <= len(data):
            for vi in range(s_num_verts):
                u, v = struct.unpack_from("<2f", data, v_off); v_off += 8
                uvs.append((u, v))

        # Triangles: 3 ints each
        faces = []
        t_off = surf_start + s_ofs_tris
        if t_off + s_num_tris * 12 <= len(data):
            for ti in range(s_num_tris):
                i0, i1, i2 = struct.unpack_from("<3i", data, t_off); t_off += 12
                faces.append((i2, i1, i0))  # flip winding CW→CCW

        if not verts or not faces:
            continue

        # Create Blender mesh
        mesh = bpy.data.meshes.new(s_name)
        mesh.from_pydata(verts, [], faces)
        mesh.validate(clean_customdata=False)

        # UVs
        if uvs:
            uv_layer = mesh.uv_layers.new(name="UVMap")
            for li, loop in enumerate(mesh.loops):
                vi = loop.vertex_index
                if vi < len(uvs):
                    u, v = uvs[vi]
                    uv_layer.data[li].uv = (u, 1.0 - v)

        # Normals imported via from_pydata vertex data (Blender auto-calculates)

        # Material: prefer skin file texture, then shader, then directory texture match
        shader = ''
        if s_name in skin_textures:
            shader = skin_textures[s_name]
        elif si < len(surf_shaders) and surf_shaders[si] and surf_shaders[si] != '[nomaterial]':
            shader = surf_shaders[si]

        # Resolve texture path
        tex_path = None
        tex_display_name = s_name

        if shader:
            tex_display_name = os.path.splitext(os.path.basename(shader))[0]
            # Check if the shader path itself exists relative to GLM dir
            shader_basename = os.path.basename(shader)
            shader_no_ext = os.path.splitext(shader_basename)[0].lower()
            if shader_no_ext in dir_textures:
                tex_path = dir_textures[shader_no_ext]
            else:
                # Try full shader path relative to GLM dir
                for ext in ('.jpg', '.tga', '.png', '.dds', ''):
                    check = os.path.join(glm_dir, shader_basename if not ext else os.path.splitext(shader_basename)[0] + ext)
                    if os.path.exists(check):
                        tex_path = check
                        break

        # If no texture found from shader/skin, try matching surface name to files in directory
        if not tex_path:
            s_lower = s_name.lower()
            if s_lower in dir_textures:
                tex_path = dir_textures[s_lower]
                tex_display_name = os.path.splitext(os.path.basename(tex_path))[0]

        # Create material
        mat = bpy.data.materials.get(tex_display_name)
        if not mat:
            mat = bpy.data.materials.new(tex_display_name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            bsdf = nodes.get('Principled BSDF')
            if not bsdf:
                bsdf = nodes.new('ShaderNodeBsdfPrincipled')

            if tex_path or shader:
                tex_node = nodes.new('ShaderNodeTexImage')
                tex_node.location = (-300, 300)
                img = None
                if tex_path:
                    try:
                        img = bpy.data.images.load(tex_path)
                    except Exception:
                        pass
                if not img:
                    # Try finding in Blender's loaded images
                    for ext in ('.jpg', '.tga', '.png', '.dds', ''):
                        check_name = tex_display_name + ext if ext else tex_display_name
                        img = bpy.data.images.get(check_name)
                        if img:
                            break
                if not img:
                    fname = tex_display_name + '.jpg'
                    img = bpy.data.images.new(fname, width=1, height=1)
                    img.filepath = fname
                    img.source = 'FILE'
                tex_node.image = img
                mat.node_tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            mat.use_backface_culling = False
        mesh.materials.append(mat)

        mesh.update()

        obj = bpy.data.objects.new(s_name, mesh)
        bpy.context.collection.objects.link(obj)
        created_objects.append(obj)
      except Exception:
        continue

    if not created_objects:
        raise ValueError("No mesh surfaces found in GLM file")

    return created_objects


class RFCHAR_OT_ImportCustomMesh(bpy.types.Operator, ImportHelper):
    bl_idname = "rfchar.import_custom_mesh"
    bl_label = "Import Custom Mesh"
    bl_description = "Import a mesh file into the scene for binding to the RF armature"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".obj"
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(
        default="*.obj;*.fbx;*.gltf;*.glb;*.stl;*.ply;*.dae;*.glm",
        options={'HIDDEN'}
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        filepath = self.filepath
        if not filepath or not os.path.isfile(filepath):
            self.report({'ERROR'}, "No file selected")
            return {'CANCELLED'}
        ext = os.path.splitext(filepath)[1].lower()

        try:
            if ext == '.obj':
                if hasattr(bpy.ops.wm, 'obj_import'):
                    bpy.ops.wm.obj_import(filepath=filepath)
                else:
                    bpy.ops.import_scene.obj(filepath=filepath)
            elif ext == '.fbx':
                bpy.ops.import_scene.fbx(filepath=filepath)
            elif ext in ('.gltf', '.glb'):
                bpy.ops.import_scene.gltf(filepath=filepath)
            elif ext == '.stl':
                if hasattr(bpy.ops.wm, 'stl_import'):
                    bpy.ops.wm.stl_import(filepath=filepath)
                else:
                    bpy.ops.import_mesh.stl(filepath=filepath)
            elif ext == '.ply':
                if hasattr(bpy.ops.wm, 'ply_import'):
                    bpy.ops.wm.ply_import(filepath=filepath)
                else:
                    bpy.ops.import_mesh.ply(filepath=filepath)
            elif ext == '.dae':
                # Try Blender's built-in first (Blender < 5.0), fall back to our parser
                try:
                    bpy.ops.wm.collada_import(filepath=filepath)
                except (AttributeError, RuntimeError):
                    _import_dae(filepath)
            elif ext == '.glm':
                _import_glm(filepath)
            else:
                self.report({'ERROR'}, f"Unsupported format: {ext}")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Imported '{os.path.basename(filepath)}'")
        return {'FINISHED'}


class RFCHAR_OT_BindToArmature(bpy.types.Operator):
    bl_idname = "rfchar.bind_to_armature"
    bl_label = "Bind with Auto Weights"
    bl_description = (
        "Parent your custom mesh to the RF armature using Blender's automatic bone heat weights. "
        "Quick but approximate — best for meshes that don't closely match the original character shape, "
        "or when the original RF mesh isn't available. May need weight painting cleanup afterward. "
        "Select your custom mesh first."
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature found. Import a V3C first.")
            return {'CANCELLED'}

        # Find the custom mesh: active object if it's a mesh, otherwise any selected mesh
        # that isn't the RF original
        obj = None
        if context.active_object and context.active_object.type == 'MESH':
            obj = context.active_object
        if not obj:
            for o in context.selected_objects:
                if o.type == 'MESH' and not o.get('rf_original_mesh'):
                    obj = o
                    break

        if not obj:
            self.report({'ERROR'}, "Select your custom mesh first")
            return {'CANCELLED'}

        if obj.get('rf_original_mesh'):
            self.report({'WARNING'}, "That's the RF original mesh. Select your custom mesh instead.")
            return {'CANCELLED'}

        if obj.parent == arm_obj and any(m.type == 'ARMATURE' for m in obj.modifiers):
            self.report({'WARNING'}, f"'{obj.name}' is already bound to this armature")
            return {'CANCELLED'}

        # Clear existing parent if any
        if obj.parent:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj

        self.report({'INFO'}, f"Bound '{obj.name}' to '{arm_obj.name}' with automatic weights")
        return {'FINISHED'}


class RFCHAR_OT_TransferWeights(bpy.types.Operator):
    bl_idname = "rfchar.transfer_weights"
    bl_label = "Copy Weights from RF Mesh"
    bl_description = (
        "Copy vertex weights from the original RF character mesh onto your custom mesh using "
        "nearest-surface interpolation. More accurate than auto-weights and preserves RF's exact "
        "bone weighting — best when your mesh overlaps the original character's shape. "
        "Requires the original RF mesh to be in the scene (don't delete it after import)."
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        target = context.active_object
        if not target or target.type != 'MESH':
            self.report({'ERROR'}, "Select your custom mesh first")
            return {'CANCELLED'}

        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature found")
            return {'CANCELLED'}

        source = None
        for c in arm_obj.children:
            if c.type == 'MESH' and c != target and c.get('rf_original_mesh'):
                source = c
                break

        if not source:
            self.report({'ERROR'}, "No RF mesh found to transfer weights from")
            return {'CANCELLED'}

        if target.parent != arm_obj:
            target.parent = arm_obj
            mod = target.modifiers.get('Armature')
            if not mod:
                mod = target.modifiers.new(name='Armature', type='ARMATURE')
            mod.object = arm_obj

        for vg in source.vertex_groups:
            if vg.name not in target.vertex_groups:
                target.vertex_groups.new(name=vg.name)

        dt = target.modifiers.new(name='WeightTransfer', type='DATA_TRANSFER')
        dt.object = source
        dt.use_vert_data = True
        dt.data_types_verts = {'VGROUP_WEIGHTS'}
        dt.vert_mapping = 'POLYINTERP_NEAREST'

        bpy.ops.object.select_all(action='DESELECT')
        target.select_set(True)
        context.view_layer.objects.active = target
        bpy.ops.object.modifier_apply(modifier='WeightTransfer')

        self.report({'INFO'}, f"Transferred weights from '{source.name}' to '{target.name}'")
        return {'FINISHED'}


class RFCHAR_OT_ToggleRFMesh(bpy.types.Operator):
    bl_idname = "rfchar.toggle_rf_mesh"
    bl_label = "Toggle RF Mesh"
    bl_description = "Show/hide the original RF mesh (LOD 0 only, does not affect custom meshes)"

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            return {'CANCELLED'}

        rf_meshes = [c for c in arm_obj.children
                     if c.type == 'MESH' and c.get('rf_original_mesh')
                     and c.get('rf_lod_level', 0) == 0]
        if not rf_meshes:
            self.report({'INFO'}, "No original RF mesh found")
            return {'CANCELLED'}

        for c in rf_meshes:
            c.hide_viewport = not c.hide_viewport
            c.hide_render = c.hide_viewport
        state = "hidden" if rf_meshes[0].hide_viewport else "visible"
        self.report({'INFO'}, f"RF mesh: {state}")
        return {'FINISHED'}


class RFCHAR_OT_ToggleHitboxes(bpy.types.Operator):
    bl_idname = "rfchar.toggle_hitboxes"
    bl_label = "Toggle Hitboxes"
    bl_description = "Show/hide collision sphere hitboxes"

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            return {'CANCELLED'}

        cs_objs = [c for c in arm_obj.children if c.type == 'EMPTY' and c.name.startswith('CS_')]
        if not cs_objs:
            self.report({'INFO'}, "No hitboxes found")
            return {'CANCELLED'}

        # Toggle based on first one's state
        new_state = not cs_objs[0].hide_viewport
        for c in cs_objs:
            c.hide_viewport = new_state
            c.hide_render = new_state

        state = "hidden" if new_state else "visible"
        self.report({'INFO'}, f"{len(cs_objs)} hitboxes: {state}")
        return {'FINISHED'}


class RFCHAR_OT_TogglePropPoints(bpy.types.Operator):
    bl_idname = "rfchar.toggle_prop_points"
    bl_label = "Toggle Prop Points"
    bl_description = "Show/hide prop point empties (weapon, eye, flag, etc.)"

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            return {'CANCELLED'}

        pp_objs = [c for c in arm_obj.children if c.type == 'EMPTY' and c.get('rf_prop_point')]
        if not pp_objs:
            self.report({'INFO'}, "No prop points found")
            return {'CANCELLED'}

        new_state = not pp_objs[0].hide_viewport
        for c in pp_objs:
            c.hide_viewport = new_state
            c.hide_render = new_state

        state = "hidden" if new_state else "visible"
        self.report({'INFO'}, f"{len(pp_objs)} prop points: {state}")
        return {'FINISHED'}


class RFCHAR_OT_ToggleLODs(bpy.types.Operator):
    bl_idname = "rfchar.toggle_lods"
    bl_label = "Toggle LODs"
    bl_description = "Show/hide lower detail LOD meshes. First click imports them from stored data"

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            return {'CANCELLED'}

        # Check if LOD meshes already exist
        lod_meshes = [c for c in arm_obj.children
                      if c.type == 'MESH' and c.get('rf_original_mesh') and c.get('rf_lod_level', 0) > 0]

        if lod_meshes:
            # Toggle visibility
            new_state = not lod_meshes[0].hide_viewport
            for c in lod_meshes:
                c.hide_viewport = new_state
                c.hide_render = new_state
            state = "hidden" if new_state else "visible"
            self.report({'INFO'}, f"{len(lod_meshes)} LOD meshes: {state}")
        else:
            # First time: create LOD meshes from stored data
            lod_json = arm_obj.get('rf_lower_lods')
            if not lod_json:
                self.report({'INFO'}, "No LOD data stored (single-LOD model)")
                return {'CANCELLED'}

            stored_lods = json.loads(lod_json)
            bones_json = arm_obj.get('rf_bones', '[]')
            bones_data = json.loads(bones_json)
            bone_names = [b['name'] for b in bones_data]
            num_bones = len(bone_names)
            sm_name = ""
            for c in arm_obj.children:
                if c.type == 'MESH' and c.get('rf_lod_level', 0) == 0:
                    sm_name = c.name
                    break

            created = 0
            for li, lod_info in enumerate(stored_lods):
                lod_level = li + 1
                mesh_name = f"{sm_name}_LOD{lod_level}"

                all_verts = []
                all_faces = []
                all_uvs = []
                all_bone_links = []
                all_mat_indices = []
                vert_offset = 0

                for chunk in lod_info['chunks']:
                    for p in chunk['positions']:
                        all_verts.append(rf_to_bl_pos(*p))
                    for u, v in chunk['uvs']:
                        all_uvs.append((u, 1.0 - v))
                    for tri in chunk['triangles']:
                        all_faces.append((tri[0]+vert_offset, tri[2]+vert_offset, tri[1]+vert_offset))
                        all_mat_indices.append(chunk['texture_index'])
                    for bl_entry in chunk['bone_links']:
                        w = tuple(bl_entry[:4])
                        b = tuple(bl_entry[4:])
                        cj = tuple(bi if bi < num_bones else 0 for bi in b)
                        cw = tuple(wi if bi < num_bones else 0 for wi, bi in zip(w, b))
                        all_bone_links.append((cw, cj))
                    vert_offset += len(chunk['positions'])

                mesh = bpy.data.meshes.new(mesh_name)
                mesh.from_pydata([tuple(v) for v in all_verts], [], all_faces)

                # UVs
                if all_uvs:
                    uv_layer = mesh.uv_layers.new(name="UVMap")
                    for loop_i, loop in enumerate(mesh.loops):
                        vi = loop.vertex_index
                        if vi < len(all_uvs):
                            uv_layer.data[loop_i].uv = all_uvs[vi]

                mesh.update()

                obj = bpy.data.objects.new(mesh_name, mesh)
                obj['rf_original_mesh'] = True
                obj['rf_lod_level'] = lod_level
                obj['rf_lod_distance'] = lod_info['distance']
                obj.hide_viewport = False
                bpy.context.collection.objects.link(obj)

                # Vertex groups and weights
                if bone_names and all_bone_links:
                    for bname in bone_names:
                        obj.vertex_groups.new(name=bname)
                    for vi, (weights, bone_ids) in enumerate(all_bone_links):
                        w_sum = sum(weights)
                        for wi in range(4):
                            if weights[wi] > 0 and bone_ids[wi] < num_bones:
                                w_norm = weights[wi] / w_sum if w_sum > 0 else 0
                                if w_norm > 0:
                                    obj.vertex_groups[bone_names[bone_ids[wi]]].add([vi], w_norm, 'REPLACE')

                # Parent to armature
                obj.parent = arm_obj
                mod = obj.modifiers.new(name='Armature', type='ARMATURE')
                mod.object = arm_obj

                created += 1

            self.report({'INFO'}, f"Imported {created} LOD mesh{'es' if created > 1 else ''}")

        return {'FINISHED'}


class RFCHAR_OT_RestPose(bpy.types.Operator):
    bl_idname = "rfchar.rest_pose"
    bl_label = "Rest Pose"
    bl_description = "Clear animation and return to rest/bind pose"

    def execute(self, context):
        arm_obj = _get_armature(context)
        if not arm_obj or not arm_obj.animation_data:
            return {'CANCELLED'}
        arm_obj.animation_data.action = None
        for pb in arm_obj.pose.bones:
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)
            pb.scale = (1, 1, 1)
        context.scene.frame_set(context.scene.frame_current)
        return {'FINISHED'}


class RFCHAR_OT_DeleteAnimation(bpy.types.Operator):
    bl_idname = "rfchar.delete_animation"
    bl_label = "Delete Animation"
    bl_description = "Remove this animation from the armature"
    bl_options = {'UNDO'}

    action_name: StringProperty()

    def execute(self, context):
        arm_obj = _get_armature(context)
        if not arm_obj or not arm_obj.animation_data:
            return {'CANCELLED'}

        # Find and remove the NLA track containing this action
        action = bpy.data.actions.get(self.action_name)
        if not action:
            return {'CANCELLED'}

        # Clear active action if it matches
        if arm_obj.animation_data.action == action:
            arm_obj.animation_data.action = None
            for pb in arm_obj.pose.bones:
                pb.rotation_quaternion = (1, 0, 0, 0)
                pb.location = (0, 0, 0)

        # Remove NLA tracks that use this action
        tracks_to_remove = []
        for track in arm_obj.animation_data.nla_tracks:
            for strip in track.strips:
                if strip.action == action:
                    tracks_to_remove.append(track)
                    break
        for track in tracks_to_remove:
            arm_obj.animation_data.nla_tracks.remove(track)

        # Remove the action (mark fake user false so it gets cleaned up)
        action.use_fake_user = False
        if action.users == 0:
            bpy.data.actions.remove(action)

        self.report({'INFO'}, f"Deleted '{self.action_name}'")
        return {'FINISHED'}


class RFCHAR_OT_DeleteAllAnimations(bpy.types.Operator):
    bl_idname = "rfchar.delete_all_animations"
    bl_label = "Delete All Animations"
    bl_description = "Remove all imported animations from the armature"
    bl_options = {'UNDO'}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        arm_obj = _get_armature(context)
        if not arm_obj or not arm_obj.animation_data:
            return {'CANCELLED'}

        # Collect all RF actions
        rf_actions = _get_rf_actions(arm_obj)

        # Clear active action
        arm_obj.animation_data.action = None
        for pb in arm_obj.pose.bones:
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)

        # Remove all NLA tracks
        while arm_obj.animation_data.nla_tracks:
            arm_obj.animation_data.nla_tracks.remove(arm_obj.animation_data.nla_tracks[0])

        # Remove the actions
        count = 0
        for aname, action in rf_actions.items():
            action.use_fake_user = False
            if action.users == 0:
                bpy.data.actions.remove(action)
            count += 1

        context.scene.frame_set(context.scene.frame_current)
        self.report({'INFO'}, f"Deleted {count} animations")
        return {'FINISHED'}


class RFCHAR_OT_CheckWeights(bpy.types.Operator):
    bl_idname = "rfchar.check_weights"
    bl_label = "Check Weights"
    bl_description = "Find vertices with no bone weights assigned. Selects them so you can fix them"
    bl_options = {'UNDO'}

    def execute(self, context):
        arm_obj = _get_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature found")
            return {'CANCELLED'}

        # Find the mesh to check (active selection or first visible child)
        mesh_obj = None
        if context.active_object and context.active_object.type == 'MESH':
            mesh_obj = context.active_object
        else:
            for c in arm_obj.children:
                if c.type == 'MESH' and not c.hide_viewport and not c.get('rf_original_mesh'):
                    mesh_obj = c
                    break
            if not mesh_obj:
                for c in arm_obj.children:
                    if c.type == 'MESH' and not c.hide_viewport:
                        mesh_obj = c
                        break

        if not mesh_obj:
            self.report({'ERROR'}, "No visible mesh found")
            return {'CANCELLED'}

        mesh = mesh_obj.data
        bone_names = {b.name for b in arm_obj.data.bones}
        bone_groups = set()
        for vg in mesh_obj.vertex_groups:
            if vg.name in bone_names:
                bone_groups.add(vg.index)

        # Check each vertex
        unweighted = []
        zero_weight = []
        for vi, vert in enumerate(mesh.vertices):
            bone_weight_sum = 0.0
            for g in vert.groups:
                if g.group in bone_groups:
                    bone_weight_sum += g.weight
            if bone_weight_sum < 0.001:
                if any(g.group in bone_groups for g in vert.groups):
                    zero_weight.append(vi)
                else:
                    unweighted.append(vi)

        total_bad = len(unweighted) + len(zero_weight)

        if total_bad == 0:
            self.report({'INFO'}, f"All {len(mesh.vertices)} vertices have bone weights assigned")
            return {'FINISHED'}

        # Select the problem vertices
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)

        # Deselect all verts first
        for v in mesh.vertices:
            v.select = False
        for e in mesh.edges:
            e.select = False
        for f in mesh.polygons:
            f.select = False

        # Select problem verts
        for vi in unweighted:
            mesh.vertices[vi].select = True
        for vi in zero_weight:
            mesh.vertices[vi].select = True

        mesh.update()
        try:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.context.tool_settings.mesh_select_mode = (True, False, False)
        except Exception:
            pass

        parts = []
        if unweighted:
            parts.append(f"{len(unweighted)} unweighted")
        if zero_weight:
            parts.append(f"{len(zero_weight)} near-zero weight")
        self.report({'WARNING'},
            f"{total_bad} problem vertices selected on '{mesh_obj.name}': {', '.join(parts)}")
        return {'FINISHED'}


class RFCHAR_OT_FindTextures(bpy.types.Operator):
    bl_idname = "rfchar.find_textures"
    bl_label = "Find Missing Textures"
    bl_description = "Browse for a folder to locate missing texture files"

    directory: StringProperty(subtype='DIR_PATH')

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        # Use Blender's built-in missing files resolver
        bpy.ops.file.find_missing_files(directory=self.directory)
        self.report({'INFO'}, "Searched for missing textures")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════════
#  REQUIRED ANIMATIONS LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_load_anims_from_folder(arm_obj, required, folder):
    """Import required animations from a folder into the armature. Returns count imported."""
    rf_bones_json = arm_obj.get('rf_bones')
    if not rf_bones_json:
        return 0
    bones_data = json.loads(rf_bones_json)
    for b in bones_data:
        b['inv_bind_quat'] = tuple(b['inv_bind_quat'])
        b['inv_bind_pos'] = tuple(b['inv_bind_pos'])

    # Already-loaded action names
    rf_actions = _get_rf_actions(arm_obj)
    loaded_names = set()
    for aname in rf_actions.keys():
        loaded_names.add(aname.lower())
        loaded_names.add(aname.lower() + '.rfa')
        if aname.lower().endswith('.rfa'):
            loaded_names.add(aname.lower()[:-4])

    # Build lowercase filename map of the folder
    folder_files = {}
    try:
        for fn in os.listdir(folder):
            if fn.lower().endswith(('.rfa', '.mvf')):
                folder_files[fn.lower()] = fn
                folder_files[os.path.splitext(fn)[0].lower()] = fn
    except Exception:
        return 0

    # Find matching files and import
    imported = 0
    for rfa in required:
        name_lc = rfa.lower()
        name_no_ext = os.path.splitext(rfa)[0].lower()
        if name_lc in loaded_names or name_no_ext in loaded_names:
            continue
        actual = folder_files.get(name_lc) or folder_files.get(name_no_ext)
        if not actual:
            actual = folder_files.get(name_no_ext + '.mvf')
        if not actual:
            continue
        fp = os.path.join(folder, actual)
        anim_name = os.path.splitext(os.path.basename(fp))[0]
        try:
            rfa_data = parse_rfa(fp)
            _import_rfa(rfa_data, arm_obj, bones_data, anim_name)
            action = arm_obj.animation_data.action
            if action:
                track = arm_obj.animation_data.nla_tracks.new()
                track.name = action.name
                start = int(action.frame_range[0])
                track.strips.new(action.name, start, action)
                track.mute = True
                arm_obj.animation_data.action = None
            imported += 1
        except Exception as e:
            print(f"[RF Character] Failed to auto-load '{anim_name}': {e}")

    # Force rest pose after all imports: clear action, reset pose bones, mute all tracks
    if imported > 0 and arm_obj.animation_data:
        try:
            arm_obj.animation_data.action = None
            for track in arm_obj.animation_data.nla_tracks:
                track.mute = True
            for pb in arm_obj.pose.bones:
                pb.rotation_quaternion = (1, 0, 0, 0)
                pb.location = (0, 0, 0)
                pb.scale = (1, 1, 1)
            try:
                bpy.context.scene.frame_set(bpy.context.scene.frame_current)
            except Exception:
                pass
        except Exception as e:
            print(f"[RF Character] Could not set rest pose: {e}")

    return imported


def _get_master_config_path():
    """Path to the master folder config file (persists across sessions)."""
    try:
        config_dir = bpy.utils.user_resource('CONFIG')
        return os.path.join(config_dir, 'rf_character_master.txt')
    except Exception:
        return None


def _save_master_folder(path):
    """Save master folder path to config file."""
    cfg = _get_master_config_path()
    if not cfg:
        return False
    try:
        os.makedirs(os.path.dirname(cfg), exist_ok=True)
        with open(cfg, 'w', encoding='utf-8') as f:
            f.write(path or '')
        return True
    except Exception as e:
        print(f"[RF Character] Could not save master folder: {e}")
        return False


def _load_master_folder():
    """Load master folder path from config file."""
    cfg = _get_master_config_path()
    if not cfg or not os.path.isfile(cfg):
        return ''
    try:
        with open(cfg, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ''


def _get_master_folder(context=None):
    """Get master folder path from saved config."""
    saved = _load_master_folder()
    if saved:
        try:
            return bpy.path.abspath(saved)
        except Exception:
            return saved
    return ''


def _get_auto_load_enabled(context=None):
    """Get auto-load setting - always on unless explicitly disabled."""
    if context is None:
        context = bpy.context
    try:
        return getattr(context.scene, 'rfchar_auto_load_anims', True)
    except Exception:
        return True


class RFCHAR_OT_SetMasterFolder(bpy.types.Operator):
    bl_idname = "rfchar.set_master_folder"
    bl_label = "Set Master Animation Folder"
    bl_description = "Choose the folder containing all your RF .rfa animation files. Required animations will be auto-loaded from here whenever a V3C is imported"
    bl_options = {'REGISTER'}

    directory: StringProperty(subtype='DIR_PATH')

    def invoke(self, context, event):
        # Pre-fill with current saved folder
        current = _get_master_folder(context)
        if current and os.path.isdir(current):
            self.directory = current
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        folder = bpy.path.abspath(self.directory) if self.directory else ''
        if not folder or not os.path.isdir(folder):
            self.report({'ERROR'}, "Not a valid folder")
            return {'CANCELLED'}
        if _save_master_folder(folder):
            try:
                count = sum(1 for f in os.listdir(folder)
                           if f.lower().endswith(('.rfa', '.mvf')))
                self.report({'INFO'}, f"Master folder set: {count} animation files found")
            except Exception:
                self.report({'INFO'}, f"Master folder set: {folder}")
        else:
            self.report({'WARNING'}, "Folder set but could not save to config")
        return {'FINISHED'}


class RFCHAR_OT_ClearMasterFolder(bpy.types.Operator):
    bl_idname = "rfchar.clear_master_folder"
    bl_label = "Clear Master Folder"
    bl_description = "Remove the saved master animation folder"
    bl_options = {'REGISTER'}

    def execute(self, context):
        _save_master_folder('')
        self.report({'INFO'}, "Master folder cleared")
        return {'FINISHED'}


class RFCHAR_OT_AutoLoadFromMaster(bpy.types.Operator):
    bl_idname = "rfchar.autoload_from_master"
    bl_label = "Auto-Load from Master Folder"
    bl_description = "Load required animations from the master folder set in Preferences"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature found")
            return {'CANCELLED'}

        master_abs = _get_master_folder(context)
        if not master_abs:
            self.report({'ERROR'}, "No master folder set. Open Edit > Preferences > Add-ons > RF Character Import")
            return {'CANCELLED'}
        if not os.path.isdir(master_abs):
            self.report({'ERROR'}, f"Master folder doesn't exist: {master_abs}")
            return {'CANCELLED'}

        req_json = arm_obj.get('rf_required_anims', '[]')
        required = json.loads(req_json) if isinstance(req_json, str) else []
        if not required:
            self.report({'INFO'}, "No required animations for this character")
            return {'CANCELLED'}

        loaded = _auto_load_anims_from_folder(arm_obj, required, master_abs)
        if loaded > 0:
            arm_obj['rf_anim_folder'] = master_abs
            context.scene['rf_last_anim_folder'] = master_abs
            self.report({'INFO'}, f"Loaded {loaded} animation{'s' if loaded != 1 else ''} from master folder")
        else:
            self.report({'INFO'}, "All required animations already loaded (or none found in folder)")
        return {'FINISHED'}


class RFCHAR_OT_LoadRequiredAnims(bpy.types.Operator):
    bl_idname = "rfchar.load_required_anims"
    bl_label = "Load Required Animations"
    bl_description = "Import all missing required animations from your RF data folder"
    bl_options = {'REGISTER', 'UNDO'}

    directory: StringProperty(subtype='DIR_PATH')
    load_all: BoolProperty(name="Load All", default=True,
        description="Load all missing animations. If False, only loads checked ones")
    force_browse: BoolProperty(default=False, options={'HIDDEN'})

    def invoke(self, context, event):
        # Try to auto-use the remembered folder if one exists
        if not self.force_browse:
            arm_obj = _find_rf_armature(context)
            remembered = None
            if arm_obj:
                remembered = arm_obj.get('rf_anim_folder')
            if not remembered:
                remembered = context.scene.get('rf_last_anim_folder')
            if not remembered:
                # Fall back to master folder
                master = _get_master_folder(context)
                if master:
                    remembered = master
            if remembered and os.path.isdir(remembered):
                self.directory = remembered
                return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature found")
            return {'CANCELLED'}

        rf_bones_json = arm_obj.get('rf_bones')
        if not rf_bones_json:
            self.report({'ERROR'}, "No RF bone data on armature")
            return {'CANCELLED'}
        bones_data = json.loads(rf_bones_json)
        for b in bones_data:
            b['inv_bind_quat'] = tuple(b['inv_bind_quat'])
            b['inv_bind_pos'] = tuple(b['inv_bind_pos'])

        req_json = arm_obj.get('rf_required_anims', '[]')
        required = json.loads(req_json) if isinstance(req_json, str) else []
        if not required:
            self.report({'ERROR'}, "No required animations list")
            return {'CANCELLED'}

        # Get already-loaded action names
        rf_actions = _get_rf_actions(arm_obj)
        loaded_names = set()
        for aname in rf_actions.keys():
            loaded_names.add(aname.lower())
            loaded_names.add(aname.lower() + '.rfa')
            if aname.lower().endswith('.rfa'):
                loaded_names.add(aname.lower()[:-4])

        # Scan folder for matching files (case-insensitive)
        folder = bpy.path.abspath(self.directory)
        if not os.path.isdir(folder):
            self.report({'ERROR'}, f"Folder not found: {folder}")
            return {'CANCELLED'}

        # Build lowercase→actual filename map for the folder
        folder_files = {}
        for fn in os.listdir(folder):
            if fn.lower().endswith(('.rfa', '.mvf')):
                folder_files[fn.lower()] = fn
                # Also map without extension
                folder_files[os.path.splitext(fn)[0].lower()] = fn

        # Find which required anims are missing and available in folder
        to_import = []
        for rfa in required:
            name_lower = rfa.lower()
            name_no_ext = os.path.splitext(rfa)[0].lower()

            # Skip if already loaded
            if name_lower in loaded_names or name_no_ext in loaded_names:
                continue

            # Check folder (try .rfa then .mvf)
            actual = folder_files.get(name_lower)
            if not actual:
                actual = folder_files.get(name_no_ext)
            if not actual:
                mvf_name = name_no_ext + '.mvf'
                actual = folder_files.get(mvf_name)
            if actual:
                to_import.append(os.path.join(folder, actual))

        if not to_import:
            missing_count = sum(1 for r in required
                              if r.lower() not in loaded_names
                              and os.path.splitext(r)[0].lower() not in loaded_names)
            if missing_count == 0:
                self.report({'INFO'}, "All required animations already loaded")
            else:
                self.report({'WARNING'}, f"{missing_count} animations still missing (not found in folder)")
            return {'FINISHED'}

        # Import them
        imported = 0
        failed = 0
        for fp in to_import:
            anim_name = os.path.splitext(os.path.basename(fp))[0]
            try:
                rfa = parse_rfa(fp)
                _import_rfa(rfa, arm_obj, bones_data, anim_name)
                action = arm_obj.animation_data.action
                if action:
                    track = arm_obj.animation_data.nla_tracks.new()
                    track.name = action.name
                    start = int(action.frame_range[0])
                    strip = track.strips.new(action.name, start, action)
                    track.mute = True
                    arm_obj.animation_data.action = None
                imported += 1
            except Exception as e:
                self.report({'WARNING'}, f"Failed '{anim_name}': {e}")
                failed += 1

        # Recount missing
        rf_actions = _get_rf_actions(arm_obj)
        loaded_names = set()
        for aname in rf_actions.keys():
            loaded_names.add(aname.lower())
            loaded_names.add(aname.lower() + '.rfa')
            if aname.lower().endswith('.rfa'):
                loaded_names.add(aname.lower()[:-4])
        still_missing = sum(1 for r in required
                           if r.lower() not in loaded_names
                           and os.path.splitext(r)[0].lower() not in loaded_names)

        # Force rest pose after loading
        if imported > 0 and arm_obj.animation_data:
            try:
                arm_obj.animation_data.action = None
                for track in arm_obj.animation_data.nla_tracks:
                    track.mute = True
                for pb in arm_obj.pose.bones:
                    pb.rotation_quaternion = (1, 0, 0, 0)
                    pb.location = (0, 0, 0)
                    pb.scale = (1, 1, 1)
                context.scene.frame_set(context.scene.frame_current)
            except Exception:
                pass

        msg = f"Imported {imported} animation{'s' if imported != 1 else ''}"
        if failed:
            msg += f", {failed} failed"
        if still_missing:
            msg += f", {still_missing} still missing"
        else:
            msg += " — all required animations loaded"
        self.report({'INFO'}, msg)

        # Remember this folder for future loads
        if imported > 0:
            arm_obj['rf_anim_folder'] = folder
            context.scene['rf_last_anim_folder'] = folder

        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORT OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class RFCHAR_OT_ExportRFA(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.rf_rfa"
    bl_label = "Export RF Animation (.rfa)"
    bl_description = "Export the active animation as an RFA file"
    bl_options = {'REGISTER'}

    filename_ext = ".rfa"
    filter_glob: StringProperty(default="*.rfa", options={'HIDDEN'})

    def invoke(self, context, event):
        arm_obj = _find_rf_armature(context)
        if arm_obj and arm_obj.animation_data and arm_obj.animation_data.action:
            self.filepath = arm_obj.animation_data.action.name + ".rfa"
        return super().invoke(context, event)

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature found")
            return {'CANCELLED'}

        rf_bones_json = arm_obj.get('rf_bones')
        if not rf_bones_json:
            self.report({'ERROR'}, "No RF bone data on armature. Import a V3C first.")
            return {'CANCELLED'}
        bones_data = json.loads(rf_bones_json)
        for b in bones_data:
            b['inv_bind_quat'] = tuple(b['inv_bind_quat'])
            b['inv_bind_pos'] = tuple(b['inv_bind_pos'])

        if not arm_obj.animation_data or not arm_obj.animation_data.action:
            self.report({'ERROR'}, "No active animation on armature")
            return {'CANCELLED'}

        action = arm_obj.animation_data.action
        try:
            st, et = _export_rfa(arm_obj, action, self.filepath, bones_data)
            dur = (et - st) / RFA_TPS
            self.report({'INFO'}, f"Exported '{action.name}' ({dur:.2f}s, {len(bones_data)} bones)")
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            traceback.print_exc()
            return {'CANCELLED'}

        return {'FINISHED'}


class RFCHAR_OT_ExportAllRFA(bpy.types.Operator):
    bl_idname = "rfchar.export_all_rfa"
    bl_label = "Export All Animations"
    bl_description = "Export all loaded animations as .rfa files to a chosen folder"
    bl_options = {'REGISTER'}

    directory: StringProperty(subtype='DIR_PATH')

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature found")
            return {'CANCELLED'}

        rf_bones_json = arm_obj.get('rf_bones')
        if not rf_bones_json:
            self.report({'ERROR'}, "No RF bone data on armature")
            return {'CANCELLED'}
        bones_data = json.loads(rf_bones_json)
        for b in bones_data:
            b['inv_bind_quat'] = tuple(b['inv_bind_quat'])
            b['inv_bind_pos'] = tuple(b['inv_bind_pos'])

        rf_actions = _get_rf_actions(arm_obj)
        if not rf_actions:
            self.report({'ERROR'}, "No animations loaded")
            return {'CANCELLED'}

        exported = 0
        failed = 0
        for aname, action in sorted(rf_actions.items()):
            filepath = os.path.join(self.directory, aname + ".rfa")
            try:
                _export_rfa(arm_obj, action, filepath, bones_data)
                exported += 1
            except Exception as e:
                self.report({'WARNING'}, f"Failed '{aname}': {e}")
                failed += 1

        msg = f"Exported {exported} animation{'s' if exported != 1 else ''}"
        if failed:
            msg += f", {failed} failed"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class RFCHAR_OT_ExportV3C(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.rf_v3c"
    bl_label = "Export RF Character Mesh (.v3c)"
    bl_description = "Export the character mesh, skeleton, and collision spheres as a V3C file"
    bl_options = {'REGISTER'}

    filename_ext = ".v3c"
    filter_glob: StringProperty(default="*.v3c;*.v3m", options={'HIDDEN'})

    def invoke(self, context, event):
        arm_obj = _find_rf_armature(context)
        if arm_obj:
            source = arm_obj.get('rf_source_file', '')
            if source:
                self.filepath = source
        return super().invoke(context, event)

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature found")
            return {'CANCELLED'}

        rf_bones_json = arm_obj.get('rf_bones')
        if not rf_bones_json:
            self.report({'ERROR'}, "No RF bone data on armature. Import a V3C first.")
            return {'CANCELLED'}
        bones_data = json.loads(rf_bones_json)
        for b in bones_data:
            b['inv_bind_quat'] = tuple(b['inv_bind_quat'])
            b['inv_bind_pos'] = tuple(b['inv_bind_pos'])

        # Find the mesh to export (prefer active selection, then first child mesh)
        mesh_obj = None
        if context.active_object and context.active_object.type == 'MESH':
            mesh_obj = context.active_object
        else:
            for c in arm_obj.children:
                if c.type == 'MESH' and not c.hide_viewport:
                    mesh_obj = c
                    break

        if not mesh_obj:
            self.report({'ERROR'}, "No mesh found to export")
            return {'CANCELLED'}

        # Check for unapplied transforms (common with FBX imports)
        # Auto-apply rotation and scale so vertex data is in correct space
        scl = mesh_obj.scale
        has_scale = abs(scl.x - 1) > 0.001 or abs(scl.y - 1) > 0.001 or abs(scl.z - 1) > 0.001
        has_rotation = False
        if mesh_obj.rotation_mode == 'QUATERNION':
            q = mesh_obj.rotation_quaternion
            has_rotation = abs(q.w - 1) > 0.001 or abs(q.x) > 0.001 or abs(q.y) > 0.001 or abs(q.z) > 0.001
        else:
            rot = mesh_obj.rotation_euler
            has_rotation = abs(rot.x) > 0.001 or abs(rot.y) > 0.001 or abs(rot.z) > 0.001
        if has_rotation or has_scale:
            # Ensure object mode for transform apply
            prev_mode = context.mode
            if prev_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            prev_active = context.view_layer.objects.active
            prev_selected = [o for o in context.selected_objects]
            bpy.ops.object.select_all(action='DESELECT')
            mesh_obj.select_set(True)
            context.view_layer.objects.active = mesh_obj
            try:
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
                self.report({'WARNING'}, "Applied unapplied rotation/scale on mesh before export")
            except Exception:
                self.report({'WARNING'}, "Could not auto-apply transforms. Try Ctrl+A → Rotation & Scale manually.")
            bpy.ops.object.select_all(action='DESELECT')
            for o in prev_selected:
                if o.name in context.view_layer.objects:
                    o.select_set(True)
            context.view_layer.objects.active = prev_active

        # Gather collision spheres and prop points from live scene empties
        bone_names = [b['name'] for b in bones_data]
        cspheres = _gather_scene_cspheres(arm_obj, bone_names) if arm_obj else []
        prop_points = _gather_scene_prop_points(arm_obj, bone_names) if arm_obj else []

        try:
            nv, nt = _export_v3c(mesh_obj, arm_obj, self.filepath, bones_data, cspheres, prop_points)
            self.report({'INFO'},
                f"Exported '{mesh_obj.name}': {nv} verts, {nt} tris, "
                f"{len(bones_data)} bones, {len(cspheres)} hitboxes, {len(prop_points)} props")
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            traceback.print_exc()
            return {'CANCELLED'}

        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════════
#  PANEL: HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _get_armature(context):
    """Get RF armature from selection."""
    obj = context.active_object
    if obj and obj.type == 'ARMATURE' and obj.get('rf_bones'):
        return obj
    if obj and obj.type == 'MESH' and obj.parent and obj.parent.type == 'ARMATURE':
        return obj.parent
    return None


def _find_rf_armature(context):
    """Find any RF armature in the scene (for when custom mesh is selected)."""
    arm = _get_armature(context)
    if arm:
        return arm
    for o in context.scene.objects:
        if o.type == 'ARMATURE' and o.get('rf_bones'):
            return o
    return None


def _get_rf_actions(arm_obj):
    """Collect all actions associated with this armature (from NLA + active)."""
    actions = {}
    if not arm_obj or not arm_obj.animation_data:
        return actions
    for track in (arm_obj.animation_data.nla_tracks or []):
        for strip in track.strips:
            if strip.action:
                actions[strip.action.name] = strip.action
    active = arm_obj.animation_data.action
    if active:
        actions[active.name] = active
    return actions


# ═══════════════════════════════════════════════════════════════════════════════
#  PANEL: UI
# ═══════════════════════════════════════════════════════════════════════════════

class RFCHAR_PT_WorkflowPanel(bpy.types.Panel):
    bl_label = "Workflow"
    bl_idname = "RFCHAR_PT_workflow"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "RF Character"
    bl_order = 0
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="1. Import a V3C character mesh")
        col.label(text="2. Find Textures to link materials")
        col.label(text="3. Batch import or add animations")
        col.label(text="4. Edit animations in Blender")
        col.label(text="5. Export RFA to use in-game")


class RFCHAR_OT_ImportRecent(bpy.types.Operator):
    bl_idname = "rfchar.import_recent"
    bl_label = "Import Recent V3C"
    bl_description = "Re-import a recently used V3C file"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty()

    def execute(self, context):
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "File no longer exists")
            # Remove from recent list
            recent = list(context.scene.get('rf_recent_v3c', []))
            if self.filepath in recent:
                recent.remove(self.filepath)
                context.scene['rf_recent_v3c'] = recent
            return {'CANCELLED'}
        bpy.ops.import_scene.rf_v3c(filepath=self.filepath)
        return {'FINISHED'}


class RFCHAR_OT_ClearRecent(bpy.types.Operator):
    bl_idname = "rfchar.clear_recent"
    bl_label = "Clear Recent Files"
    bl_options = {'REGISTER'}

    def execute(self, context):
        context.scene['rf_recent_v3c'] = []
        return {'FINISHED'}


class RFCHAR_OT_ValidateExport(bpy.types.Operator):
    bl_idname = "rfchar.validate_export"
    bl_label = "Validate for Export"
    bl_description = "Check the current character for export problems before you export"
    bl_options = {'REGISTER'}

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature")
            return {'CANCELLED'}

        issues = []
        warnings = []
        info = []

        # Bone data
        rf_bones_json = arm_obj.get('rf_bones')
        if not rf_bones_json:
            issues.append("No RF bone data on armature (import a V3C first)")
        else:
            bones_data = json.loads(rf_bones_json)
            info.append(f"{len(bones_data)} RF bones available")
            # Check armature matches
            arm_bone_names = {b.name for b in arm_obj.data.bones}
            rf_bone_names = {b['name'] for b in bones_data}
            missing_in_arm = rf_bone_names - arm_bone_names
            extra_in_arm = arm_bone_names - rf_bone_names
            if missing_in_arm:
                warnings.append(f"{len(missing_in_arm)} RF bones missing from armature: {', '.join(list(missing_in_arm)[:3])}{'...' if len(missing_in_arm) > 3 else ''}")
            if extra_in_arm:
                warnings.append(f"{len(extra_in_arm)} extra bones on armature (will be ignored): {', '.join(list(extra_in_arm)[:3])}{'...' if len(extra_in_arm) > 3 else ''}")

        # Find mesh
        mesh_obj = None
        if context.active_object and context.active_object.type == 'MESH':
            mesh_obj = context.active_object
        else:
            for c in arm_obj.children:
                if c.type == 'MESH' and not c.hide_viewport:
                    mesh_obj = c; break

        if not mesh_obj:
            issues.append("No mesh to export")
        else:
            mesh = mesh_obj.data
            vcount = len(mesh.vertices)
            fcount = len(mesh.polygons)
            info.append(f"Mesh '{mesh_obj.name}': {vcount} verts, {fcount} faces")

            # Check triangulation
            non_tri = sum(1 for p in mesh.polygons if len(p.vertices) != 3)
            if non_tri:
                warnings.append(f"{non_tri} non-triangle faces (will be triangulated on export)")

            # Check weights
            if rf_bones_json:
                rf_bone_names = {b['name'] for b in bones_data}
                over_weighted = 0
                unweighted = 0
                wrong_bone = 0
                for v in mesh.vertices:
                    valid = [g for g in v.groups if mesh_obj.vertex_groups[g.group].name in rf_bone_names and g.weight > 0.001]
                    if len(valid) > 4:
                        over_weighted += 1
                    if not valid:
                        unweighted += 1
                    invalid = [g for g in v.groups if mesh_obj.vertex_groups[g.group].name not in rf_bone_names and g.weight > 0.001]
                    if invalid:
                        wrong_bone += 1
                if unweighted:
                    issues.append(f"{unweighted} vertices with no weights (will fail to bind)")
                if over_weighted:
                    warnings.append(f"{over_weighted} vertices with >4 bone influences (excess will be dropped)")
                if wrong_bone:
                    warnings.append(f"{wrong_bone} vertices weighted to non-RF bones (will be dropped)")

            # Check materials
            if not mesh.materials:
                warnings.append("Mesh has no materials")
            else:
                no_tex = 0
                wrong_ext = []
                name_mismatch = []
                for m in mesh.materials:
                    if not m: no_tex += 1; continue
                    img = None
                    if m.use_nodes and m.node_tree:
                        bsdf = m.node_tree.nodes.get('Principled BSDF')
                        if bsdf and 'Base Color' in bsdf.inputs:
                            for link in bsdf.inputs['Base Color'].links:
                                if link.from_node.type == 'TEX_IMAGE' and link.from_node.image:
                                    img = link.from_node.image
                                    break
                        if not img:
                            for n in m.node_tree.nodes:
                                if n.type == 'TEX_IMAGE' and n.image:
                                    img = n.image; break
                    if not img:
                        no_tex += 1
                        continue
                    # Check extension
                    if not img.name.lower().endswith('.tga'):
                        wrong_ext.append(img.name)
                    # Check material name matches image base
                    img_base = os.path.splitext(img.name)[0]
                    if m.name != img_base:
                        name_mismatch.append(f"{m.name}→{img_base}")
                if no_tex:
                    warnings.append(f"{no_tex}/{len(mesh.materials)} materials have no texture")
                if wrong_ext:
                    preview = ', '.join(wrong_ext[:3]) + ('...' if len(wrong_ext) > 3 else '')
                    warnings.append(f"{len(wrong_ext)} image(s) not .tga: {preview} (use Sync Texture Names)")
                if name_mismatch:
                    preview = ', '.join(name_mismatch[:3]) + ('...' if len(name_mismatch) > 3 else '')
                    info.append(f"{len(name_mismatch)} material/image name mismatch: {preview}")

            # Chunk size warning
            if vcount > 5400 and vcount <= 5461:
                warnings.append(f"Vertex count {vcount} is near the 5,461 per-chunk limit")
            if vcount > 5461:
                info.append(f"Will be split into {(vcount // 5400) + 1} chunks on export")

        # Report
        lines = []
        if issues:
            lines.append(f"{len(issues)} ERROR(S):")
            lines.extend([f"  ✗ {i}" for i in issues])
        if warnings:
            lines.append(f"{len(warnings)} WARNING(S):")
            lines.extend([f"  ⚠ {w}" for w in warnings])
        if info:
            lines.extend([f"  · {i}" for i in info])

        if not issues and not warnings:
            self.report({'INFO'}, "Export validation passed — ready to export")
        else:
            for line in lines:
                self.report({'WARNING'} if warnings and not issues else ({'ERROR'} if issues else {'INFO'}), line)
            self.report({'INFO'}, f"Validation: {len(issues)} errors, {len(warnings)} warnings. See System Console for details.")
            # Also print to console
            print("\n=== RF Character Export Validation ===")
            for line in lines:
                print(line)
            print("======================================")

        return {'FINISHED'} if not issues else {'CANCELLED'}


class RFCHAR_OT_RenameActionsToDB(bpy.types.Operator):
    bl_idname = "rfchar.rename_actions_to_db"
    bl_label = "Rename Actions to DB"
    bl_description = "Match loaded animation names (case-insensitive) to the required animations database and rename them to canonical form"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm_obj = _find_rf_armature(context)
        if not arm_obj:
            self.report({'ERROR'}, "No RF armature")
            return {'CANCELLED'}

        req_json = arm_obj.get('rf_required_anims', '[]')
        required = json.loads(req_json) if isinstance(req_json, str) else []
        if not required:
            self.report({'INFO'}, "No required animations list to match against")
            return {'CANCELLED'}

        # Build canonical name map (lowercase no-ext → canonical name w/o ext)
        canonical = {}
        for rfa in required:
            no_ext = os.path.splitext(rfa)[0]
            canonical[no_ext.lower()] = no_ext

        rf_actions = _get_rf_actions(arm_obj)
        renamed = 0
        for aname, action in list(rf_actions.items()):
            lc = aname.lower()
            # Strip .rfa extension if present
            if lc.endswith('.rfa'):
                lc = lc[:-4]
            if lc in canonical and action.name != canonical[lc]:
                action.name = canonical[lc]
                renamed += 1

        self.report({'INFO'}, f"Renamed {renamed} action{'s' if renamed != 1 else ''}")
        return {'FINISHED'}


class RFCHAR_OT_BatchExportV3C(bpy.types.Operator):
    bl_idname = "rfchar.batch_export_v3c"
    bl_label = "Batch Export V3Cs"
    bl_description = "Export each selected armature to its own V3C file in a chosen folder"
    bl_options = {'REGISTER'}

    directory: StringProperty(subtype='DIR_PATH')

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        # Collect armatures to export (selected + active)
        armatures = [o for o in context.selected_objects if o.type == 'ARMATURE' and o.get('rf_bones')]
        if context.active_object and context.active_object.type == 'ARMATURE' and context.active_object not in armatures:
            if context.active_object.get('rf_bones'):
                armatures.append(context.active_object)

        if not armatures:
            self.report({'ERROR'}, "Select one or more RF armatures to export")
            return {'CANCELLED'}

        folder = bpy.path.abspath(self.directory)
        if not os.path.isdir(folder):
            self.report({'ERROR'}, f"Folder not found: {folder}")
            return {'CANCELLED'}

        original_active = context.view_layer.objects.active
        exported = 0
        failed = 0
        for arm in armatures:
            source = arm.get('rf_source_file', arm.name + '.v3c')
            if not source.lower().endswith('.v3c'):
                source = os.path.splitext(source)[0] + '.v3c'
            out_path = os.path.join(folder, source)
            try:
                context.view_layer.objects.active = arm
                bpy.ops.export_scene.rf_v3c(filepath=out_path)
                exported += 1
            except Exception as e:
                self.report({'WARNING'}, f"Failed {arm.name}: {e}")
                failed += 1

        context.view_layer.objects.active = original_active
        self.report({'INFO'}, f"Exported {exported}/{len(armatures)} V3Cs to {folder}")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXTURE ATLAS — combine multiple material textures into one
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_mesh_images(meshes):
    """Returns OrderedDict: image → list of (mesh_obj, material_index)"""
    from collections import OrderedDict
    result = OrderedDict()
    for obj in meshes:
        for slot_idx, slot in enumerate(obj.material_slots):
            mat = slot.material
            if not mat or not mat.use_nodes or not mat.node_tree:
                continue
            img = None
            # Prefer the image connected to Base Color, else first image node
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if bsdf and 'Base Color' in bsdf.inputs:
                for link in bsdf.inputs['Base Color'].links:
                    if link.from_node.type == 'TEX_IMAGE' and link.from_node.image:
                        img = link.from_node.image
                        break
            if not img:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        img = node.image
                        break
            if img:
                result.setdefault(img, []).append((obj, slot_idx))
    return result


class RFCHAR_OT_GenerateAtlas(bpy.types.Operator):
    bl_idname = "rfchar.generate_atlas"
    bl_label = "Generate Texture Atlas"
    bl_description = "Combine the selected mesh's material textures into a single atlas TGA, remap UVs, and replace materials"
    bl_options = {'REGISTER', 'UNDO'}

    atlas_size: EnumProperty(
        name="Size",
        items=[
            ('128', "128x128", "128x128 atlas"),
            ('256', "256x256", "256x256 atlas"),
            ('512', "512x512", "512x512 atlas (recommended)"),
            ('1024', "1024x1024", "1024x1024 atlas (high quality)"),
            ('2048', "2048x2048", "2048x2048 atlas (very high quality)"),
        ],
        default='512',
    )

    atlas_name: StringProperty(
        name="Name",
        description="Atlas filename without extension",
        default="atlas",
    )

    output_dir: StringProperty(
        name="Output Folder",
        subtype='DIR_PATH',
        default="",
    )

    file_format: EnumProperty(
        name="Format",
        items=[
            ('TGA', "TGA", "Targa (RF standard)"),
            ('PNG', "PNG", "PNG (lossless)"),
        ],
        default='TGA',
    )

    replace_materials: BoolProperty(
        name="Replace materials with atlas material",
        description="Remove individual material slots and assign single atlas material",
        default=True,
    )

    remap_uvs: BoolProperty(
        name="Remap UVs",
        description="Transform UVs to reference the correct atlas region",
        default=True,
    )

    def invoke(self, context, event):
        # Default name from active object or armature
        arm_obj = _find_rf_armature(context)
        if arm_obj and arm_obj.get('rf_source_file'):
            self.atlas_name = os.path.splitext(arm_obj['rf_source_file'])[0]
        elif context.active_object:
            self.atlas_name = context.active_object.name

        # Default folder from scene memory
        remembered = context.scene.get('rf_texture_folder') or context.scene.get('rf_last_anim_folder')
        if remembered:
            self.output_dir = remembered

        return context.window_manager.invoke_props_dialog(self, width=400)

    def draw(self, context):
        layout = self.layout
        meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if not meshes and context.active_object and context.active_object.type == 'MESH':
            meshes = [context.active_object]
        images = _collect_mesh_images(meshes)

        if not images:
            layout.label(text="No textured materials found on selected mesh(es)", icon='ERROR')
            return

        box = layout.box()
        box.label(text=f"Found {len(images)} texture(s) on {len(meshes)} mesh(es):", icon='IMAGE_DATA')
        col = box.column(align=True)
        for img, uses in list(images.items())[:8]:
            size_txt = f"{img.size[0]}x{img.size[1]}"
            col.label(text=f"  · {img.name}  ({size_txt})", icon='IMAGE')
        if len(images) > 8:
            col.label(text=f"  ...and {len(images) - 8} more")

        layout.separator()
        layout.prop(self, "atlas_name")
        layout.prop(self, "atlas_size")
        layout.prop(self, "file_format")
        layout.prop(self, "output_dir")
        layout.separator()
        layout.prop(self, "remap_uvs")
        layout.prop(self, "replace_materials")

    def execute(self, context):
        # Collect meshes
        meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if not meshes and context.active_object and context.active_object.type == 'MESH':
            meshes = [context.active_object]
        if not meshes:
            self.report({'ERROR'}, "Select a mesh (or meshes) to atlas")
            return {'CANCELLED'}

        images_map = _collect_mesh_images(meshes)
        if not images_map:
            self.report({'ERROR'}, "No image textures found on any material")
            return {'CANCELLED'}

        img_list = list(images_map.keys())
        n = len(img_list)

        # Grid layout
        grid_w = math.ceil(math.sqrt(n))
        grid_h = math.ceil(n / grid_w)
        atlas_size = int(self.atlas_size)
        tile_size = atlas_size // max(grid_w, grid_h)
        if tile_size < 8:
            self.report({'ERROR'}, f"Atlas size too small for {n} textures. Increase atlas size.")
            return {'CANCELLED'}

        # Assign each image a tile position
        # Layout: image i goes at col=i%grid_w, row=i//grid_w, row 0 at TOP
        tile_positions = {}  # image → (u_offset, v_offset, u_scale, v_scale, px, py_from_bottom)
        for i, img in enumerate(img_list):
            col = i % grid_w
            row = i // grid_w
            px = col * tile_size
            py_from_bottom = (grid_h - 1 - row) * tile_size
            u_off = px / atlas_size
            v_off = py_from_bottom / atlas_size
            tile_scale = tile_size / atlas_size
            tile_positions[img] = (u_off, v_off, tile_scale, tile_scale, px, py_from_bottom)

        # Build atlas using numpy
        try:
            import numpy as np
        except ImportError:
            self.report({'ERROR'}, "numpy required for atlas generation (should ship with Blender)")
            return {'CANCELLED'}

        atlas_arr = np.zeros((atlas_size, atlas_size, 4), dtype=np.float32)
        atlas_arr[:, :, 3] = 1.0  # opaque default

        skipped = []
        success = 0
        temp_copies = []
        try:
            for img in img_list:
                u_off, v_off, u_sc, v_sc, px, py = tile_positions[img]

                # Ensure source image is loaded
                try:
                    if img.source == 'FILE' and not img.has_data:
                        img.reload()
                except Exception:
                    pass

                # Get source size and validate
                src_w, src_h = img.size[0], img.size[1]
                if src_w == 0 or src_h == 0:
                    skipped.append(f"{img.name} (size 0x0 - not loaded)")
                    continue

                channels = img.channels if hasattr(img, 'channels') else 4

                # Read pixels from original (don't trust copy.pixels)
                try:
                    src_pixels = np.empty(src_w * src_h * channels, dtype=np.float32)
                    img.pixels.foreach_get(src_pixels)
                except Exception as e:
                    skipped.append(f"{img.name} (pixel read failed: {e})")
                    continue

                # Validate we got actual data
                if src_pixels.size == 0:
                    skipped.append(f"{img.name} (empty pixel data)")
                    continue

                # Reshape to (h, w, channels)
                try:
                    src_arr = src_pixels.reshape(src_h, src_w, channels)
                except Exception as e:
                    skipped.append(f"{img.name} (reshape failed: {e})")
                    continue

                # Convert to RGBA
                if channels == 3:
                    rgba = np.ones((src_h, src_w, 4), dtype=np.float32)
                    rgba[:, :, :3] = src_arr
                    src_arr = rgba
                elif channels == 1:
                    rgba = np.ones((src_h, src_w, 4), dtype=np.float32)
                    rgba[:, :, 0] = src_arr[:, :, 0]
                    rgba[:, :, 1] = src_arr[:, :, 0]
                    rgba[:, :, 2] = src_arr[:, :, 0]
                    src_arr = rgba
                elif channels != 4:
                    skipped.append(f"{img.name} (unsupported channels: {channels})")
                    continue

                # Fix broken alpha (all zeros means opaque)
                if np.all(src_arr[:, :, 3] == 0):
                    src_arr = src_arr.copy()
                    src_arr[:, :, 3] = 1.0

                # Resize to tile_size × tile_size using nearest-neighbor via numpy
                # (Pillow would be better but may not be available)
                if src_w != tile_size or src_h != tile_size:
                    # Use linear interpolation via Blender's scale() on a copy
                    try:
                        img_copy = img.copy()
                        temp_copies.append(img_copy)
                        img_copy.scale(tile_size, tile_size)
                        scaled_pixels = np.empty(tile_size * tile_size * img_copy.channels, dtype=np.float32)
                        img_copy.pixels.foreach_get(scaled_pixels)
                        scaled_arr = scaled_pixels.reshape(tile_size, tile_size, img_copy.channels)
                        # Convert to RGBA if needed
                        if img_copy.channels == 3:
                            rgba = np.ones((tile_size, tile_size, 4), dtype=np.float32)
                            rgba[:, :, :3] = scaled_arr
                            scaled_arr = rgba
                        elif img_copy.channels == 1:
                            rgba = np.ones((tile_size, tile_size, 4), dtype=np.float32)
                            rgba[:, :, 0] = scaled_arr[:, :, 0]
                            rgba[:, :, 1] = scaled_arr[:, :, 0]
                            rgba[:, :, 2] = scaled_arr[:, :, 0]
                            scaled_arr = rgba
                        if np.all(scaled_arr[:, :, 3] == 0):
                            scaled_arr = scaled_arr.copy()
                            scaled_arr[:, :, 3] = 1.0
                        src_arr = scaled_arr
                    except Exception as e:
                        # Fallback: nearest-neighbor via numpy
                        y_idx = (np.arange(tile_size) * src_h // tile_size).astype(np.int32)
                        x_idx = (np.arange(tile_size) * src_w // tile_size).astype(np.int32)
                        src_arr = src_arr[y_idx[:, None], x_idx[None, :]]

                # Write to atlas
                try:
                    atlas_arr[py:py+tile_size, px:px+tile_size] = src_arr
                    success += 1
                except Exception as e:
                    skipped.append(f"{img.name} (atlas write failed: {e})")
                    continue
        finally:
            for tc in temp_copies:
                try:
                    bpy.data.images.remove(tc)
                except Exception:
                    pass

        if success == 0:
            self.report({'ERROR'}, f"No textures could be read. Skipped: {'; '.join(skipped[:3])}")
            return {'CANCELLED'}

        if skipped:
            self.report({'WARNING'}, f"Skipped {len(skipped)}: {'; '.join(skipped[:2])}")

        # Create Blender atlas image
        atlas_img_name = self.atlas_name + (".tga" if self.file_format == 'TGA' else ".png")
        # Remove existing image with same name if any
        existing = bpy.data.images.get(atlas_img_name)
        if existing:
            bpy.data.images.remove(existing)
        atlas_img = bpy.data.images.new(atlas_img_name, atlas_size, atlas_size, alpha=True)

        # Write pixels — foreach_set is much faster and more reliable than = list
        flat = atlas_arr.flatten()
        try:
            atlas_img.pixels.foreach_set(flat)
        except Exception:
            atlas_img.pixels = flat.tolist()
        atlas_img.update()

        # Save to disk
        if not self.output_dir:
            self.report({'ERROR'}, "Pick an output folder")
            return {'CANCELLED'}
        out_folder = bpy.path.abspath(self.output_dir)
        if not os.path.isdir(out_folder):
            try:
                os.makedirs(out_folder, exist_ok=True)
            except Exception as e:
                self.report({'ERROR'}, f"Can't create folder: {e}")
                return {'CANCELLED'}

        out_path = os.path.join(out_folder, atlas_img_name)
        atlas_img.filepath_raw = out_path
        atlas_img.file_format = 'TARGA' if self.file_format == 'TGA' else 'PNG'
        try:
            atlas_img.save()
        except Exception as e:
            self.report({'ERROR'}, f"Save failed: {e}")
            return {'CANCELLED'}

        # Remember folder
        context.scene['rf_texture_folder'] = out_folder

        # Remap UVs
        if self.remap_uvs:
            for obj in meshes:
                mesh = obj.data
                uv_layer = mesh.uv_layers.active
                if not uv_layer:
                    continue
                for poly in mesh.polygons:
                    mat_idx = poly.material_index
                    if mat_idx >= len(obj.material_slots):
                        continue
                    mat = obj.material_slots[mat_idx].material
                    if not mat:
                        continue
                    # Find image for this material
                    img = None
                    if mat.use_nodes and mat.node_tree:
                        bsdf = mat.node_tree.nodes.get('Principled BSDF')
                        if bsdf and 'Base Color' in bsdf.inputs:
                            for link in bsdf.inputs['Base Color'].links:
                                if link.from_node.type == 'TEX_IMAGE' and link.from_node.image:
                                    img = link.from_node.image
                                    break
                        if not img:
                            for node in mat.node_tree.nodes:
                                if node.type == 'TEX_IMAGE' and node.image:
                                    img = node.image
                                    break
                    if img not in tile_positions:
                        continue
                    u_off, v_off, u_sc, v_sc, _, _ = tile_positions[img]
                    for loop_idx in poly.loop_indices:
                        old_u, old_v = uv_layer.data[loop_idx].uv
                        new_u = old_u * u_sc + u_off
                        new_v = old_v * v_sc + v_off
                        uv_layer.data[loop_idx].uv = (new_u, new_v)

        # Replace materials with atlas material
        if self.replace_materials:
            # Name the material EXACTLY like the atlas filename (without extension)
            # This ensures the material name matches the texture for V3C export
            atlas_mat_name = self.atlas_name
            atlas_mat = bpy.data.materials.get(atlas_mat_name)
            if atlas_mat:
                bpy.data.materials.remove(atlas_mat)
            atlas_mat = bpy.data.materials.new(atlas_mat_name)
            atlas_mat.use_nodes = True
            nodes = atlas_mat.node_tree.nodes
            bsdf = nodes.get('Principled BSDF')
            if not bsdf:
                bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.image = atlas_img
            tex_node.location = (-300, 300)
            atlas_mat.node_tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
            atlas_mat.use_backface_culling = False

            for obj in meshes:
                obj.data.materials.clear()
                obj.data.materials.append(atlas_mat)

        self.report({'INFO'},
            f"Atlas '{atlas_img_name}' saved — {n} textures → {atlas_size}x{atlas_size} (tile {tile_size}x{tile_size})")
        return {'FINISHED'}


class RFCHAR_OT_FixTextureNames(bpy.types.Operator):
    bl_idname = "rfchar.fix_texture_names"
    bl_label = "Sync Texture Names"
    bl_description = "Ensure each material's image texture filename ends in .tga and matches what the V3C exporter will write. Renames material to match image filename."
    bl_options = {'REGISTER', 'UNDO'}

    force_tga: BoolProperty(
        name="Force .tga extension",
        description="Change extensions like .png/.jpg/.dds on image names to .tga (filename only, not the actual file)",
        default=True,
    )
    sync_material_names: BoolProperty(
        name="Rename material to match texture",
        description="Rename each material to match its image's base filename (without extension)",
        default=True,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=400)

    def execute(self, context):
        # Find all mesh objects to process
        meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if not meshes and context.active_object and context.active_object.type == 'MESH':
            meshes = [context.active_object]
        # Also include armature's mesh children if an armature is active
        arm_obj = _find_rf_armature(context)
        if arm_obj:
            for c in arm_obj.children:
                if c.type == 'MESH' and c not in meshes and not c.hide_viewport:
                    meshes.append(c)
        if not meshes:
            self.report({'ERROR'}, "No meshes found — select a mesh or armature")
            return {'CANCELLED'}

        fixed_imgs = 0
        fixed_mats = 0
        scanned_mats = 0
        issues = []

        seen_materials = set()
        for obj in meshes:
            for slot in obj.material_slots:
                mat = slot.material
                if not mat or mat.name in seen_materials:
                    continue
                seen_materials.add(mat.name)
                scanned_mats += 1

                # Find the image in this material
                img = None
                if mat.use_nodes and mat.node_tree:
                    # Prefer Base Color connected image
                    bsdf = mat.node_tree.nodes.get('Principled BSDF')
                    if bsdf and 'Base Color' in bsdf.inputs:
                        for link in bsdf.inputs['Base Color'].links:
                            if link.from_node.type == 'TEX_IMAGE' and link.from_node.image:
                                img = link.from_node.image
                                break
                    if not img:
                        for node in mat.node_tree.nodes:
                            if node.type == 'TEX_IMAGE' and node.image:
                                img = node.image
                                break

                if not img:
                    issues.append(f"'{mat.name}' has no image texture")
                    continue

                # Force .tga extension on image name
                old_img_name = img.name
                if self.force_tga:
                    base = os.path.splitext(img.name)[0]
                    new_img_name = base + '.tga'
                    if img.name != new_img_name:
                        img.name = new_img_name
                        fixed_imgs += 1

                # Sync material name to match image (without extension)
                if self.sync_material_names:
                    base_name = os.path.splitext(img.name)[0]
                    if mat.name != base_name:
                        # Check for conflicts
                        existing = bpy.data.materials.get(base_name)
                        if existing and existing != mat:
                            issues.append(f"Can't rename '{mat.name}' → '{base_name}' (name taken)")
                        else:
                            mat.name = base_name
                            fixed_mats += 1

        lines = [f"Scanned {scanned_mats} materials"]
        if fixed_imgs:
            lines.append(f"Renamed {fixed_imgs} image(s) to .tga extension")
        if fixed_mats:
            lines.append(f"Renamed {fixed_mats} material(s) to match texture")
        for i in issues:
            lines.append(f"⚠ {i}")

        self.report({'INFO'}, " · ".join(lines))
        return {'FINISHED'}


class RFCHAR_PT_MainPanel(bpy.types.Panel):
    bl_label = "RF Character"
    bl_idname = "RFCHAR_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "RF Character"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        arm_obj = _find_rf_armature(context)

        # ── Info ──
        if arm_obj:
            box = layout.box()
            col = box.column(align=True)
            col.label(text=arm_obj.name, icon='ARMATURE_DATA')
            mesh_children = [c for c in arm_obj.children if c.type == 'MESH' and not c.hide_viewport]
            info = f"{len(arm_obj.data.bones)} bones"
            if mesh_children:
                names = ', '.join(c.name for c in mesh_children[:3])
                info += f"  ·  {names}"
            col.label(text=info)
            rf_actions = _get_rf_actions(arm_obj)
            extras = []
            if rf_actions:
                extras.append(f"{len(rf_actions)} anims")
            pp_count = len([c for c in arm_obj.children if c.type == 'EMPTY' and c.get('rf_prop_point')])
            cs_count = len([c for c in arm_obj.children if c.type == 'EMPTY' and c.name.startswith('CS_')])
            if pp_count:
                extras.append(f"{pp_count} props")
            if cs_count:
                extras.append(f"{cs_count} hitboxes")
            if extras:
                col.label(text="  ·  ".join(extras))

        # ── Import ──
        layout.separator()
        row = layout.row(align=True)
        row.operator("import_scene.rf_v3c", text="V3C", icon='IMPORT')
        row.operator("import_scene.rf_rfa", text="RFA", icon='ACTION')
        if arm_obj:
            row = layout.row(align=True)
            row.operator("rfchar.browse_anim_folder", text="Batch Anims", icon='FILE_FOLDER')
            row.operator("rfchar.find_textures", text="Find Textures", icon='VIEWZOOM')
        else:
            layout.operator("rfchar.find_textures", text="Find Textures", icon='VIEWZOOM')

        # ── Master Animation Folder ──
        layout.separator()
        box = layout.box()
        box.label(text="Master Animation Folder", icon='BOOKMARKS')
        master = _get_master_folder(context)
        if master:
            row = box.row(align=True)
            row.label(text=os.path.basename(master.rstrip(os.sep)) or master, icon='CHECKMARK')
            row.operator("rfchar.set_master_folder", text="", icon='FILEBROWSER')
            row.operator("rfchar.clear_master_folder", text="", icon='X')
            box.prop(scn, 'rfchar_auto_load_anims')
        else:
            box.operator("rfchar.set_master_folder", text="Set Folder...", icon='FILE_FOLDER')
            box.label(text="Auto-loads anims when you import a V3C", icon='INFO')

        # ── Recent Files ──
        recent = list(scn.get('rf_recent_v3c', []))
        if recent:
            box = layout.box()
            row = box.row()
            row.label(text="Recent V3Cs", icon='FILE_REFRESH')
            row.operator("rfchar.clear_recent", text="", icon='X', emboss=False)
            col = box.column(align=True)
            for fp in recent[:5]:
                name = os.path.basename(fp)
                op = col.operator("rfchar.import_recent", text=name, icon='FILE')
                op.filepath = fp

        # ── Batch File List (only when files scanned) ──
        if arm_obj and len(scn.rfchar_anim_files) > 0:
            layout.separator()
            row = layout.row(align=True)
            row.label(text=f"{len(scn.rfchar_anim_files)} files")
            row.operator("rfchar.select_all_anims", text="All")
            row.operator("rfchar.deselect_all_anims", text="None")

            layout.template_list(
                "RFCHAR_UL_AnimList", "",
                scn, "rfchar_anim_files",
                scn, "rfchar_anim_list_index",
                rows=5
            )

            selected_count = sum(1 for f in scn.rfchar_anim_files if f.selected)
            row = layout.row(align=True)
            row.operator("rfchar.import_selected_anims",
                        text=f"Import ({selected_count})", icon='IMPORT')
            row.operator("rfchar.clear_anim_list", text="", icon='X')

        # ── Custom Mesh ──
        if arm_obj:
            layout.separator()
            box = layout.box()
            box.label(text="Custom Mesh", icon='MESH_DATA')
            col = box.column(align=True)
            col.operator("rfchar.import_custom_mesh", text="Import Mesh", icon='FILE_FOLDER')
            col.separator()
            col.operator("rfchar.bind_to_armature", icon='CONSTRAINT_BONE')
            col.operator("rfchar.transfer_weights", icon='MOD_DATA_TRANSFER')
            col.operator("rfchar.check_weights", icon='VIEWZOOM')
            col.separator()
            col.operator("rfchar.generate_atlas", text="Combine Textures → Atlas", icon='TEXTURE')
            col.operator("rfchar.fix_texture_names", text="Sync Texture Names", icon='FILE_REFRESH')

        # ── Visibility ──
        if arm_obj:
            layout.separator()
            box = layout.box()
            box.label(text="Visibility", icon='HIDE_OFF')
            row = box.row(align=True)
            row.operator("rfchar.toggle_rf_mesh", text="Mesh", icon='MESH_DATA')
            row.operator("rfchar.toggle_hitboxes", text="Hitboxes", icon='MESH_UVSPHERE')
            row = box.row(align=True)
            row.operator("rfchar.toggle_prop_points", text="Props", icon='EMPTY_ARROWS')
            row.operator("rfchar.toggle_lods", text="LODs", icon='MOD_DECIM')

        # ── Export ──
        if arm_obj:
            layout.separator()
            box = layout.box()
            box.label(text="Export", icon='EXPORT')
            col = box.column(align=True)
            col.operator("rfchar.validate_export", text="Validate", icon='CHECKMARK')
            col.operator("export_scene.rf_v3c", text="Export V3C", icon='MESH_DATA')
            col.operator("rfchar.batch_export_v3c", text="Batch Export", icon='FILE_FOLDER')
            col.separator()
            active_action = arm_obj.animation_data.action if arm_obj.animation_data else None
            if active_action:
                col.operator("export_scene.rf_rfa",
                           text=f"Export '{active_action.name}'", icon='ACTION')
            else:
                col.label(text="No active animation", icon='INFO')


class RFCHAR_PT_LoadedAnimsPanel(bpy.types.Panel):
    bl_label = "Loaded Animations"
    bl_idname = "RFCHAR_PT_loaded_anims"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "RF Character"
    bl_parent_id = "RFCHAR_PT_main"

    @classmethod
    def poll(cls, context):
        arm_obj = _find_rf_armature(context)
        return arm_obj is not None and _get_rf_actions(arm_obj)

    def draw_header(self, context):
        arm_obj = _find_rf_armature(context)
        rf_actions = _get_rf_actions(arm_obj) if arm_obj else {}
        self.layout.label(text=f"({len(rf_actions)})")

    def draw(self, context):
        layout = self.layout
        arm_obj = _find_rf_armature(context)
        rf_actions = _get_rf_actions(arm_obj)
        active_action = arm_obj.animation_data.action if arm_obj.animation_data else None
        active_name = active_action.name if active_action else ""

        # Rest pose button
        layout.operator("rfchar.rest_pose", text="Rest Pose", icon='ARMATURE_DATA')

        # Search box
        row = layout.row(align=True)
        row.prop(context.scene, 'rfchar_anim_search', text="", icon='VIEWZOOM')

        # Grouping toggle
        row = layout.row(align=True)
        row.prop(context.scene, 'rfchar_anim_group_by', text="Group by", expand=True)

        search = (context.scene.rfchar_anim_search or '').lower()
        group_by = context.scene.rfchar_anim_group_by

        def categorize(name):
            n = name.lower()
            weapons = {
                '12mm': ('12mm', 'hg_', '_hg'),
                'Assault Rifle': ('_ar_', 'arifle', 'ar_'),
                'Shotgun': ('_sg_', 'shotgun', 'sg_'),
                'Sniper Rifle': ('_sr_', 'sniper'),
                'Rocket Launcher': ('_rl_', '_rl.', 'rl_'),
                'Flamethrower': ('_ft_', 'flamethrower', 'ft_'),
                'Machine Pistol': ('_mp_', '_smc_', 'smw_'),
                'Heavy MG': ('_hmac_', 'hmac'),
                'Grenade': ('_grn_', '_gren_', 'grenade'),
                'Rail Driver': ('_rr_', 'rrif', 'rail'),
                'Riot Stick': ('riotstick', 'rstick', '_rs_'),
                'Riot Shield': ('rshield', 'riotshield'),
                'Remote Charge': ('rcharge', 'remotecharge', '_rc_'),
                'Scope AR': ('_sar_', 'scope'),
                'Shoulder Cannon': ('shoulder', '_sc_'),
            }
            for cat, patterns in weapons.items():
                if any(p in n for p in patterns):
                    return cat
            if 'death' in n or '_dead' in n: return 'Death'
            if 'flinch' in n or 'blast' in n or 'hit_' in n: return 'Flinch/Hit'
            if 'corpse' in n: return 'Corpse'
            if any(s in n for s in ('walk', 'run', 'stand', 'crouch', 'idle', 'swim', 'jump', 'freefall', 'sidestep')):
                return 'Movement'
            if 'talk' in n or 'cower' in n or 'alarm' in n: return 'Voice/Idle'
            return 'Other'

        # Filter
        filtered_names = [n for n in rf_actions.keys() if not search or search in n.lower()]

        if not filtered_names:
            layout.label(text="No matches", icon='INFO')
            return

        if group_by == 'CATEGORY':
            groups = {}
            for name in filtered_names:
                cat = categorize(name)
                groups.setdefault(cat, []).append(name)
            cat_order = ['Movement', '12mm', 'Assault Rifle', 'Shotgun', 'Sniper Rifle',
                        'Rocket Launcher', 'Flamethrower', 'Machine Pistol', 'Heavy MG',
                        'Scope AR', 'Grenade', 'Rail Driver', 'Riot Stick', 'Riot Shield',
                        'Remote Charge', 'Shoulder Cannon', 'Death', 'Flinch/Hit',
                        'Corpse', 'Voice/Idle', 'Other']
            for cat in cat_order:
                if cat not in groups:
                    continue
                names = sorted(groups[cat])
                box = layout.box()
                box.label(text=f"{cat} ({len(names)})", icon='DISCLOSURE_TRI_DOWN')
                col = box.column(align=True)
                for aname in names:
                    self._draw_anim_button(col, aname, active_name)
        else:
            col = layout.column(align=True)
            for aname in sorted(filtered_names):
                self._draw_anim_button(col, aname, active_name)

    def _draw_anim_button(self, col, aname, active_name):
        is_active = (aname == active_name)
        row = col.row(align=True)
        op = row.operator("rfchar.set_active_action", text=aname,
                         icon='PLAY' if is_active else 'DOT',
                         depress=is_active)
        op.action_name = aname
        row.operator("rfchar.delete_animation", text="", icon='X').action_name = aname


class RFCHAR_PT_RequiredAnimsPanel(bpy.types.Panel):
    bl_label = "Required Animations"
    bl_idname = "RFCHAR_PT_required_anims"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "RF Character"
    bl_parent_id = "RFCHAR_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        arm_obj = _find_rf_armature(context)
        return arm_obj is not None and arm_obj.get('rf_required_anims')

    def draw_header(self, context):
        arm_obj = _find_rf_armature(context)
        req_json = arm_obj.get('rf_required_anims', '[]') if arm_obj else '[]'
        required = json.loads(req_json) if isinstance(req_json, str) else []
        rf_actions = _get_rf_actions(arm_obj) if arm_obj else {}
        loaded_names = set()
        for aname in rf_actions.keys():
            loaded_names.add(aname.lower())
            loaded_names.add(aname.lower() + '.rfa')
            if aname.lower().endswith('.rfa'):
                loaded_names.add(aname.lower()[:-4])
        loaded_count = sum(1 for r in required if r.lower() in loaded_names
                          or os.path.splitext(r)[0].lower() in loaded_names)
        self.layout.label(text=f"({loaded_count}/{len(required)})")

    def draw(self, context):
        layout = self.layout
        arm_obj = _find_rf_armature(context)
        req_json = arm_obj.get('rf_required_anims', '[]')
        required = json.loads(req_json) if isinstance(req_json, str) else []
        if not required:
            return

        # Get loaded action names (lowercase, stripped of extension for matching)
        rf_actions = _get_rf_actions(arm_obj)
        loaded_names = set()
        for aname in rf_actions.keys():
            loaded_names.add(aname.lower())
            loaded_names.add(aname.lower() + '.rfa')
            if aname.lower().endswith('.rfa'):
                loaded_names.add(aname.lower()[:-4])

        # Count loaded vs total
        loaded_count = sum(1 for r in required if r.lower() in loaded_names
                          or os.path.splitext(r)[0].lower() in loaded_names)
        total = len(required)
        missing = total - loaded_count

        # Summary
        box = layout.box()
        row = box.row()
        if missing == 0:
            row.label(text=f"All {total} animations loaded")
        else:
            row.label(text=f"{loaded_count}/{total} loaded, {missing} missing",
                     icon='ERROR' if missing > 0 else 'NONE')

        # Load / Delete buttons
        row = box.row(align=True)
        if missing > 0:
            row.operator("rfchar.load_required_anims",
                        text=f"Load Missing ({missing})", icon='IMPORT')
        if loaded_count > 0:
            row.operator("rfchar.delete_all_animations",
                        text=f"Delete All ({loaded_count})", icon='TRASH')

        # Show remembered folder (if any) with change option
        folder = arm_obj.get('rf_anim_folder') or context.scene.get('rf_last_anim_folder')
        if folder:
            row = box.row(align=True)
            row.label(text=os.path.basename(folder.rstrip(os.sep)) or folder, icon='FILE_FOLDER')
            op = row.operator("rfchar.load_required_anims", text="", icon='FILEBROWSER')
            op.force_browse = True

        # Master folder status
        try:
            master = _get_master_folder(context)
            row = box.row(align=True)
            if master:
                row.label(text=f"Master: {os.path.basename(master.rstrip(os.sep)) or master}",
                         icon='BOOKMARKS')
                if missing > 0:
                    row.operator("rfchar.autoload_from_master", text="Auto-Load", icon='IMPORT')
            else:
                row.operator("rfchar.set_master_folder", text="Set Master Folder...", icon='FILE_FOLDER')
        except Exception:
            pass

        # Toggle between full list and missing-only (collapsible via bl_options)
        row = box.row(align=True)
        row.prop(context.scene, 'rfchar_required_filter', expand=True)

        req_filter = context.scene.rfchar_required_filter

        # Flat checklist
        col = box.column(align=True)
        for rfa in sorted(required):
            name_no_ext = os.path.splitext(rfa)[0]
            name_no_ext_lc = name_no_ext.lower()
            is_loaded = (rfa.lower() in loaded_names or name_no_ext_lc in loaded_names)

            if req_filter == 'MISSING' and is_loaded:
                continue
            if req_filter == 'LOADED' and not is_loaded:
                continue

            icon = 'CHECKBOX_HLT' if is_loaded else 'CHECKBOX_DEHLT'
            col.label(text=name_no_ext, icon=icon)


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def _menu_import(self, context):
    self.layout.operator(RFCHAR_OT_ImportV3C.bl_idname, text="RF Character Mesh (.v3c/.v3m)")
    self.layout.operator(RFCHAR_OT_ImportRFA.bl_idname, text="RF Animation (.rfa)")

def _menu_export(self, context):
    self.layout.operator(RFCHAR_OT_ExportV3C.bl_idname, text="RF Character Mesh (.v3c)")
    self.layout.operator(RFCHAR_OT_ExportRFA.bl_idname, text="RF Animation (.rfa)")

_classes = (
    RFCHAR_AnimFileItem,
    RFCHAR_UL_AnimList,
    RFCHAR_OT_ImportV3C,
    RFCHAR_OT_ImportRFA,
    RFCHAR_OT_BrowseAnimFolder,
    RFCHAR_OT_ImportSelectedAnims,
    RFCHAR_OT_SelectAllAnims,
    RFCHAR_OT_DeselectAllAnims,
    RFCHAR_OT_ClearAnimList,
    RFCHAR_OT_SetActiveAction,
    RFCHAR_OT_ImportCustomMesh,
    RFCHAR_OT_BindToArmature,
    RFCHAR_OT_TransferWeights,
    RFCHAR_OT_ToggleRFMesh,
    RFCHAR_OT_ToggleHitboxes,
    RFCHAR_OT_TogglePropPoints,
    RFCHAR_OT_ToggleLODs,
    RFCHAR_OT_RestPose,
    RFCHAR_OT_DeleteAnimation,
    RFCHAR_OT_DeleteAllAnimations,
    RFCHAR_OT_CheckWeights,
    RFCHAR_OT_FindTextures,
    RFCHAR_OT_LoadRequiredAnims,
    RFCHAR_OT_AutoLoadFromMaster,
    RFCHAR_OT_SetMasterFolder,
    RFCHAR_OT_ClearMasterFolder,
    RFCHAR_OT_ImportRecent,
    RFCHAR_OT_ClearRecent,
    RFCHAR_OT_ValidateExport,
    RFCHAR_OT_RenameActionsToDB,
    RFCHAR_OT_BatchExportV3C,
    RFCHAR_OT_GenerateAtlas,
    RFCHAR_OT_FixTextureNames,
    RFCHAR_OT_ExportRFA,
    RFCHAR_OT_ExportAllRFA,
    RFCHAR_OT_ExportV3C,
    RFCHAR_PT_WorkflowPanel,
    RFCHAR_PT_MainPanel,
    RFCHAR_PT_LoadedAnimsPanel,
    RFCHAR_PT_RequiredAnimsPanel,
)

def register():
    print(f"[RF Character] Registering as __name__='{__name__}'")
    # Register classes, skipping any already registered (handles reload scenarios)
    for c in _classes:
        try:
            bpy.utils.register_class(c)
        except (ValueError, RuntimeError):
            # Already registered — try re-registering after unregister
            try:
                bpy.utils.unregister_class(c)
                bpy.utils.register_class(c)
            except Exception as e:
                print(f"[RF Character] Register failed for {c.__name__}: {e}")

    try:
        bpy.types.TOPBAR_MT_file_import.append(_menu_import)
    except Exception:
        pass
    try:
        bpy.types.TOPBAR_MT_file_export.append(_menu_export)
    except Exception:
        pass

    # Scene properties (safe to re-assign)
    bpy.types.Scene.rfchar_anim_dir = StringProperty(
        name="Animation Folder",
        subtype='DIR_PATH',
        default="",
        update=_auto_scan_dir
    )
    bpy.types.Scene.rfchar_anim_files = CollectionProperty(type=RFCHAR_AnimFileItem)
    bpy.types.Scene.rfchar_anim_list_index = IntProperty(name="Index", default=0)
    bpy.types.Scene.rfchar_anim_search = StringProperty(
        name="Search", description="Filter animations by name", default="")
    bpy.types.Scene.rfchar_anim_show_loaded = BoolProperty(
        name="Show loaded", description="Show already-loaded animations", default=True)
    bpy.types.Scene.rfchar_anim_show_missing = BoolProperty(
        name="Show missing", description="Show not-yet-loaded animations", default=True)
    bpy.types.Scene.rfchar_anim_group_by = EnumProperty(
        name="Group",
        items=[('NONE', "None", "Flat list"),
               ('CATEGORY', "Category", "Group by weapon/state type")],
        default='CATEGORY',
    )
    bpy.types.Scene.rfchar_required_filter = EnumProperty(
        name="Show",
        items=[('ALL', "All", "Show all required animations"),
               ('MISSING', "Missing", "Show only not-yet-loaded"),
               ('LOADED', "Loaded", "Show only loaded")],
        default='ALL',
    )
    bpy.types.Scene.rfchar_auto_load_anims = BoolProperty(
        name="Auto-load animations on V3C import",
        description="When enabled, importing a V3C automatically imports its required animations from the master folder",
        default=True,
    )


def unregister():
    # Remove menu hooks safely
    for menu_attr, fn in [('TOPBAR_MT_file_export', _menu_export),
                          ('TOPBAR_MT_file_import', _menu_import)]:
        try:
            getattr(bpy.types, menu_attr).remove(fn)
        except Exception:
            pass

    # Remove scene properties safely
    for prop in ('rfchar_auto_load_anims', 'rfchar_required_filter', 'rfchar_anim_group_by',
                 'rfchar_anim_show_missing', 'rfchar_anim_show_loaded',
                 'rfchar_anim_search', 'rfchar_anim_list_index',
                 'rfchar_anim_files', 'rfchar_anim_dir'):
        try:
            delattr(bpy.types.Scene, prop)
        except Exception:
            pass

    # Unregister classes safely
    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass

if __name__ == "__main__":
    register()
