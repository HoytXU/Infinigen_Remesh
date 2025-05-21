# remesh_worker.py
import bpy
import sys
import time

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def remesh(obj, voxel_size=0.002):
    modifier = obj.modifiers.new(name="Remesh", type='REMESH')
    modifier.mode = 'VOXEL'
    modifier.voxel_size = voxel_size
    modifier.use_smooth_shade = False
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Remesh")

# def decimate(obj, ratio=0.3):
#     modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
#     modifier.ratio = ratio
#     bpy.context.view_layer.objects.active = obj
#     bpy.ops.object.modifier_apply(modifier="Decimate")

def auto_decimate(obj, target_faces=50_000):
    face_count = len(obj.data.polygons)

    if face_count <= target_faces:
        print(f"[AutoDecimate] {obj.name} | faces={face_count} ✅ ≤ {target_faces}, skipping.")
        return

    ratio = target_faces / face_count
    print(f"[AutoDecimate] {obj.name} | faces={face_count} → ratio={ratio:.8f}")

    mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    mod.ratio = ratio
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Decimate")
    print(f"[AutoDecimate] {obj.name} | final face count: {len(obj.data.polygons)}")


argv = sys.argv
argv = argv[argv.index("--") + 1:]
input_path = argv[0]
output_path = argv[1]
voxel_size = float(argv[2])

start_total = time.time()

clear_scene()

# === 1. 导入 .obj ===
start_import = time.time()
bpy.ops.import_scene.obj(filepath=input_path)
obj = bpy.context.selected_objects[0]
end_import = time.time()

# === 2. Remesh ===
start_remesh = time.time()
remesh(obj, voxel_size=voxel_size)
end_remesh = time.time()

# === 3. Decimate ===
start_decimate = time.time()
auto_decimate(obj)
end_decimate = time.time()

# === 4. 导出 ===
start_export = time.time()
bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)
end_export = time.time()

# === 总结报告 ===
end_total = time.time()
print(f"[DONE] {input_path} → {output_path}")

print("\n⏱️ Timing Breakdown:")
print(f"📂 Import OBJ     : {end_import - start_import:.2f} sec")
print(f"🔄 Remesh         : {end_remesh - start_remesh:.2f} sec")
print(f"🪓 Decimate       : {end_decimate - start_decimate:.2f} sec")
print(f"💾 Export OBJ     : {end_export - start_export:.2f} sec")
print(f"🧠 TOTAL TIME     : {end_total - start_total:.2f} sec\n")

