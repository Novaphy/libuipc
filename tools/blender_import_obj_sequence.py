#!/usr/bin/env python3
"""
将 corex_demo 导出的 OBJ 序列导入 Blender，并按帧切换可见性，便于在时间轴上预览与渲染。

适用：Blender 3.4+ / 4.x / 5.x（使用 wm.obj_import）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
用法一：在 Blender 里运行（推荐先试用）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 打开 Blender，切到「脚本」工作区，或 Text Editor。
2. 打开本脚本，修改下方 CONFIG 中的 obj_dir（或使用环境变量 UIPC_OBJ_SEQUENCE_DIR）。
3. 点击「运行脚本」。

Windows 路径写法（任选其一，避免 SyntaxError）：
  - 使用原始字符串前缀 r：  r"C:\\Users\\..."
  - 或全部用正斜杠：         "C:/Users/China/Desktop/..."
  切勿写普通字符串 "C:\\Users\\..." 时漏掉 r，否则 \\U 会被当成 Unicode 转义而报错。

用法二：命令行后台导入（无界面，适合批处理）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  blender --background --python blender_import_obj_sequence.py -- "D:/sim_output/my_scene"

  Linux / macOS:
  blender --background --python blender_import_obj_sequence.py -- "/path/to/obj/folder"

用法三：环境变量指定目录
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  set UIPC_OBJ_SEQUENCE_DIR=D:\\sim_output\\slope
  blender --background --python blender_import_obj_sequence.py

参数（均跟在单独一个 -- 之后）：
  第 1 个位置参数     OBJ 所在目录（内含 scene_surface_0000.obj 等）
  --fps N             帧率，默认 24
  --pattern GLOB      匹配文件名，默认 scene_surface_*.obj
  --no-clear          不删除场景里已有物体（默认会清空场景）
  --collection NAME   把所有帧放进指定集合，默认 ObjSequence

输出文件约定（与 corex_demo 一致）：
  scene_surface_0000.obj, scene_surface_0001.obj, ...

注意：
  - 每帧一个独立网格物体，帧数多时会占用较多内存；一般演示（几十～几百帧）可接受。
  - 若需极低内存，可改用 Alembic 等管线；本脚本专注「零依赖、直接吃 OBJ 序列」。
  - 若曾出现「点运行后 Blender 卡住」：旧版对每帧×每物体打关键帧（O(n²)）会极慢；请更新到当前脚本（每物体约 3 个关键帧）。
  - 导入阶段仍会随 OBJ 数量变久（控制台会打印进度）；上千帧时请耐心等待或先减少 --frames 再导出。
"""

from __future__ import annotations

import glob
import os
import re
import sys

# ---------------------------------------------------------------------------
# 在 Blender GUI 里直接「运行脚本」时改这里；命令行参数会覆盖此目录。
# Windows：必须用 r"..." 或 "C:/Users/..."，不能写裸的 "C:\Users\..."（\U 会语法错误）。
# ---------------------------------------------------------------------------
CONFIG = {
    "obj_dir": "",  # 例: r"C:\Users\China\Desktop\corex_scene_outputs\domino"
    # 或: "C:/Users/China/Desktop/corex_scene_outputs/domino"
    "fps": 24,
    "pattern": "scene_surface_*.obj",
    "clear_scene": True,
    "collection_name": "ObjSequence",
}


def _parse_args_after_double_dash(argv: list[str]) -> tuple[dict, list[str]]:
    """解析 blender --python script.py -- ... 之后的参数。"""
    cfg = dict(CONFIG)
    extra: list[str] = []
    if "--" not in argv:
        return cfg, extra
    i = argv.index("--") + 1
    rest = argv[i:]
    pos: list[str] = []
    j = 0
    while j < len(rest):
        a = rest[j]
        if a == "--fps" and j + 1 < len(rest):
            cfg["fps"] = max(1, int(rest[j + 1]))
            j += 2
            continue
        if a.startswith("--fps="):
            cfg["fps"] = max(1, int(a.split("=", 1)[1]))
            j += 1
            continue
        if a == "--pattern" and j + 1 < len(rest):
            cfg["pattern"] = rest[j + 1]
            j += 2
            continue
        if a.startswith("--pattern="):
            cfg["pattern"] = a.split("=", 1)[1]
            j += 1
            continue
        if a == "--no-clear":
            cfg["clear_scene"] = False
            j += 1
            continue
        if a == "--collection" and j + 1 < len(rest):
            cfg["collection_name"] = rest[j + 1]
            j += 2
            continue
        if a.startswith("--collection="):
            cfg["collection_name"] = a.split("=", 1)[1]
            j += 1
            continue
        if not a.startswith("-"):
            pos.append(a)
        j += 1
    if pos:
        cfg["obj_dir"] = pos[0]
    return cfg, extra


def _natural_sort_key(path: str) -> tuple:
    """scene_surface_2.obj 排在 scene_surface_10.obj 前。"""
    base = os.path.basename(path)
    parts = re.split(r"(\d+)", base)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p)
    return tuple(key)


def _iter_action_fcurves(action):
    """Blender 5.x 分层 Action：FCurve 在 layer → strip → channelbag → fcurves；旧版在 action.fcurves。"""
    if action is None:
        return
    # 优先遍历分层结构（Blender 4.4+ / 5.x）
    seen_any = False
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            bag = getattr(strip, "channelbag", None)
            if bag is not None:
                fcs = getattr(bag, "fcurves", None)
                if fcs is not None:
                    seen_any = True
                    yield from fcs
            else:
                fcs = getattr(strip, "fcurves", None)
                if fcs is not None:
                    seen_any = True
                    yield from fcs
    if seen_any:
        return
    # 旧版：action.fcurves
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        yield from legacy


def _set_hide_fcurves_constant(ob) -> None:
    """布尔 hide 关键帧用 CONSTANT，避免中间帧插值异常。"""
    ad = ob.animation_data
    if not ad or not ad.action:
        return
    try:
        for fc in _iter_action_fcurves(ad.action):
            dp = getattr(fc, "data_path", "") or ""
            if "hide" not in dp:
                continue
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"
    except Exception:
        # 若 API 再变，跳过插值修正，动画仍可播放
        pass


def main() -> None:
    try:
        import bpy  # type: ignore
    except ImportError:
        print(
            "错误：请在 Blender 内运行本脚本（菜单 脚本 → 运行脚本），或使用：\n"
            "  blender --background --python blender_import_obj_sequence.py -- <OBJ目录>",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg, _ = _parse_args_after_double_dash(sys.argv)

    obj_dir = cfg["obj_dir"] or os.environ.get("UIPC_OBJ_SEQUENCE_DIR", "")
    if not obj_dir or not os.path.isdir(obj_dir):
        print(
            "请指定 OBJ 目录：\n"
            "  1) 修改脚本内 CONFIG['obj_dir']，或\n"
            "  2) 设置环境变量 UIPC_OBJ_SEQUENCE_DIR，或\n"
            "  3) blender --python ... -- \"D:/path/to/folder\"",
            file=sys.stderr,
        )
        sys.exit(1)

    obj_dir = os.path.abspath(obj_dir)
    pattern = os.path.join(obj_dir, cfg["pattern"])
    files = sorted(glob.glob(pattern), key=_natural_sort_key)
    if not files:
        print(f"未找到匹配文件: {pattern}", file=sys.stderr)
        sys.exit(1)

    fps = int(cfg["fps"])
    clear_scene = bool(cfg["clear_scene"])
    coll_name = cfg["collection_name"]

    import bpy  # noqa: F811

    # 清空场景
    if clear_scene:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        # 清理孤立数据块（可选）
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1

    # 集合
    if coll_name in bpy.data.collections:
        col = bpy.data.collections[coll_name]
    else:
        col = bpy.data.collections.new(coll_name)
        bpy.context.scene.collection.children.link(col)

    n_files = len(files)
    wm = getattr(bpy.context, "window_manager", None)
    use_progress = wm is not None and not getattr(bpy.app, "background", False)
    if use_progress:
        try:
            wm.progress_begin(0, n_files)
        except Exception:
            use_progress = False

    imported: list = []
    try:
        for idx, fpath in enumerate(files):
            if use_progress:
                wm.progress_update(idx)
            if idx == 0 or (idx + 1) % max(1, n_files // 20) == 0 or idx == n_files - 1:
                print(f"[obj_import] {idx + 1} / {n_files}  {os.path.basename(fpath)}")
            try:
                bpy.ops.wm.obj_import(filepath=fpath)
            except Exception as e:
                print(f"导入失败 {fpath}: {e}", file=sys.stderr)
                raise
            objs = list(bpy.context.selected_objects)
            for ob in objs:
                ob.name = f"Frame_{idx:04d}"
                for c in ob.users_collection:
                    c.objects.unlink(ob)
                col.objects.link(ob)
                imported.append(ob)
    finally:
        if use_progress:
            wm.progress_end()

    n = len(imported)
    scene.frame_end = n

    # 旧版：对每一帧给全部物体打关键帧 → O(n²)，几百帧就会卡死。改为每个物体最多 3 个关键帧 → O(n)。
    # 第 i 个物体（0 起）仅在第 i+1 帧显示。
    print(f"[keyframes] 正在为 {n} 个物体写入时间轴（约 {3 * n} 个关键帧）…")
    for ob in imported:
        ob.hide_viewport = True
        ob.hide_render = True

    for i, ob in enumerate(imported):
        if i > 0:
            scene.frame_set(i)
            ob.hide_viewport = True
            ob.hide_render = True
            ob.keyframe_insert(data_path="hide_viewport", frame=i)
            ob.keyframe_insert(data_path="hide_render", frame=i)
        scene.frame_set(i + 1)
        ob.hide_viewport = False
        ob.hide_render = False
        ob.keyframe_insert(data_path="hide_viewport", frame=i + 1)
        ob.keyframe_insert(data_path="hide_render", frame=i + 1)
        if i < n - 1:
            scene.frame_set(i + 2)
            ob.hide_viewport = True
            ob.hide_render = True
            ob.keyframe_insert(data_path="hide_viewport", frame=i + 2)
            ob.keyframe_insert(data_path="hide_render", frame=i + 2)
        _set_hide_fcurves_constant(ob)

    scene.frame_set(1)
    if imported:
        bpy.context.view_layer.objects.active = imported[0]

    print(f"OK: 已导入 {n} 个 OBJ，目录: {obj_dir}")
    print(f"    时间轴: 1 ~ {n}，fps={fps}，集合: {coll_name}")
    print("    可直接拖动时间轴预览，或设置输出为图像序列/视频后渲染。")


if __name__ == "__main__":
    main()
