import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import (
  INIT_STATE,
  SQuRO_XML,
  get_spec,
)
from mjlab.scripts._diagnostics import DIAGNOSTICS_ROOT


@dataclass(frozen=True)
class GroupSpec:
  key: str
  label: str
  anchor: str
  bodies: tuple[str, ...]
  prefixes: tuple[str, ...]
  color: tuple[int, int, int]


# 五组完整划分见 docs/SQuRo_五盒几何素材.md。
GROUPS = (
  GroupSpec(
    "forebody",
    "Forebody + forelegs",
    "F_body_Link",
    ("F_body_Link",),
    ("FL_", "FR_"),
    (70, 104, 168),
  ),
  GroupSpec(
    "anterior_spine",
    "Anterior spine",
    "F_spine1_Link",
    ("base_Link", "FU_spine_Link", "FD_spine_Link", "F_spine1_Link", "F_spine2_Link"),
    (),
    (118, 87, 163),
  ),
  GroupSpec(
    "posterior_spine",
    "Posterior spine",
    "H_spine1_Link",
    ("HL_spine_Link", "HR_spine_Link", "H_spine1_Link", "H_spine2_Link"),
    (),
    (49, 133, 150),
  ),
  GroupSpec(
    "hindbody",
    "Hindbody + hindlegs",
    "H_body_Link",
    ("H_body_Link",),
    ("HL_", "HR_"),
    (93, 123, 161),
  ),
  GroupSpec(
    "head",
    "Head + neck",
    "Neck_pitch_Link",
    ("Neck_yaw_Link", "Neck_pitch_Link"),
    (),
    (143, 110, 171),
  ),
)
BOX_SIGNS = np.asarray(tuple(product((-1, 1), repeat=3)))
BOX_EDGES = tuple(
  (i, j)
  for i in range(8)
  for j in range(i + 1, 8)
  if np.count_nonzero(BOX_SIGNS[i] != BOX_SIGNS[j]) == 1
)
BOX_FACES = tuple(
  tuple(np.flatnonzero(BOX_SIGNS[:, axis] == sign))
  for axis in range(3)
  for sign in (-1, 1)
)
VIEWS = (
  ("oblique", "Oblique view", 135.0, -24.0),
  ("top", "Top view (xy)", 90.0, -90.0),
  ("side", "Side view (xz)", 90.0, 0.0),
  ("front", "Front view (yz)", 180.0, 0.0),
)


# 按名称完整归属机器人网格，脊柱名称优先于腿部前缀。
def resolve_group_geoms(model: mujoco.MjModel) -> list[list[int]]:
  groups: list[list[int]] = [[] for _ in GROUPS]
  for geom_id in range(model.ngeom):
    body_id = int(model.geom_bodyid[geom_id])
    if body_id == 0:
      continue
    name = model.body(body_id).name
    exact = [i for i, group in enumerate(GROUPS) if name in group.bodies]
    matches = exact or [
      i
      for i, group in enumerate(GROUPS)
      if any(name.startswith(prefix) for prefix in group.prefixes)
    ]
    if len(matches) != 1:
      raise ValueError(f"机器人部件归属不唯一：{name}")
    if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
      raise ValueError(f"暂不支持非网格机器人几何：{name}")
    groups[matches[0]].append(geom_id)
  if any(not group for group in groups):
    raise ValueError("五盒中存在没有网格的分组")
  return groups


# 读取当前姿态下的真实网格顶点。
def mesh_world_vertices(
  model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> np.ndarray:
  mesh_id = int(model.geom_dataid[geom_id])
  start = int(model.mesh_vertadr[mesh_id])
  count = int(model.mesh_vertnum[mesh_id])
  local = model.mesh_vert[start : start + count].astype(np.float64)
  return local @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]


# 在分组锚点朝向中逐姿态重算包围盒并检查全部顶点覆盖。
def compute_boxes(
  model: mujoco.MjModel, data: mujoco.MjData, groups: list[list[int]], margin: float
) -> list[dict]:
  reference = mujoco.MjData(model)
  mujoco.mj_forward(model, reference)
  boxes = []
  for spec, geom_ids in zip(GROUPS, groups, strict=True):
    anchor = model.body(spec.anchor).id
    rotation = data.xmat[anchor].reshape(3, 3) @ reference.xmat[anchor].reshape(3, 3).T
    origin = data.xpos[anchor]
    world = np.concatenate([mesh_world_vertices(model, data, gid) for gid in geom_ids])
    local = (world - origin) @ rotation
    lower, upper = local.min(axis=0) - margin, local.max(axis=0) + margin
    outside = np.maximum(lower - local, local - upper).max()
    if outside > 1e-9:
      raise ValueError(f"网格顶点超出包络：{spec.key}")
    center = origin + ((lower + upper) / 2) @ rotation.T
    half = (upper - lower) / 2
    corners = center + (BOX_SIGNS * half) @ rotation.T
    boxes.append(
      {
        "spec": spec,
        "center": center,
        "rotation": rotation,
        "half": half,
        "corners": corners,
        "geom_ids": geom_ids,
        "vertex_count": len(world),
        "max_outside_m": float(max(outside, 0)),
      }
    )
  return boxes


# 保留 XML 根姿态，采用项目初始关节配置或完整 qpos 快照。
def load_pose(model: mujoco.MjModel, qpos_file: Path | None) -> mujoco.MjData:
  data = mujoco.MjData(model)
  if qpos_file is not None:
    qpos = np.load(qpos_file, allow_pickle=False)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
      raise ValueError(f"姿态快照应是长度 {model.nq} 的有限实数数组")
    data.qpos[:] = qpos
    mujoco.mj_normalizeQuat(model, data.qpos)
  else:
    for name, value in (INIT_STATE.joint_pos or {}).items():
      if name != ".*":
        data.qpos[model.joint(name).qposadr[0]] = value
  mujoco.mj_forward(model, data)
  return data


# 仅调整离屏渲染副本的材质与光照。
def configure_rendering(model: mujoco.MjModel, width: int, height: int) -> None:
  model.vis.global_.offwidth = width
  model.vis.global_.offheight = height
  model.vis.global_.ipd = 0
  model.vis.quality.offsamples = 4
  model.vis.headlight.active = 1
  model.vis.headlight.ambient[:] = 0.50
  model.vis.headlight.diffuse[:] = 0.65
  model.vis.headlight.specular[:] = 0.12
  for material_id in range(model.nmat):
    if np.mean(model.mat_rgba[material_id, :3]) > 0.5:
      model.mat_rgba[material_id, :3] = (0.73, 0.75, 0.77)
    model.mat_reflectance[material_id] = 0
    model.mat_shininess[material_id] = 0.3
  model.light_active[:] = 0


# 用同一正交相机对齐原生渲染与几何投影。
def prepare_camera(
  renderer: mujoco.Renderer,
  data: mujoco.MjData,
  boxes: list[dict],
  azimuth: float,
  elevation: float,
  width: int,
  height: int,
) -> tuple[np.ndarray, np.ndarray, float]:
  vertices = np.concatenate([box["corners"] for box in boxes])
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.azimuth, camera.elevation = azimuth, elevation
  camera.lookat[:] = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
  camera.distance = 1.0
  option = mujoco.MjvOption()
  option.geomgroup[:] = 0
  option.geomgroup[1] = 1
  option.sitegroup[:] = 0
  renderer.update_scene(data, camera=camera, scene_option=option)
  gl_camera = renderer.scene.camera[0]
  forward = np.array(gl_camera.forward, dtype=float)
  up = np.array(gl_camera.up, dtype=float)
  right = np.cross(forward, up)
  basis = np.column_stack((right, up, forward))
  projected = (vertices - camera.lookat) @ basis
  center_xy = (projected[:, :2].min(axis=0) + projected[:, :2].max(axis=0)) / 2
  camera.lookat[:] += basis[:, :2] @ center_xy
  span = np.ptp(projected[:, :2], axis=0)
  view_height = max(span[1], span[0] * height / width) * 1.16
  renderer.update_scene(data, camera=camera, scene_option=option)
  for gl_camera in renderer.scene.camera:
    gl_camera.orthographic = 1
    gl_camera.frustum_top = view_height / 2
    gl_camera.frustum_bottom = -view_height / 2
    gl_camera.frustum_center = 0
    gl_camera.frustum_width = 0
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
  return np.array(camera.lookat), basis, view_height


# 将已校准的正交相机坐标转换成像素。
def project_points(
  points: np.ndarray,
  center: np.ndarray,
  basis: np.ndarray,
  view_height: float,
  width: int,
  height: int,
) -> np.ndarray:
  projected = (points - center) @ basis
  scale = height / view_height
  return np.column_stack(
    (
      width / 2 + projected[:, 0] * scale,
      height / 2 - projected[:, 1] * scale,
      projected[:, 2],
    )
  )


# 在透明背景上绘制具有独立面积的五个包围盒。
def draw_boxes(
  base: Image.Image,
  boxes: list[dict],
  center: np.ndarray,
  basis: np.ndarray,
  view_height: float,
  geometry_only: bool,
) -> Image.Image:
  width, height = base.size
  projected = [
    project_points(box["corners"], center, basis, view_height, width, height)
    for box in boxes
  ]
  faces = []
  for box_id, points in enumerate(projected):
    for face in BOX_FACES:
      polygon = points[list(face)]
      midpoint = polygon[:, :2].mean(axis=0)
      order = np.argsort(
        np.arctan2(polygon[:, 1] - midpoint[1], polygon[:, 0] - midpoint[0])
      )
      faces.append((polygon[:, 2].mean(), box_id, polygon[order, :2]))
  result = base.copy()
  for _, box_id, polygon in sorted(faces, key=lambda face: face[0], reverse=True):
    layer = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(layer)
    color = boxes[box_id]["spec"].color
    draw.polygon(
      [tuple(point) for point in polygon], fill=(*color, 72 if geometry_only else 24)
    )
    result = Image.alpha_composite(result, layer)
  layer = Image.new("RGBA", base.size)
  draw = ImageDraw.Draw(layer)
  for box, points in zip(boxes, projected, strict=True):
    color = tuple(round(channel * 0.80) for channel in box["spec"].color)
    for start, end in BOX_EDGES:
      draw.line(
        [tuple(points[start, :2]), tuple(points[end, :2])],
        fill=(*color, 235),
        width=max(2, round(width / 650)),
      )
  return Image.alpha_composite(result, layer)


# 用分割通道获得机器人透明底，保留真实腿部网格。
def render_view(
  renderer: mujoco.Renderer,
  data: mujoco.MjData,
  boxes: list[dict],
  view: tuple,
  width: int,
  height: int,
) -> tuple[dict[str, Image.Image], dict]:
  _, _, azimuth, elevation = view
  center, basis, view_height = prepare_camera(
    renderer, data, boxes, azimuth, elevation, width, height
  )
  rgb = renderer.render().copy()
  renderer.enable_segmentation_rendering()
  segmentation = renderer.render().copy()
  renderer.disable_segmentation_rendering()
  mask = (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (
    segmentation[..., 0] >= 0
  )
  robot = Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255)))
  overlay = draw_boxes(robot, boxes, center, basis, view_height, False)
  geometry = draw_boxes(
    Image.new("RGBA", (width, height)), boxes, center, basis, view_height, True
  )
  coverage = np.asarray(geometry.getchannel("A").filter(ImageFilter.MaxFilter(5))) > 0
  outside_pixels = int(np.count_nonzero(mask & ~coverage))
  if outside_pixels:
    raise ValueError(f"{view[0]} 视图有 {outside_pixels} 个机器人像素超出投影包络")
  checks = {
    "silhouette_pixels": int(mask.sum()),
    "outside_pixels": outside_pixels,
    "raster_tolerance_pixels": 2,
  }
  return {"robot": robot, "overlay": overlay, "geometry": geometry}, checks


# 将透明素材放入预览区域。
def place_asset(
  canvas: Image.Image, asset: Image.Image, rect: tuple[int, int, int, int]
) -> None:
  bounds = asset.getbbox()
  if bounds is None:
    raise ValueError("渲染素材为空")
  cropped = asset.crop(bounds)
  left, top, width, height = rect
  cropped.thumbnail((width, height), Image.Resampling.LANCZOS)
  canvas.alpha_composite(
    cropped, (left + (width - cropped.width) // 2, top + (height - cropped.height) // 2)
  )


# 生成单素材预览页，英文标签沿用论文框图约定。
def create_preview(assets: dict, output: Path) -> None:
  canvas = Image.new("RGBA", (2400, 1560), "white")
  draw = ImageDraw.Draw(canvas)
  font_dir = Path("C:/Windows/Fonts")
  title = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 46)
  label = ImageFont.truetype(str(font_dir / "arial.ttf"), 31)
  small = ImageFont.truetype(str(font_dir / "arial.ttf"), 28)
  draw.text((75, 45), "Five-box whole-robot approximation", font=title, fill="#243348")
  draw.text(
    (75, 110),
    "Geometry derived from the current MuJoCo model",
    font=small,
    fill="#66717F",
  )
  draw.text((75, 190), "Robot + bounding boxes", font=label, fill="#243348")
  draw.text((1370, 190), "Five-box geometry", font=label, fill="#243348")
  place_asset(canvas, assets["oblique"]["overlay"], (60, 250, 1250, 650))
  place_asset(canvas, assets["oblique"]["geometry"], (1360, 250, 970, 650))
  for index, (key, text, _, _) in enumerate(VIEWS[1:]):
    left = 70 + index * 780
    draw.text((left, 970), text, font=label, fill="#243348")
    place_asset(canvas, assets[key]["overlay"], (left, 1035, 700, 280))
  for index, spec in enumerate(GROUPS):
    left = 80 + (index % 3) * 790
    top = 1390 + (index // 3) * 60
    draw.rectangle((left, top + 4, left + 26, top + 30), fill=spec.color)
    draw.text((left + 42, top), spec.label, font=small, fill="#344050")
  canvas.convert("RGB").save(output, dpi=(300, 300))


# 保存可复算的分组、尺寸、角点和模型指纹。
def save_metadata(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  boxes: list[dict],
  output: Path,
  margin: float,
  qpos_file: Path | None,
  projection_checks: dict,
) -> None:
  groups = []
  for box in boxes:
    groups.append(
      {
        "key": box["spec"].key,
        "label": box["spec"].label,
        "anchor": box["spec"].anchor,
        "bodies": [
          model.body(int(model.geom_bodyid[gid])).name for gid in box["geom_ids"]
        ],
        "dimensions_mm": (box["half"] * 2000).tolist(),
        "center_m": box["center"].tolist(),
        "rotation": box["rotation"].tolist(),
        "corners_m": box["corners"].tolist(),
        "vertex_count": box["vertex_count"],
        "max_outside_m": box["max_outside_m"],
      }
    )
  metadata = {
    "model_xml": str(SQuRO_XML),
    "xml_sha256": hashlib.sha256(SQuRO_XML.read_bytes()).hexdigest(),
    "mesh_vertices_sha256": hashlib.sha256(model.mesh_vert.tobytes()).hexdigest(),
    "mujoco_version": mujoco.__version__,
    "margin_mm": margin * 1000,
    "pose_source": str(qpos_file)
    if qpos_file
    else "project_initial_joint_configuration",
    "coordinate_system": "world +x forward, +y left, +z up at the neutral pose",
    "frame_definition": "anchor orientation relative to model qpos0 orientation",
    "projection_checks": projection_checks,
    "groups": groups,
  }
  output.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  np.save(output.with_name("pose_qpos.npy"), data.qpos)


# 导出五盒独立素材，不改动任务环境或模型文件。
def main() -> None:
  parser = argparse.ArgumentParser(description="从 MuJoCo 网格导出整机五盒包络素材")
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DIAGNOSTICS_ROOT / "SQuRo_Envelope" / "five_box_geometry",
  )
  parser.add_argument(
    "--qpos-file", type=Path, help="与当前模型关节顺序一致的完整 qpos.npy"
  )
  parser.add_argument(
    "--margin-mm", type=float, default=0, help="包络几何外扩量，默认不外扩"
  )
  parser.add_argument("--width", type=int, default=2000)
  parser.add_argument("--height", type=int, default=1300)
  args = parser.parse_args()
  if args.margin_mm < 0 or args.width < 400 or args.height < 300:
    parser.error("外扩量必须非负，图像至少为 400×300")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  model = get_spec().compile()
  data = load_pose(model, args.qpos_file)
  groups = resolve_group_geoms(model)
  boxes = compute_boxes(model, data, groups, args.margin_mm / 1000)
  configure_rendering(model, args.width, args.height)
  assets = {}
  projection_checks = {}
  with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
    for view in VIEWS:
      assets[view[0]], projection_checks[view[0]] = render_view(
        renderer, data, boxes, view, args.width, args.height
      )
      for kind, asset in assets[view[0]].items():
        asset.save(args.output_dir / f"five_box_{kind}_{view[0]}.png", dpi=(300, 300))
  create_preview(assets, args.output_dir / "five_box_preview.png")
  save_metadata(
    model,
    data,
    boxes,
    args.output_dir / "five_box_geometry.json",
    args.margin_mm / 1000,
    args.qpos_file,
    projection_checks,
  )
  count = sum(box["vertex_count"] for box in boxes)
  print(
    f"已覆盖 {sum(map(len, groups))} 个机器人网格、{count} 个顶点，五组均无顶点越界。"
  )
  print("四个视图的原生机器人轮廓均包含在对应的五盒投影内。")
  print(f"素材已保存：{args.output_dir.resolve()}")


if __name__ == "__main__":
  main()
