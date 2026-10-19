import argparse
import ast
import hashlib
import json
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from mjlab import MJLAB_SRC_PATH
from mjlab.asset_zoo.robots.SQuRo.SQuRo_constants import SQuRO_XML
from mjlab.scripts._diagnostics import DIAGNOSTICS_ROOT
from mjlab.scripts.Figure.render_squro_five_box import (
  BOX_EDGES,
  BOX_FACES,
  BOX_SIGNS,
  compute_boxes,
  configure_rendering,
  get_spec,
  load_pose,
  mesh_world_vertices,
  project_points,
  resolve_group_geoms,
)

OBSTACLE_BLUE = (33, 137, 225)
OBSTACLE_FILL = (145, 195, 232)
OBSTACLE_FILL_ALPHA = 40
BOX_TEAL = (32, 129, 126)
OVERLAP_MAUVE = (166, 120, 166)
OVERLAP_FILL_ALPHA = 70
BOARD_GRAY = (0.50, 0.55, 0.60, 1.0)
DEFAULT_OUTPUT = DIAGNOSTICS_ROOT / "SQuRo_Envelope" / "contact_pair_20261019"


# 直接读取唯一几何常量，避免触发无关任务和奖励注册。
def geometry_constant(name):
  path = MJLAB_SRC_PATH / "tasks" / "SQuRo_Hole" / "mdp" / "config.py"
  nodes = {
    target.id: node.value
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body
    if isinstance(node, ast.Assign)
    for target in node.targets
    if isinstance(target, ast.Name)
  }

  # 解析几何常量中引用的标量和元组。
  def resolve(node):
    if isinstance(node, ast.Name):
      return resolve(nodes[node.id])
    if isinstance(node, (ast.Tuple, ast.List)):
      return tuple(resolve(item) for item in node.elts)
    return ast.literal_eval(node)

  return resolve(nodes[name])


# 构建同源限高板，独立素材可覆盖完整尺寸而不改变任务配置。
def build_model(
  contact: bool,
  height: float,
  width: int,
  image_height: int,
  obstacle_size: tuple[float, float, float] | None = None,
):
  spec = get_spec()
  mask = int(contact)
  cfg = SimpleNamespace(
    position=(*geometry_constant("OBSTACLE_POSITION")[:2], height),
    size=tuple(value / 2 for value in obstacle_size)
    if obstacle_size is not None
    else geometry_constant("OBSTACLE_SIZE"),
  )
  body = spec.worldbody.add_body(name="figure_obstacle", pos=cfg.position, mocap=True)
  body.add_geom(
    name="figure_obstacle_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0, 0, cfg.size[2]),
    size=cfg.size,
    contype=mask,
    conaffinity=mask,
    rgba=BOARD_GRAY if contact else (0.0, 0.0, 0.0, 0.0),
    mass=1.0,
  )
  spec.geom("floor").pos = (cfg.position[0] - 0.035, 0, 0)
  spec.geom("floor").size = (0.18, 0.10, 0.25)
  model = spec.compile()
  configure_rendering(model, width, image_height)
  floor = model.geom("floor").id
  model.geom_rgba[floor] = (0.955, 0.965, 0.97, 1.0)
  ground_material = int(model.geom_matid[floor])
  model.mat_texid[ground_material] = -1
  model.mat_rgba[ground_material] = (0.94, 0.955, 0.965, 1.0)
  model.mat_emission[ground_material] = 0.45
  return model, model.geom("figure_obstacle_geom").id, cfg


# 用原机器人模型解析五组网格索引并检查追加实体未改变它们。
def robot_groups(model):
  original = get_spec().compile()
  groups = resolve_group_geoms(original)
  for ids in groups:
    for index in ids:
      assert (
        model.body(int(model.geom_bodyid[index])).name
        == original.body(int(original.geom_bodyid[index])).name
      )
      assert model.geom_type[index] == original.geom_type[index]
  return groups


# 读取板体八个世界角点。
def obstacle_corners(model, data, geom_id):
  rotation = data.geom_xmat[geom_id].reshape(3, 3)
  return data.geom_xpos[geom_id] + (BOX_SIGNS * model.geom_size[geom_id]) @ rotation.T


# 固定驱动关节，将被动闭链拟合到一致初始状态。
def close_linkages(model, data):
  driven = set(int(i) for i in model.actuator_trnid[:, 0])
  passive = [
    i
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE and i not in driven
  ]
  addresses = model.jnt_qposadr[passive]

  # 只优化闭链等式位置残差。
  def residual(values):
    data.qpos[addresses] = values
    mujoco.mj_forward(model, data)
    return data.efc_pos[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY].copy()

  result = least_squares(
    residual,
    data.qpos[addresses].copy(),
    bounds=model.jnt_range[passive].T,
    ftol=1e-11,
    xtol=1e-11,
    gtol=1e-11,
    max_nfev=150,
  )
  if np.max(np.abs(residual(result.x))) > 2e-6:
    raise ValueError("初始姿态闭链误差超限")


# 生成接触关闭、保留地面作用的中性站姿机制示意。
def training_pose(model, groups, board_x):
  data = load_pose(model, None)
  close_linkages(model, data)
  free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  assert free.size == 1
  address = int(model.jnt_qposadr[free[0]])
  data.qpos[address] += board_x - data.xpos[model.body("F_body_Link").id, 0]
  mujoco.mj_forward(model, data)
  boxes = compute_boxes(model, data, groups, 0)
  data.qpos[address + 2] -= min(box["corners"][:, 2].min() for box in boxes) - 0.0002
  data.ctrl[:] = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
  for _ in range(90):
    mujoco.mj_step(model, data)
  mujoco.mj_forward(model, data)
  return data


# 在本次板体几何上重检候选帧，选择靠近板中心且无明显穿透的姿态。
def evaluation_pose(model, board_id, candidates):
  best = None
  feet = [model.geom(f"{name}_foot_collision").id for name in ("FL", "FR", "HL", "HR")]
  for index, values in enumerate(candidates):
    data = mujoco.MjData(model)
    if values.shape != data.qpos.shape:
      raise ValueError("候选姿态与原生模型维度不一致")
    data.qpos[:] = values
    mujoco.mj_forward(model, data)
    distances = [
      float(c.dist) for c in data.contact if board_id in (int(c.geom1), int(c.geom2))
    ]
    minimum = min(distances, default=0.0)
    if minimum < -0.0008:
      continue
    front = data.xpos[model.body("F_body_Link").id]
    foot_heights = [
      float(mesh_world_vertices(model, data, geom)[:, 2].min()) for geom in feet
    ]
    score = abs(front[0] - data.geom_xpos[board_id, 0]) + abs(front[1]) * 0.2
    score += sum(max(value, 0) for value in foot_heights) * 0.25
    if best is None or score < best[0]:
      best = score, index, data, minimum, foot_heights
  if best is None:
    raise ValueError("没有找到板下接触有效的评价姿态")
  return best[2], {
    "candidate_index": best[1],
    "minimum_board_contact_distance_m": best[3],
    "foot_mesh_bottom_z_m": best[4],
  }


# 计算有向检测盒与实际板体的凸多面体交集。
def box_intersection(box, board):
  rotation, center, half = box["rotation"], box["center"], box["half"]
  lower, upper = board.min(axis=0), board.max(axis=0)
  matrix = np.vstack((rotation.T, -rotation.T, np.eye(3), -np.eye(3)))
  bound = np.r_[half + rotation.T @ center, half - rotation.T @ center, upper, -lower]
  points = []
  for indices in combinations(range(len(bound)), 3):
    selected = matrix[list(indices)]
    if abs(np.linalg.det(selected)) < 1e-10:
      continue
    point = np.linalg.solve(selected, bound[list(indices)])
    if np.max(matrix @ point - bound) <= 1e-9:
      points.append(point)
  if len(points) < 4:
    return None
  points = np.unique(np.round(points, 11), axis=0)
  if len(points) < 4:
    return None
  if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-10) < 3:
    return None
  hull = ConvexHull(points)
  if hull.volume < 1e-12:
    return None
  faces = []
  for normal, offset in zip(matrix, bound, strict=True):
    selected = points[np.abs(points @ normal - offset) < 2e-8]
    if len(selected) < 3:
      continue
    center_face = selected.mean(axis=0)
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    horizontal = np.cross(normal, axis)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    angles = np.arctan2(
      (selected - center_face) @ vertical, (selected - center_face) @ horizontal
    )
    faces.append(selected[np.argsort(angles)])
  assert np.max(matrix @ points.T - bound[:, None]) < 2e-8
  return {
    "key": box["spec"].key,
    "vertices": points,
    "faces": faces,
    "volume_m3": float(hull.volume),
    "violation_m": float(points[:, 2].max() - lower[2]),
  }


# 以两种姿态和同一障碍物一次确定相机，后续渲染保持完全相同。
def make_camera(renderer, data, points, width, height, azimuth, elevation):
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, 1.0
  camera.lookat[:] = (points.min(axis=0) + points.max(axis=0)) / 2
  option = mujoco.MjvOption()
  option.geomgroup[:] = 0
  option.geomgroup[:2] = 1
  option.sitegroup[:] = 0
  renderer.update_scene(data, camera=camera, scene_option=option)
  forward = np.asarray(renderer.scene.camera[0].forward, dtype=float)
  up = np.asarray(renderer.scene.camera[0].up, dtype=float)
  basis = np.column_stack((np.cross(forward, up), up, forward))
  projected = (points - camera.lookat) @ basis
  middle = (projected[:, :2].min(axis=0) + projected[:, :2].max(axis=0)) / 2
  camera.lookat[:] += basis[:, :2] @ middle
  span = np.ptp(projected[:, :2], axis=0)
  view_height = max(span[1], span[0] * height / width) * 1.13
  return camera, option, basis, float(view_height)


# 用分割通道输出机器人、有限地面及障碍物的透明底。
def render_base(renderer, data, camera, option, view_height):
  renderer.update_scene(data, camera=camera, scene_option=option)
  for gl in renderer.scene.camera:
    gl.orthographic = 1
    gl.frustum_top, gl.frustum_bottom = view_height / 2, -view_height / 2
    gl.frustum_center = gl.frustum_width = 0
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
  renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
  rgb = renderer.render().copy()
  renderer.enable_segmentation_rendering()
  segmentation = renderer.render().copy()
  renderer.disable_segmentation_rendering()
  mask = (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (
    segmentation[..., 0] >= 0
  )
  return Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255)))


# 将凸面顶点按投影环序排列，再按深度叠加透明面。
def face_layer(size, faces, camera, basis, view_height, color, alpha):
  output = Image.new("RGBA", size)
  projected = [
    project_points(face, camera.lookat, basis, view_height, *size) for face in faces
  ]
  for polygon in sorted(projected, key=lambda item: item[:, 2].mean(), reverse=True):
    layer = Image.new("RGBA", size)
    points = polygon[:, :2]
    delta = points - points.mean(axis=0)
    points = points[np.argsort(np.arctan2(delta[:, 1], delta[:, 0]))]
    ImageDraw.Draw(layer).polygon([tuple(p) for p in points], fill=(*color, alpha))
    output = Image.alpha_composite(output, layer)
  return output


# 绘制可调虚线长度的边界线。
def dashed_line(draw, first, last, color, width, dash):
  start, end = np.asarray(first), np.asarray(last)
  distance = np.linalg.norm(end - start)
  if distance < 1e-8:
    return
  direction = (end - start) / distance
  for current in np.arange(0, distance, dash * 1.65):
    draw.line(
      [
        tuple(start + current * direction),
        tuple(start + min(current + dash, distance) * direction),
      ],
      fill=color,
      width=width,
    )


# 从同一板体角点和五盒角点生成虚拟检测叠加层。
def virtual_layers(size, boxes, board, intersections, camera, basis, view_height):
  faces = [box["corners"][list(face)] for box in boxes for face in BOX_FACES]
  boxes_layer = face_layer(size, faces, camera, basis, view_height, BOX_TEAL, 13)
  draw = ImageDraw.Draw(boxes_layer)
  for box in boxes:
    projected = project_points(box["corners"], camera.lookat, basis, view_height, *size)
    for a, b in BOX_EDGES:
      draw.line(
        [tuple(projected[a, :2]), tuple(projected[b, :2])],
        fill=(*BOX_TEAL, 235),
        width=4,
      )
  obstacle_layer = face_layer(
    size,
    [board[list(face)] for face in BOX_FACES],
    camera,
    basis,
    view_height,
    OBSTACLE_FILL,
    OBSTACLE_FILL_ALPHA,
  )
  projected = project_points(board, camera.lookat, basis, view_height, *size)
  draw = ImageDraw.Draw(obstacle_layer)
  for a, b in BOX_EDGES:
    dashed_line(draw, projected[a, :2], projected[b, :2], (*OBSTACLE_BLUE, 255), 4, 13)
  overlap_layer = face_layer(
    size,
    [f for region in intersections for f in region["faces"]],
    camera,
    basis,
    view_height,
    OVERLAP_MAUVE,
    OVERLAP_FILL_ALPHA,
  )
  draw = ImageDraw.Draw(overlap_layer)
  for region in intersections:
    for face in region["faces"]:
      polygon = project_points(face, camera.lookat, basis, view_height, *size)
      draw.line(
        [*map(tuple, polygon[:, :2]), tuple(polygon[0, :2])],
        fill=(*OVERLAP_MAUVE, 255),
        width=3,
      )
  return boxes_layer, obstacle_layer, overlap_layer


# 绘制局部越界距离，保留无标注版本供 PPT 编辑。
def detail_image(image, intersections, board, camera, basis, view_height, output):
  main = max(intersections, key=lambda region: region["volume_m3"])
  points = project_points(
    main["vertices"], camera.lookat, basis, view_height, *image.size
  )
  middle = points[:, :2].mean(axis=0)
  side = max(340, int(np.ptp(points[:, :2], axis=0).max() * 1.7))
  left = max(0, min(image.width - side, int(middle[0] - side / 2)))
  top = max(0, min(image.height - side, int(middle[1] - side * 0.28)))
  crop = image.crop((left, top, left + side, top + side)).resize(
    (900, 900), Image.Resampling.LANCZOS
  )
  crop.save(output / "03_violation_detail_clean.png", dpi=(300, 300))
  annotated = Image.new("RGBA", crop.size, (255, 255, 255, 255))
  annotated.alpha_composite(crop)
  highest = main["vertices"][np.argmax(main["vertices"][:, 2])].copy()
  lower = highest.copy()
  lower[2] = board[:, 2].min()
  projected = project_points(
    np.array([highest, lower]), camera.lookat, basis, view_height, *image.size
  )[:, :2]
  projected = (projected - (left, top)) * 900 / side
  x = min(820, max(projected[:, 0]) + 65)
  draw = ImageDraw.Draw(annotated)
  for point in projected:
    draw.line([tuple(point), (x + 12, point[1])], fill=OVERLAP_MAUVE, width=4)
  a, b = sorted(projected[:, 1])
  draw.line([(x, a), (x, b)], fill=OVERLAP_MAUVE, width=5)
  draw.polygon([(x, a), (x - 8, a + 17), (x + 8, a + 17)], fill=OVERLAP_MAUVE)
  draw.polygon([(x, b), (x - 8, b - 17), (x + 8, b - 17)], fill=OVERLAP_MAUVE)
  font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 46)
  draw.text((x + 20, (a + b) / 2 - 27), "δ", font=font, fill=OVERLAP_MAUVE)
  annotated.save(output / "04_violation_detail_delta.png", dpi=(300, 300))
  return {
    "crop_pixels": [left, top, side, side],
    "highlighted_group": main["key"],
    "violation_mm": main["violation_m"] * 1000,
  }


# 生成不改变两张素材相对比例的并排预览。
def preview(first, second, output):
  canvas = Image.new("RGB", (2200, 900), "white")
  draw = ImageDraw.Draw(canvas)
  title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 39)
  subtitle = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
  for index, asset in enumerate((first, second)):
    x = 55 + index * 1100
    draw.text((x, 42), ("Training", "Evaluation")[index], font=title, fill="#253E50")
    draw.text(
      (x, 98),
      ("Obstacle contact OFF", "Obstacle contact ON")[index],
      font=subtitle,
      fill="#526773",
    )
    resized = asset.resize((1040, 624), Image.Resampling.LANCZOS)
    canvas.paste(resized, (x, 177), resized)
    draw.text(
      (x, 824),
      (
        "Five-box envelope with local overlap",
        "Same obstacle geometry with solid contact",
      )[index],
      font=subtitle,
      fill="#526773",
    )
  canvas.save(output / "05_pair_preview.png", dpi=(300, 300))


# 输出同源虚实碰撞素材、姿态与可重复几何检查。
def main():
  parser = argparse.ArgumentParser(
    description="从真实 MuJoCo 模型绘制五盒虚实碰撞示意素材"
  )
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument(
    "--evaluation-candidates",
    type=Path,
    default=DEFAULT_OUTPUT / "evaluation_candidates.npz",
  )
  parser.add_argument("--height", type=float, default=0.075)
  parser.add_argument(
    "--obstacle-size",
    type=float,
    nargs=3,
    metavar=("LENGTH", "WIDTH", "THICKNESS"),
    help="仅覆盖论文素材板体的完整长宽厚，单位米，任务配置保持不变",
  )
  parser.add_argument("--width", type=int, default=2000)
  parser.add_argument("--image-height", type=int, default=1200)
  parser.add_argument("--azimuth", type=float, default=105)
  parser.add_argument("--elevation", type=float, default=-2.5)
  args = parser.parse_args()
  if args.height <= 0 or args.width < 800 or args.image_height < 500:
    parser.error("板高必须为正，输出图像至少为 800×500")
  if args.obstacle_size is not None and not all(
    np.isfinite(value) and value > 0 for value in args.obstacle_size
  ):
    parser.error("板体完整长宽厚必须为正的有限数值")
  output = args.output_dir
  output.mkdir(parents=True, exist_ok=True)
  off_model, off_id, off_cfg = build_model(
    False, args.height, args.width, args.image_height, args.obstacle_size
  )
  on_model, on_id, on_cfg = build_model(
    True, args.height, args.width, args.image_height, args.obstacle_size
  )
  assert off_cfg.position == on_cfg.position and off_cfg.size == on_cfg.size
  groups = robot_groups(off_model)
  off_data = training_pose(off_model, groups, off_cfg.position[0])
  provenance_path = args.evaluation_candidates.with_suffix(".json")
  provenance = {}
  if provenance_path.is_file():
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if abs(float(provenance["obstacle_height_m"]) - args.height) > 1e-12:
      raise ValueError("评价取帧板高与本次配对板高不一致")
  candidates = np.load(args.evaluation_candidates, allow_pickle=False)["qpos"]
  on_data, evaluation = evaluation_pose(on_model, on_id, candidates)
  off_boxes = compute_boxes(off_model, off_data, groups, 0)
  on_boxes = compute_boxes(on_model, on_data, groups, 0)
  board = obstacle_corners(off_model, off_data, off_id)
  on_board = obstacle_corners(on_model, on_data, on_id)
  assert np.array_equal(board, on_board)
  assert off_model.geom_contype[off_id] == off_model.geom_conaffinity[off_id] == 0
  assert on_model.geom_contype[on_id] == on_model.geom_conaffinity[on_id] == 1
  assert on_model.geom_contype[on_model.geom("floor").id] == 1
  intersections = [
    region for box in off_boxes if (region := box_intersection(box, board)) is not None
  ]
  if not intersections:
    raise ValueError("训练姿态没有检测盒越界，无法生成机制示意")
  points = np.concatenate([box["corners"] for box in off_boxes + on_boxes] + [board])
  with mujoco.Renderer(
    off_model, height=args.image_height, width=args.width
  ) as renderer:
    camera, option, basis, view_height = make_camera(
      renderer,
      off_data,
      points,
      args.width,
      args.image_height,
      args.azimuth,
      args.elevation,
    )
    base = render_base(renderer, off_data, camera, option, view_height)
  boxes_layer, obstacle_layer, overlap_layer = virtual_layers(
    base.size, off_boxes, board, intersections, camera, basis, view_height
  )
  first = Image.alpha_composite(base, boxes_layer)
  first = Image.alpha_composite(first, obstacle_layer)
  first = Image.alpha_composite(first, overlap_layer)
  with mujoco.Renderer(
    on_model, height=args.image_height, width=args.width
  ) as renderer:
    second = render_base(renderer, on_data, camera, option, view_height)
  first.save(output / "01_training_virtual.png", dpi=(300, 300))
  second.save(output / "02_evaluation_contact.png", dpi=(300, 300))
  boxes_layer.save(output / "layer_robot_boxes.png", dpi=(300, 300))
  obstacle_layer.save(output / "layer_virtual_obstacle.png", dpi=(300, 300))
  overlap_layer.save(output / "layer_envelope_overlap.png", dpi=(300, 300))
  base.save(output / "layer_training_robot.png", dpi=(300, 300))
  detail = detail_image(first, intersections, board, camera, basis, view_height, output)
  preview(first, second, output)
  np.save(output / "training_qpos.npy", off_data.qpos)
  np.save(output / "evaluation_qpos.npy", on_data.qpos)
  checks = {
    "source_xml": str(SQuRO_XML),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "obstacle": {
      "geometry_source": "独立论文素材完整尺寸覆盖"
      if args.obstacle_size is not None
      else "Hole 任务几何常量",
      "position": list(off_cfg.position),
      "half_size": list(off_cfg.size),
      "full_size": [2 * value for value in off_cfg.size],
      "world_corners_m": board.tolist(),
      "same_world_corners": True,
      "same_projection": True,
    },
    "camera": {
      "azimuth": args.azimuth,
      "elevation": args.elevation,
      "lookat": camera.lookat.tolist(),
      "basis": basis.tolist(),
      "view_height": view_height,
      "size": [args.width, args.image_height],
    },
    "appearance": {
      "virtual_obstacle_fill_rgba": [*OBSTACLE_FILL, OBSTACLE_FILL_ALPHA],
      "virtual_obstacle_edge_rgb": list(OBSTACLE_BLUE),
      "overlap_fill_rgba": [*OVERLAP_MAUVE, OVERLAP_FILL_ALPHA],
    },
    "off_collision_mask": [0, 0],
    "on_collision_mask": [1, 1],
    "ground_contact_retained": True,
    "training_pose_source": "中性参考姿态在保留地面接触的原生 MuJoCo 中保持 0.36 秒，用作方法机制示意",
    "evaluation_pose_source": str(args.evaluation_candidates),
    "evaluation_pose_processing": "原回放候选姿态在本次板体上重新执行原生接触检查；不表示新几何下的策略通过结果",
    "evaluation_checkpoint": provenance.get("checkpoint"),
    "evaluation_policy": provenance.get("policy"),
    "evaluation": evaluation,
    "five_boxes_cover_all_mesh_vertices": True,
    "intersections": [
      {k: v for k, v in region.items() if k not in ("vertices", "faces")}
      for region in intersections
    ],
    "detail": detail,
    "claim_boundary": "仅展示几何检测与接触开关，不表示五盒已用于现有 Hole 奖励或报告通过率",
  }
  (output / "geometry_checks.json").write_text(
    json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(
    f"已生成同源虚实碰撞素材，越界检测盒 {len(intersections)} 个，最大局部越界 {max(r['violation_m'] for r in intersections) * 1000:.2f} 毫米"
  )
  print(f"结果保存在：{output.resolve()}")


if __name__ == "__main__":
  main()
