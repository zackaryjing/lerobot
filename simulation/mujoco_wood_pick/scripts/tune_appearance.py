#!/usr/bin/env python3
"""Interactive RGB, roughness and light tuner for the MuJoCo workcell."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


MATERIAL_GROUPS = {
    "桌面": ("table",),
    "平台白色部分": ("platform_white",),
    "平台左侧青色": ("platform_teal",),
    "纸盒外侧": ("cardboard",),
    "盒子内部底面": ("box_bottom",),
    "木条": ("wood",),
    "机械臂白色打印件": (
        "base_motor_holder_so101_v1_material",
        "base_so101_v2_material",
        "waveshare_mounting_plate_so101_v2_material",
        "motor_holder_so101_base_v1_material",
        "rotation_pitch_so101_v1_material",
        "upper_arm_so101_v1_material",
        "under_arm_so101_v1_material",
        "motor_holder_so101_wrist_v1_material",
        "wrist_roll_pitch_so101_v2_material",
    ),
    "黑色电机": ("sts3215_03a_v1_material", "sts3215_03a_no_horn_v1_material"),
    "黑色夹爪": ("wrist_roll_follower_so101_v1_material", "moving_jaw_so101_v1_material"),
}


def _restart_with_system_tk() -> None:
    """Use Ubuntu's Xft-enabled Tk instead of Conda's core-X11-only build.

    The Tk bundled in the local ``lerobot`` environment cannot discover CJK
    fonts and renders missing glyphs as literal ``\\uXXXX`` strings.  Loading
    the ABI-compatible Ubuntu Tcl/Tk libraries before Python starts restores
    Fontconfig/Xft support while keeping every Python/MuJoCo dependency in the
    Conda environment.
    """
    marker = "JOYAND_SYSTEM_TK_LOADED"
    if not sys.platform.startswith("linux") or os.environ.get(marker) == "1":
        return
    libraries = (
        Path("/lib/x86_64-linux-gnu/libtk8.6.so"),
        Path("/lib/x86_64-linux-gnu/libtcl8.6.so"),
    )
    if not all(path.is_file() for path in libraries):
        return
    preload = ":".join(str(path) for path in libraries)
    if os.environ.get("LD_PRELOAD"):
        preload += ":" + os.environ["LD_PRELOAD"]
    environment = os.environ.copy()
    environment["LD_PRELOAD"] = preload
    environment[marker] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def main() -> None:
    _restart_with_system_tk()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gl", choices=("egl", "glfw"), default="egl")
    parser.add_argument("--adapter", default="NVIDIA")
    parser.add_argument("--camera", choices=("front", "wrist", "overview"), default="front")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", args.gl)
    if args.adapter:
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", args.adapter)

    import tkinter as tk
    from tkinter import colorchooser, font as tkfont, messagebox, ttk

    import mujoco
    import numpy as np
    from PIL import Image, ImageTk

    from mujoco_wood_pick import WoodPickEnv
    from mujoco_wood_pick.env import DEFAULT_APPEARANCE_PATH, HOME_QPOS, LOCAL_APPEARANCE_PATH

    class AppearanceTuner:
        def __init__(self) -> None:
            self.env = WoodPickEnv(render_images=False)
            self.env.reset()
            self.env.advance_physics(HOME_QPOS, 15)
            self.root = tk.Tk()
            self.root.title("SO-101 MuJoCo 外观调色器")
            self.root.protocol("WM_DELETE_WINDOW", self.close)
            available_fonts = set(tkfont.families(self.root))
            cjk_family = next(
                (name for name in ("Droid Sans Fallback", "Noto Sans CJK SC", "Microsoft YaHei") if name in available_fonts),
                None,
            )
            if cjk_family is not None:
                for font_name in (
                    "TkDefaultFont",
                    "TkTextFont",
                    "TkMenuFont",
                    "TkHeadingFont",
                    "TkCaptionFont",
                    "TkSmallCaptionFont",
                    "TkIconFont",
                    "TkTooltipFont",
                ):
                    try:
                        tkfont.nametofont(font_name).configure(family=cjk_family)
                    except tk.TclError:
                        pass
            self._photo: ImageTk.PhotoImage | None = None
            self._render_job: str | None = None
            self._updating = False

            controls = ttk.Frame(self.root, padding=10)
            controls.grid(row=0, column=0, sticky="nsew")
            preview = ttk.Frame(self.root, padding=(0, 10, 10, 10))
            preview.grid(row=0, column=1, sticky="nsew")
            self.root.columnconfigure(1, weight=1)
            self.root.rowconfigure(0, weight=1)

            ttk.Label(controls, text="材质组").grid(row=0, column=0, columnspan=2, sticky="w")
            self.material_var = tk.StringVar(value=next(iter(MATERIAL_GROUPS)))
            material_box = ttk.Combobox(
                controls, textvariable=self.material_var, values=tuple(MATERIAL_GROUPS), state="readonly", width=24
            )
            material_box.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 8))
            material_box.bind("<<ComboboxSelected>>", self.load_selected_material)

            self.rgb_vars = [tk.IntVar() for _ in range(3)]
            for row, (label, variable) in enumerate(zip(("R", "G", "B"), self.rgb_vars), start=2):
                ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w")
                tk.Scale(
                    controls, from_=0, to=255, orient="horizontal", variable=variable, length=250,
                    command=self.material_changed,
                ).grid(row=row, column=1, sticky="ew")

            self.roughness_var = tk.IntVar()
            ttk.Label(controls, text="粗糙度").grid(row=5, column=0, sticky="w")
            tk.Scale(
                controls, from_=0, to=100, orient="horizontal", variable=self.roughness_var, length=250,
                command=self.material_changed,
            ).grid(row=5, column=1, sticky="ew")

            self.hex_var = tk.StringVar()
            ttk.Entry(controls, textvariable=self.hex_var, width=10).grid(row=6, column=0, sticky="ew", pady=5)
            ttk.Button(controls, text="输入 Hex", command=self.apply_hex).grid(row=6, column=1, sticky="w", pady=5)
            ttk.Button(controls, text="系统选色器", command=self.choose_color).grid(
                row=7, column=0, columnspan=2, sticky="ew", pady=(0, 10)
            )

            ttk.Separator(controls).grid(row=8, column=0, columnspan=2, sticky="ew", pady=5)
            ttk.Label(controls, text="灯光强度").grid(row=9, column=0, columnspan=2, sticky="w")
            self.light_vars = {
                "ambient": tk.IntVar(value=24),
                "head": tk.IntVar(value=27),
                "key": tk.IntVar(value=78),
                "fill": tk.IntVar(value=34),
            }
            for row, (key, label) in enumerate(
                (("ambient", "环境光"), ("head", "相机补光"), ("key", "右上主灯"), ("fill", "左侧补光")), start=10
            ):
                ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w")
                tk.Scale(
                    controls, from_=0, to=120, orient="horizontal", variable=self.light_vars[key], length=250,
                    command=self.lighting_changed,
                ).grid(row=row, column=1, sticky="ew")

            ttk.Label(controls, text="预览相机").grid(row=14, column=0, sticky="w", pady=(10, 0))
            self.camera_var = tk.StringVar(value=args.camera)
            camera_box = ttk.Combobox(
                controls, textvariable=self.camera_var, values=("front", "wrist", "overview"), state="readonly"
            )
            camera_box.grid(row=14, column=1, sticky="ew", pady=(10, 0))
            camera_box.bind("<<ComboboxSelected>>", lambda _event: self.schedule_render())

            buttons = ttk.Frame(controls)
            buttons.grid(row=15, column=0, columnspan=2, sticky="ew", pady=(14, 0))
            ttk.Button(buttons, text="保存本地配置", command=self.save).pack(side="left", expand=True, fill="x")
            ttk.Button(buttons, text="恢复默认", command=self.restore_default).pack(side="left", expand=True, fill="x")

            self.status_var = tk.StringVar(value=f"当前配置：{self.env.appearance_path.name}")
            ttk.Label(controls, textvariable=self.status_var, wraplength=300).grid(
                row=16, column=0, columnspan=2, sticky="w", pady=(8, 0)
            )
            self.image_label = ttk.Label(preview)
            self.image_label.pack(fill="both", expand=True)

            self.load_selected_material()
            self._load_light_controls()
            self.schedule_render()

        def _material_ids(self) -> list[int]:
            return [
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
                for name in MATERIAL_GROUPS[self.material_var.get()]
            ]

        def load_selected_material(self, _event: object | None = None) -> None:
            self._updating = True
            material_id = self._material_ids()[0]
            rgb = np.clip(self.env.model.mat_rgba[material_id, :3], 0.0, 1.0)
            for variable, value in zip(self.rgb_vars, rgb):
                variable.set(round(float(value) * 255))
            self.roughness_var.set(round(float(self.env.model.mat_roughness[material_id]) * 100))
            self._update_hex()
            self._updating = False

        def material_changed(self, _value: str = "") -> None:
            if self._updating:
                return
            rgb = np.array([variable.get() / 255.0 for variable in self.rgb_vars], dtype=np.float32)
            roughness = self.roughness_var.get() / 100.0
            for material_id in self._material_ids():
                self.env.model.mat_rgba[material_id, :3] = rgb
                self.env.model.mat_roughness[material_id] = roughness
            self._update_hex()
            self.schedule_render()

        def _update_hex(self) -> None:
            self.hex_var.set("#" + "".join(f"{variable.get():02X}" for variable in self.rgb_vars))

        def apply_hex(self) -> None:
            value = self.hex_var.get().strip().lstrip("#")
            if len(value) != 6:
                messagebox.showerror("Hex 格式错误", "请输入类似 #C0BDB3 的六位颜色。")
                return
            try:
                channels = [int(value[index : index + 2], 16) for index in (0, 2, 4)]
            except ValueError:
                messagebox.showerror("Hex 格式错误", "颜色中包含非十六进制字符。")
                return
            for variable, channel in zip(self.rgb_vars, channels):
                variable.set(channel)
            self.material_changed()

        def choose_color(self) -> None:
            selected = colorchooser.askcolor(color=self.hex_var.get(), title=self.material_var.get())[1]
            if selected:
                self.hex_var.set(selected)
                self.apply_hex()

        def _load_light_controls(self) -> None:
            model = self.env.model
            self.light_vars["ambient"].set(round(float(np.mean(model.vis.headlight.ambient)) * 100))
            self.light_vars["head"].set(round(float(np.mean(model.vis.headlight.diffuse)) * 100))
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "key")
            fill_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "fill")
            self.light_vars["key"].set(round(float(np.max(model.light_diffuse[key_id])) * 100))
            self.light_vars["fill"].set(round(float(np.mean(model.light_diffuse[fill_id])) * 100))

        def lighting_changed(self, _value: str = "") -> None:
            model = self.env.model
            ambient = self.light_vars["ambient"].get() / 100.0
            head = self.light_vars["head"].get() / 100.0
            key = self.light_vars["key"].get() / 100.0
            fill = self.light_vars["fill"].get() / 100.0
            model.vis.headlight.ambient[:] = (ambient, ambient, ambient * 0.96)
            model.vis.headlight.diffuse[:] = (head, head, head * 0.96)
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "key")
            fill_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "fill")
            model.light_diffuse[key_id] = np.array((1.0, 0.93, 0.82)) * key
            model.light_diffuse[fill_id] = np.array((0.78, 0.88, 1.0)) * fill
            self.schedule_render()

        def schedule_render(self) -> None:
            if self._render_job is not None:
                self.root.after_cancel(self._render_job)
            self._render_job = self.root.after(45, self.render)

        def render(self) -> None:
            self._render_job = None
            image = self.env.render_camera(self.camera_var.get())
            self._photo = ImageTk.PhotoImage(Image.fromarray(image))
            self.image_label.configure(image=self._photo)

        def _serialize(self) -> dict[str, object]:
            materials: dict[str, object] = {}
            for name_group in MATERIAL_GROUPS.values():
                for name in name_group:
                    material_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
                    materials[name] = {
                        "rgb": [round(float(v), 6) for v in self.env.model.mat_rgba[material_id, :3]],
                        "roughness": round(float(self.env.model.mat_roughness[material_id]), 4),
                    }
            key_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_LIGHT, "key")
            fill_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_LIGHT, "fill")
            lighting = {
                "headlight_ambient": self.env.model.vis.headlight.ambient.tolist(),
                "headlight_diffuse": self.env.model.vis.headlight.diffuse.tolist(),
                "key_diffuse": self.env.model.light_diffuse[key_id].tolist(),
                "fill_diffuse": self.env.model.light_diffuse[fill_id].tolist(),
            }
            return {"materials": materials, "lighting": lighting}

        def save(self) -> None:
            LOCAL_APPEARANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOCAL_APPEARANCE_PATH.write_text(
                json.dumps(self._serialize(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            self.status_var.set(f"已保存：{LOCAL_APPEARANCE_PATH}")

        def restore_default(self) -> None:
            self.env.apply_appearance(DEFAULT_APPEARANCE_PATH)
            self.load_selected_material()
            self._load_light_controls()
            self.schedule_render()
            self.status_var.set("已恢复项目默认值；点击保存才会覆盖本地配置。")

        def close(self) -> None:
            self.env.close()
            self.root.destroy()

        def run(self) -> None:
            self.root.mainloop()

    AppearanceTuner().run()


if __name__ == "__main__":
    main()
