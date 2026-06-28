"""EvasionRL Streamlit demo.

Side-by-side real-time demo of a clean moving RL robot and an attacked robot.
The attacked robot receives perturbed observations; the clean robot receives true observations.

This version shows only the animations during the run and a concise final summary after
simulation. Detailed timestep logs are intentionally hidden from the user-facing demo.
"""

from __future__ import annotations

import io
import base64
import streamlit.components.v1 as components
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from stable_baselines3 import DQN

from attacks import make_adversarial_observation, predict_action_from_obs

MODEL_PATH = Path("models/evasionrl_dqn_mountaincar.zip")
SUMMARY_PATH = MODEL_PATH.with_suffix(".training_summary.json")
ENV_ID = "MountainCar-v0"
ACTION_NAMES = {0: "push left", 1: "no push", 2: "push right"}
GOAL_POSITION = 0.5

st.set_page_config(page_title="EvasionRL", page_icon="🤖", layout="wide")
st.title("🤖 EvasionRL")
st.subheader("Real-Time Observation Evasion Attack")

st.markdown("""
This demo runs **two copies of the same trained RL robot** from the same initial state.

- **Clean robot:** receives the true environment observation.
- **Attacked robot:** the environment produces a true observation, but the attacker modifies it before the DQN sees it.
- The attacker does **not** change the model, reward, or physics.
- The simulation keeps running until the selected step limit, so you can keep watching the attacked robot even after the clean robot reaches the flag. The app also overrides MountainCar's default 200-step truncation, so the sidebar max-step value is the actual demo limit. A compact explanation appears at the end. The app shows live panels during the run and also creates an optional GIF playback, which is more reliable on Streamlit Cloud.
""")

with st.expander("What exactly does the agent see?", expanded=True):
    st.markdown("""
MountainCar gives the agent a 2-value observation:

```text
observation = [position, velocity]
```

For the clean robot, the DQN receives the true observation directly:

```text
true observation from environment → DQN → action
```

For the attacked robot, the environment still has the true physical state, but the DQN does **not** see it directly. The attacker intercepts the observation:

```text
true observation from environment → adversarial perturbation → modified observation → DQN → action
```

Example:

```text
True observation:     [-0.445,  0.000]
Perturbation:         [-0.050, +0.050]
What the DQN sees:    [-0.495,  0.050]
```

The robot then executes the action selected from the **modified observation**, but the environment physics still update the real car state.
""")

with st.sidebar:
    st.header("Attack controls")
    run_button = st.button("Run side-by-side demo", type="primary")
    attack = st.selectbox("Attack type", ["none", "random", "fgsm", "pgd"], index=3)
    epsilon = st.slider(
        "Observation perturbation ε", 0.0, 0.25, 0.050, 0.001, format="%.4f"
    )

    # PGD is an iterative attack, so the step-count control is only relevant
    # when PGD is selected. FGSM is one-step, random noise has no gradient
    # steps, and the no-attack baseline should keep the sidebar simple.
    if attack == "pgd":
        pgd_steps = st.slider("PGD steps", 1, 50, 20)
    else:
        pgd_steps = 1

    if attack in {"fgsm", "pgd"}:
        objective = st.selectbox(
            "Gradient objective",
            ["flip_margin", "reduce_best_q"],
            index=0,
            help="flip_margin encourages a different action; reduce_best_q reduces the current best action value.",
        )
    else:
        objective = "flip_margin"

    max_steps = st.slider("Max demo steps", 50, 1000, 300, 10)
    st.caption(
        "This overrides MountainCar-v0's default 200-step time limit for the demo."
    )
    delay = st.slider("Frame delay", 0.00, 0.20, 0.03, 0.005, format="%.3f")
    animation_fps = st.slider("Playback FPS", 5, 30, 12, 1)
    frame_stride = st.slider(
        "Capture every N steps",
        1,
        10,
        2,
        1,
        help="Higher values make deployment lighter. Lower values make animation smoother.",
    )
    seed = st.number_input("Episode seed", value=42, min_value=0, step=1)


if not MODEL_PATH.exists():
    st.error(
        f"Model not found: `{MODEL_PATH}`. Train it first with: `python train_dqn.py --timesteps 500000`"
    )
    st.stop()


@st.cache_resource
def load_model():
    # CPU-only loading is safer for Streamlit Cloud.
    dummy_env = gym.make(ENV_ID, max_episode_steps=500)
    return DQN.load(str(MODEL_PATH), env=dummy_env, device="cpu")


model = load_model()

if SUMMARY_PATH.exists():
    try:
        train_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        mean_reward = train_summary.get("mean_reward_20_episodes")
        if mean_reward is not None and mean_reward <= -190:
            st.warning(
                "The saved DQN looks weak on clean MountainCar "
                f"(mean reward ≈ {mean_reward:.1f}). If both robots stop before the flag, "
                "retrain with: `python train_dqn.py --timesteps 500000` or higher."
            )
        elif mean_reward is not None:
            st.success(
                f"Loaded trained DQN. Clean evaluation mean reward ≈ {mean_reward:.1f}."
            )
    except Exception:
        st.info("Training summary found, but it could not be parsed.")
else:
    st.info(
        "No training summary found. If the clean robot also fails, retrain with: "
        "`python train_dqn.py --timesteps 500000`."
    )

view_left, view_right = st.columns(2)
with view_left:
    st.markdown("### ✅ Clean robot")
    clean_frame_box = st.empty()

with view_right:
    st.markdown("### ⚠️ Attacked robot")
    attack_frame_box = st.empty()

status_box = st.empty()
animation_box = st.empty()
progress_box = st.empty()
final_box = st.empty()


def reached_goal(obs: np.ndarray) -> bool:
    return float(obs[0]) >= GOAL_POSITION


def render_mountain_car_frame(
    obs: np.ndarray, title: str = "", width: int = 640, height: int = 360
) -> np.ndarray:
    """Draw MountainCar without Gymnasium's pygame renderer.

    Gymnasium's built-in RGB renderer depends on pygame/SDL, which can fail on
    Streamlit Cloud. This lightweight renderer uses only PIL and NumPy, so it is
    deployment-friendly.
    """
    position = float(obs[0])
    velocity = float(obs[1])

    min_x, max_x = -1.2, 0.6
    goal_x = GOAL_POSITION

    def hill_y(x: float) -> float:
        # Same visual shape as MountainCar's classic hill: valley in the middle,
        # right-side goal uphill. Values are normalized to [0, 1].
        return math.sin(3.0 * x) * 0.45 + 0.55

    def x_to_px(x: float) -> int:
        return int((x - min_x) / (max_x - min_x) * (width - 1))

    def y_to_px(y: float) -> int:
        # Leave margins for labels/title.
        top_margin = 55
        bottom_margin = 35
        usable = height - top_margin - bottom_margin
        return int(top_margin + (1.0 - y) * usable)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Background bands.
    draw.rectangle([0, 0, width, height], fill=(248, 250, 252))
    draw.rectangle([0, height - 35, width, height], fill=(235, 240, 245))

    # Title and state text.
    draw.text((16, 12), title, fill=(20, 30, 45))
    draw.text(
        (16, 32),
        f"position={position:+.3f}   velocity={velocity:+.3f}",
        fill=(50, 60, 75),
    )

    # Hill curve.
    points = []
    for px in range(width):
        x = min_x + (px / (width - 1)) * (max_x - min_x)
        points.append((px, y_to_px(hill_y(x))))
    draw.line(points, fill=(70, 100, 130), width=4)

    # Fill under the hill.
    polygon = points + [(width - 1, height - 35), (0, height - 35)]
    draw.polygon(polygon, fill=(225, 232, 240))
    draw.line(points, fill=(70, 100, 130), width=4)

    # Goal flag.
    goal_px = x_to_px(goal_x)
    goal_py = y_to_px(hill_y(goal_x))
    draw.line(
        [(goal_px, goal_py), (goal_px, goal_py - 75)], fill=(30, 120, 60), width=4
    )
    draw.polygon(
        [
            (goal_px, goal_py - 75),
            (goal_px + 55, goal_py - 60),
            (goal_px, goal_py - 45),
        ],
        fill=(40, 170, 80),
    )
    draw.text((max(0, goal_px - 30), max(0, goal_py - 100)), "FLAG", fill=(30, 120, 60))

    # Car position on hill.
    car_x = max(min(position, max_x), min_x)
    car_px = x_to_px(car_x)
    car_py = y_to_px(hill_y(car_x))

    # Estimate terrain angle from derivative y = sin(3x). The exact scale is
    # visual only; it just makes the car lean with the slope.
    slope = math.cos(3.0 * car_x)
    angle = math.atan(0.7 * slope)
    car_w, car_h = 48, 24

    def rotate_point(dx: float, dy: float) -> tuple[int, int]:
        ca, sa = math.cos(angle), math.sin(angle)
        x = car_px + dx * ca - dy * sa
        y = car_py - 12 + dx * sa + dy * ca
        return int(x), int(y)

    body = [
        rotate_point(-car_w / 2, -car_h / 2),
        rotate_point(car_w / 2, -car_h / 2),
        rotate_point(car_w / 2, car_h / 2),
        rotate_point(-car_w / 2, car_h / 2),
    ]
    draw.polygon(body, fill=(230, 90, 70), outline=(120, 40, 35))

    # Wheels.
    for dx in (-16, 16):
        wx, wy = rotate_point(dx, car_h / 2 + 4)
        draw.ellipse([wx - 6, wy - 6, wx + 6, wy + 6], fill=(35, 45, 55))

    # Velocity arrow.
    if abs(velocity) > 1e-4:
        arrow_len = max(-45, min(45, velocity * 900))
        draw.line(
            [(car_px, car_py - 45), (car_px + arrow_len, car_py - 45)],
            fill=(30, 90, 200),
            width=3,
        )
        head_x = car_px + arrow_len
        sign = 1 if arrow_len >= 0 else -1
        draw.polygon(
            [
                (head_x, car_py - 45),
                (head_x - 8 * sign, car_py - 51),
                (head_x - 8 * sign, car_py - 39),
            ],
            fill=(30, 90, 200),
        )
        draw.text((car_px - 32, car_py - 70), "velocity", fill=(30, 90, 200))

    return np.array(img)


def make_side_by_side_frame(
    clean_obs: np.ndarray, attack_obs: np.ndarray, clean_done: bool, attack_done: bool
) -> Image.Image:
    """Create one combined frame for the optional animated GIF playback."""
    clean_img = Image.fromarray(
        render_mountain_car_frame(
            clean_obs,
            "Clean robot" + (" - finished" if clean_done else ""),
            width=560,
            height=320,
        )
    )
    attack_img = Image.fromarray(
        render_mountain_car_frame(
            attack_obs,
            "Attacked robot" + (" - finished" if attack_done else ""),
            width=560,
            height=320,
        )
    )
    gap = 20
    header_h = 42
    combined = Image.new(
        "RGB",
        (clean_img.width + attack_img.width + gap, clean_img.height + header_h),
        "white",
    )
    draw = ImageDraw.Draw(combined)
    draw.text((16, 12), "EvasionRL side-by-side rollout", fill=(20, 30, 45))
    draw.text(
        (clean_img.width + gap + 16, 12), "Clean vs attacked robot", fill=(20, 30, 45)
    )
    combined.paste(clean_img, (0, header_h))
    combined.paste(attack_img, (clean_img.width + gap, header_h))
    return combined


def frame_to_png_data_uri(frame: Image.Image) -> str:
    """Convert a PIL frame to a browser-displayable PNG data URI."""
    buf = io.BytesIO()
    frame.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def render_browser_animation(frames: list[Image.Image], fps: int = 12) -> None:
    """Render animation in the browser instead of repeatedly rerendering Streamlit images."""
    if not frames:
        st.warning("No animation frames were captured.")
        return

    data_uris = [frame_to_png_data_uri(frame) for frame in frames]
    interval_ms = int(1000 / max(fps, 1))

    html = f"""
    <div style="width: 100%; text-align: center;">
        <img id="evasionrl_anim"
             src="{data_uris[0]}"
             style="width: 100%; max-width: 1150px; border-radius: 10px; border: 1px solid #ddd;" />
        <div style="margin-top: 8px; font-family: sans-serif;">
            <button onclick="togglePlay()" style="padding: 6px 12px;">Play / Pause</button>
            <button onclick="restartAnim()" style="padding: 6px 12px;">Restart</button>
            <span id="frame_counter" style="margin-left: 12px;">Frame 1 / {len(data_uris)}</span>
        </div>
    </div>

    <script>
        const frames = {json.dumps(data_uris)};
        let idx = 0;
        let playing = true;
        const img = document.getElementById("evasionrl_anim");
        const counter = document.getElementById("frame_counter");

        function showFrame(i) {{
            img.src = frames[i];
            counter.innerText = "Frame " + (i + 1) + " / " + frames.length;
        }}

        function stepAnim() {{
            if (playing) {{
                idx = (idx + 1) % frames.length;
                showFrame(idx);
            }}
        }}

        function togglePlay() {{
            playing = !playing;
        }}

        function restartAnim() {{
            idx = 0;
            playing = true;
            showFrame(idx);
        }}

        setInterval(stepAnim, {interval_ms});
    </script>
    """

    components.html(html, height=430, scrolling=False)


def frames_to_gif_bytes(frames: list[Image.Image], duration_ms: int = 80) -> bytes:
    """Encode frames as GIF bytes. Kept optional so static panels always work."""
    if not frames:
        raise ValueError("No frames to encode")
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()


def describe_stop(
    clean_done: bool,
    attack_done: bool,
    clean_obs: np.ndarray,
    attack_obs: np.ndarray,
    step_count: int,
    max_steps: int,
) -> str:
    clean_goal = reached_goal(clean_obs)
    attack_goal = reached_goal(attack_obs)

    if step_count >= max_steps:
        if clean_goal and attack_goal:
            return "The selected max-step limit was reached after both robots reached the flag."
        if clean_goal and not attack_goal:
            return "The selected max-step limit was reached. The clean robot reached the flag, while the attacked robot did not."
        if attack_goal and not clean_goal:
            return "The selected max-step limit was reached. The attacked robot reached the flag, while the clean robot did not."
        return "The selected max-step limit was reached before either robot reached the flag."

    if clean_done and attack_done:
        if clean_goal and attack_goal:
            return "Both environment episodes ended after reaching the flag."
        return "Both environment episodes ended before the selected max-step limit."

    if clean_done and not attack_done:
        return "The clean robot episode ended, but the attacked robot was allowed to keep running until the selected max-step limit."
    if attack_done and not clean_done:
        return "The attacked robot episode ended, but the clean robot was allowed to keep running until the selected max-step limit."

    return "The demo stopped."


if run_button:
    final_box.empty()
    animation_box.empty()

    status_box.info(
        "Running the clean and attacked robots now. "
        "The playback animation will appear below in a few moments."
    )

    progress_bar = progress_box.progress(0, text="Running side-by-side simulation...")

    # Override Gymnasium's default MountainCar time limit.
    # MountainCar-v0 normally truncates at 200 steps, which made the demo stop
    # early even when the sidebar max-step value was larger.
    env_clean = gym.make(ENV_ID, max_episode_steps=int(max_steps))
    env_attack = gym.make(ENV_ID, max_episode_steps=int(max_steps))

    clean_obs, _ = env_clean.reset(seed=int(seed))
    attack_obs, _ = env_attack.reset(seed=int(seed))

    # These are the observations used only for visualization.
    # When one robot finishes, its display observation freezes.
    # The other robot can still keep updating independently.
    clean_display_obs = clean_obs.copy()
    attack_display_obs = attack_obs.copy()

    # Always show initial panels immediately, so users see the two views even before the loop finishes.
    clean_frame_box.image(
        render_mountain_car_frame(clean_obs, "Clean robot - start"),
        caption="Clean robot start",
        width="stretch",
    )
    attack_frame_box.image(
        render_mountain_car_frame(attack_obs, "Attacked robot - start"),
        caption="Attacked robot start",
        width="stretch",
    )

    clean_total = 0.0
    attack_total = 0.0

    # Metric 1: Was the attacked robot's decision changed by the perturbation?
    attack_decision_changes = 0
    first_attack_decision_change = None

    # Metric 2: Are the clean robot and attacked robot taking different actions?
    clean_vs_attacked_action_diffs = 0
    first_clean_vs_attacked_diff = None

    last_result = None
    last_clean_action = None
    last_attacked_action = None
    last_attacked_action_without_attack = None
    step_count = 0
    final_clean_done = False
    final_attack_done = False
    clean_done_step = None
    attack_done_step = None
    attack_active_steps = 0
    both_active_steps = 0

    rollout_frames: list[Image.Image] = []

    # Important: do not stop when only one robot reaches the flag.
    # Freeze the finished robot and keep stepping the unfinished robot until max_steps.
    for step in range(max_steps):
        clean_action = None
        attacked_action = None
        attacked_action_without_attack = None

        if not final_clean_done:
            clean_action = predict_action_from_obs(model, clean_obs)

        if not final_attack_done:
            attack_active_steps += 1
            result = make_adversarial_observation(
                model=model,
                env=env_attack,
                obs=attack_obs,
                attack=attack,
                epsilon=epsilon,
                pgd_alpha=max(epsilon / max(pgd_steps, 1), 1e-6),
                pgd_steps=pgd_steps,
                objective=objective,
            )
            attacked_action = result.adv_action
            attacked_action_without_attack = result.clean_action
            last_result = result

            # Correct metric: did the perturbation alter the attacked robot's action relative
            # to what the same attacked robot would have done from its true observation?
            attack_changed_this_decision = (
                attacked_action_without_attack != attacked_action
            )
            if attack_changed_this_decision:
                attack_decision_changes += 1
                if first_attack_decision_change is None:
                    first_attack_decision_change = step

        # Separate metric: did the clean robot's actual action differ from the attacked robot's action?
        # Only compute this while both robots are still active. After one finishes, comparing actions is misleading.
        if (
            (not final_clean_done)
            and (not final_attack_done)
            and clean_action is not None
            and attacked_action is not None
        ):
            both_active_steps += 1
            clean_vs_attacked_diff_this_step = clean_action != attacked_action
            if clean_vs_attacked_diff_this_step:
                clean_vs_attacked_action_diffs += 1
                if first_clean_vs_attacked_diff is None:
                    first_clean_vs_attacked_diff = step

        clean_stepped_this_loop = False
        attack_stepped_this_loop = False

        if not final_clean_done and clean_action is not None:
            clean_next_obs, clean_reward, clean_terminated, clean_truncated, _ = (
                env_clean.step(clean_action)
            )
            clean_total += float(clean_reward)
            clean_obs = clean_next_obs
            clean_display_obs = clean_obs.copy()
            clean_stepped_this_loop = True

            final_clean_done = clean_terminated or clean_truncated
            if final_clean_done and clean_done_step is None:
                clean_done_step = step + 1

        if not final_attack_done and attacked_action is not None:
            attack_next_obs, attack_reward, attack_terminated, attack_truncated, _ = (
                env_attack.step(attacked_action)
            )
            attack_total += float(attack_reward)
            attack_obs = attack_next_obs
            attack_display_obs = attack_obs.copy()
            attack_stepped_this_loop = True

            final_attack_done = attack_terminated or attack_truncated
            if final_attack_done and attack_done_step is None:
                attack_done_step = step + 1

        clean_caption = (
            "Clean robot: finished"
            if final_clean_done
            else "Clean robot: DQN sees true [position, velocity]"
        )
        attack_caption = (
            "Attacked robot: finished"
            if final_attack_done
            else "Attacked robot: DQN sees perturbed [position, velocity]"
        )

        # Static panels are still updated during the run. On Streamlit Cloud,
        # some rapid updates may be skipped, but the panels remain visible and
        # the optional GIF below provides reliable playback after the run.
        # Capture frames for browser-side playback.
        # This is more reliable on Streamlit Cloud than live st.image updates.
        if step == 0 or step % int(frame_stride) == 0 or step == max_steps - 1:
            rollout_frames.append(
                make_side_by_side_frame(
                    clean_display_obs,
                    attack_display_obs,
                    final_clean_done,
                    final_attack_done,
                )
            )

        step_count = step + 1
        if clean_action is not None:
            last_clean_action = clean_action
        if attacked_action is not None:
            last_attacked_action = attacked_action
        if attacked_action_without_attack is not None:
            last_attacked_action_without_attack = attacked_action_without_attack

        progress_bar.progress(
            min(step_count / max_steps, 1.0),
            text=f"Running until max-step limit... step {step_count}/{max_steps}",
        )

        status_box.info(
            f"Simulating rollout... step {step_count}/{max_steps}. "
            "Playback will be generated after the run finishes."
        )

        if final_clean_done and final_attack_done:
            break

    progress_box.empty()

    status_box.success("Simulation finished. Preparing playback and final summary...")

    # Always refresh final static panels before any GIF work. This prevents a
    # GIF encoding/display issue from hiding the two main views.
    clean_frame_box.image(
        render_mountain_car_frame(clean_display_obs, "Clean robot - final"),
        caption="Clean robot final state",
        width="stretch",
    )
    attack_frame_box.image(
        render_mountain_car_frame(attack_display_obs, "Attacked robot - final"),
        caption="Attacked robot final state",
        width="stretch",
    )

    if rollout_frames:
        animation_box.markdown("## Rollout playback")
        with animation_box.container():
            render_browser_animation(rollout_frames, fps=int(animation_fps))

    status_box.empty()

    if last_result is None:
        st.error("No simulation steps were run.")
        st.stop()

    clean_success = reached_goal(clean_obs)
    attack_success = reached_goal(attack_obs)
    stop_text = describe_stop(
        final_clean_done,
        final_attack_done,
        clean_obs,
        attack_obs,
        step_count,
        max_steps,
    )
    attack_decision_change_rate = attack_decision_changes / max(attack_active_steps, 1)
    clean_vs_attacked_diff_rate = clean_vs_attacked_action_diffs / max(
        both_active_steps, 1
    )
    perturb_linf = float(np.linalg.norm(last_result.perturbation, ord=np.inf))

    outcome = (
        "Attack disrupted the robot."
        if clean_success and not attack_success
        else "Attack did not clearly disrupt success in this run."
    )

    final_box.markdown(f"""
## Final summary

**Outcome:** {outcome}

**Why did the simulation stop?** {stop_text}

| Metric | Clean robot | Attacked robot |
|---|---:|---:|
| Reached flag | {"Yes" if clean_success else "No"} | {"Yes" if attack_success else "No"} |
| Final position | `{float(clean_obs[0]):.3f}` | `{float(attack_obs[0]):.3f}` |
| Final velocity | `{float(clean_obs[1]):.3f}` | `{float(attack_obs[1]):.3f}` |
| Total return | `{clean_total:.0f}` | `{attack_total:.0f}` |
| Steps shown | `{step_count}` | `{step_count}` |
| Episode finished at step | `{clean_done_step if clean_done_step is not None else "not finished"}` | `{attack_done_step if attack_done_step is not None else "not finished"}` |
| Last action | `{ACTION_NAMES[int(last_clean_action)]}` | `{ACTION_NAMES[int(last_attacked_action)]}` |

### Attack summary

| Item | Value |
|---|---:|
| Attack type | `{attack}` |
| Epsilon | `{epsilon:.3f}` |
| Objective | `{objective}` |
| Last true attacked observation | `[{float(last_result.clean_obs[0]):.3f}, {float(last_result.clean_obs[1]):.3f}]` |
| Last observation seen by DQN | `[{float(last_result.adv_obs[0]):.3f}, {float(last_result.adv_obs[1]):.3f}]` |
| Last perturbation | `[{float(last_result.perturbation[0]):+.3f}, {float(last_result.perturbation[1]):+.3f}]` |
| Last L∞ perturbation size | `{perturb_linf:.3f}` |
| Action without attack on attacked robot | `{ACTION_NAMES[int(last_attacked_action_without_attack)]}` |
| Action after attack on attacked robot | `{ACTION_NAMES[int(last_attacked_action)]}` |
| Attack changed attacked robot's decision | `{100 * attack_decision_change_rate:.1f}%` of steps |
| Clean-vs-attacked action difference | `{100 * clean_vs_attacked_diff_rate:.1f}%` of steps |
| First attack-caused decision change | `{first_attack_decision_change if first_attack_decision_change is not None else "none"}` |
| First clean-vs-attacked action difference | `{first_clean_vs_attacked_diff if first_clean_vs_attacked_diff is not None else "none"}` |

The demo uses the sidebar max-step value as the actual environment time limit. If one robot reaches the flag early, its view freezes, but the other robot continues until it also reaches the flag or the selected max-step limit is reached.

In MountainCar, the reward is usually `-1` per active environment step until the flag is reached. A shorter successful run can therefore have a less negative return. The important visual result is whether the clean robot reaches the flag while the attacked robot remains far away.
""")
else:
    st.info(
        "Choose attack settings in the sidebar, then click **Run side-by-side demo**."
    )
