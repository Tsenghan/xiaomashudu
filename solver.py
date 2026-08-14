import copy
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_paste_button import paste_image_button

st.set_page_config(page_title="小马游戏求解器", layout="centered")

DEFAULT_PALETTE = [
    "#FFD37F", "#F4A8C7", "#C5D86D", "#E57A77", "#7C8CD6",
    "#A2D8E8", "#C4A5D9", "#F8CDA9", "#63C1E6", "#9D7BD5"
]

DEFAULT_BOARD = [
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 4, 2, 2, 2, 1, 1, 3, 0],
    [0, 0, 4, 2, 2, 2, 3, 3, 3, 0],
    [0, 0, 4, 4, 5, 5, 3, 3, 6, 0],
    [0, 0, 4, 4, 5, 5, 5, 6, 6, 0],
    [0, 8, 0, 4, 0, 6, 0, 0, 6, 0],
    [8, 8, 8, 7, 7, 6, 6, 0, 0, 0],
    [8, 7, 7, 7, 7, 6, 9, 9, 9, 0],
    [8, 7, 7, 7, 7, 0, 9, 9, 9, 9],
    [8, 7, 7, 7, 7, 7, 9, 9, 9, 9]
]

BADGE_COLORS = ["#FF4757", "#1E90FF", "#2ED573", "#FFA502", "#9C88FF", "#FF6B81", "#3742FA", "#2F3542"]


# --- 核心逻辑引擎 ---

def board_shape(board):
    """返回 (行数, 列数)，并确保棋盘是非空矩形。"""
    if not board or not board[0]:
        raise ValueError("棋盘不能为空")
    rows = len(board)
    cols = len(board[0])
    if any(len(row) != cols for row in board):
        raise ValueError("棋盘每一行的列数必须一致")
    return rows, cols


def _board_key(board):
    return tuple(tuple(int(cell) for cell in row) for row in board)


@lru_cache(maxsize=64)
def _cached_color_cells(board_key):
    cells = {}
    for r, row in enumerate(board_key):
        for c, color in enumerate(row):
            cells.setdefault(color, []).append((r, c))
    return tuple((color, tuple(coords)) for color, coords in cells.items())


def color_cells_map(board):
    return dict(_cached_color_cells(_board_key(board)))


def puzzle_rules(board):
    rows, cols = board_shape(board)
    color_count = len(color_cells_map(board))
    return {
        "rows": rows,
        "cols": cols,
        "color_count": color_count,
        "rows_required": color_count == rows,
        "cols_required": color_count == cols,
    }


def init_state(board_matrix=None, palette=None):
    if board_matrix is not None:
        rows, cols = board_shape(board_matrix)
        st.session_state.board = copy.deepcopy(board_matrix)
        st.session_state.palette = list(palette) if palette else DEFAULT_PALETTE.copy()
        st.session_state.state = [[0] * cols for _ in range(rows)]
        st.session_state.history = ["新棋盘及专属色彩已加载，等待操作。"]
    elif "board" not in st.session_state:
        st.session_state.board = copy.deepcopy(DEFAULT_BOARD)
        st.session_state.palette = DEFAULT_PALETTE.copy()
        rows, cols = board_shape(st.session_state.board)
        st.session_state.state = [[0] * cols for _ in range(rows)]
        st.session_state.history = ["默认棋盘已加载，等待操作。"]

    st.session_state.undo_stack = []
    st.session_state.step_count = 0


def save_snapshot(action_type="normal", data=None):
    st.session_state.undo_stack.append({
        "state": copy.deepcopy(st.session_state.state),
        "history": copy.deepcopy(st.session_state.history),
        "step_count": st.session_state.step_count,
        "action_type": action_type,
        "data": data,
    })


def undo():
    if st.session_state.undo_stack:
        snapshot = st.session_state.undo_stack.pop()
        st.session_state.state = snapshot["state"]
        st.session_state.history = snapshot["history"]
        st.session_state.step_count = snapshot["step_count"]
        return snapshot
    return None


def _horse_cells(state):
    return [(r, c) for r, row in enumerate(state) for c, value in enumerate(row) if value > 0]


def can_place_horse(state, board, r, c):
    """判断当前空格是否仍满足行、列、相邻、颜色四类硬约束。"""
    rows, cols = board_shape(board)
    if not (0 <= r < rows and 0 <= c < cols) or state[r][c] != 0:
        return False

    if any(state[r][j] > 0 for j in range(cols)):
        return False
    if any(state[i][c] > 0 for i in range(rows)):
        return False

    target_color = board[r][c]
    for rr, cc in color_cells_map(board)[target_color]:
        if state[rr][cc] > 0:
            return False

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and state[rr][cc] > 0:
                return False
    return True


def place_horse(state, board, r, c, depth=0):
    """放马，并同步排除同行、同列、九宫格邻接以及同颜色的其它格。"""
    rows, cols = board_shape(board)
    h_val = depth + 1
    c_val = -(depth + 1)

    state[r][c] = h_val

    for j in range(cols):
        if state[r][j] == 0:
            state[r][j] = c_val
    for i in range(rows):
        if state[i][c] == 0:
            state[i][c] = c_val

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and state[rr][cc] == 0:
                state[rr][cc] = c_val

    color = board[r][c]
    for rr, cc in color_cells_map(board)[color]:
        if state[rr][cc] == 0:
            state[rr][cc] = c_val


def _max_bipartite_match(groups, axis):
    """检查每个未放置颜色能否分配到互不重复的行/列（必要条件剪枝）。"""
    if not groups:
        return True

    adjacency = {
        color: sorted({cell[axis] for cell in cells})
        for color, cells in groups.items()
    }
    match_slot = {}

    def augment(color, seen):
        for slot in adjacency[color]:
            if slot in seen:
                continue
            seen.add(slot)
            if slot not in match_slot or augment(match_slot[slot], seen):
                match_slot[slot] = color
                return True
        return False

    ordered_colors = sorted(adjacency, key=lambda color: len(adjacency[color]))
    return all(augment(color, set()) for color in ordered_colors)


def _unplaced_color_candidates(state, board):
    result = {}
    for color, cells in color_cells_map(board).items():
        if any(state[r][c] > 0 for r, c in cells):
            continue
        result[color] = [(r, c) for r, c in cells if can_place_horse(state, board, r, c)]
    return result


def check_contradiction(state, board):
    """严格检查当前盘面是否已经违反硬约束或失去任何必要可行性。"""
    rows, cols = board_shape(board)
    rules = puzzle_rules(board)

    if len(state) != rows or any(len(row) != cols for row in state):
        return True, "状态矩阵尺寸与棋盘不一致"

    if rules["color_count"] > min(rows, cols):
        return True, f"颜色区块有 {rules['color_count']} 个，但棋盘只有 {rows} 行 × {cols} 列，受行列互斥限制必然无解"

    horses = _horse_cells(state)
    seen_rows = set()
    seen_cols = set()
    seen_colors = set()
    for r, c in horses:
        color = board[r][c]
        if r in seen_rows:
            return True, f"第 {r + 1} 行出现多匹马"
        if c in seen_cols:
            return True, f"第 {c + 1} 列出现多匹马"
        if color in seen_colors:
            return True, "同一颜色区块出现多匹马"
        seen_rows.add(r)
        seen_cols.add(c)
        seen_colors.add(color)

    for idx, (r1, c1) in enumerate(horses):
        for r2, c2 in horses[idx + 1:]:
            if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
                return True, f"({r1 + 1}, {c1 + 1}) 与 ({r2 + 1}, {c2 + 1}) 两匹马相邻"

    candidates_by_color = _unplaced_color_candidates(state, board)
    for color, candidates in candidates_by_color.items():
        if not candidates:
            return True, f"颜色区块 {color} 已没有任何可放马位置"

    if rules["rows_required"]:
        for r in range(rows):
            if any(state[r][c] > 0 for c in range(cols)):
                continue
            if not any(can_place_horse(state, board, r, c) for c in range(cols)):
                return True, f"第 {r + 1} 行已没有任何可放马位置"

    if rules["cols_required"]:
        for c in range(cols):
            if any(state[r][c] > 0 for r in range(rows)):
                continue
            if not any(can_place_horse(state, board, r, c) for r in range(rows)):
                return True, f"第 {c + 1} 列已没有任何可放马位置"

    if not _max_bipartite_match(candidates_by_color, axis=0):
        return True, "剩余颜色无法分配到互不冲突的不同行"
    if not _max_bipartite_match(candidates_by_color, axis=1):
        return True, "剩余颜色无法分配到互不冲突的不同列"

    return False, ""


def check_win(state, board):
    is_dead, _ = check_contradiction(state, board)
    if is_dead:
        return False

    rules = puzzle_rules(board)
    color_cells = color_cells_map(board)
    for cells in color_cells.values():
        if sum(1 for r, c in cells if state[r][c] > 0) != 1:
            return False

    rows, cols = rules["rows"], rules["cols"]
    if rules["rows_required"] and any(sum(1 for c in range(cols) if state[r][c] > 0) != 1 for r in range(rows)):
        return False
    if rules["cols_required"] and any(sum(1 for r in range(rows) if state[r][c] > 0) != 1 for c in range(cols)):
        return False

    return True


def logic_step(state, board, depth=0):
    is_dead, reason = check_contradiction(state, board)
    if is_dead:
        return f"💥【盘面死局】当前状态已崩溃：[{reason}]。此路不通！"

    if check_win(state, board):
        return "🎉【大吉大利】所有小马均已完美归位，成功破局！"

    rows, cols = board_shape(board)
    rules = puzzle_rules(board)
    c_val = -(depth + 1)
    icon = "🦄" if depth > 0 else "🐎"
    color_cells = color_cells_map(board)

    # 规则3：只有在“每行/列必须有一匹”成立时，唯一候选才能直接放马。
    if rules["rows_required"]:
        for r in range(rows):
            if any(state[r][c] > 0 for c in range(cols)):
                continue
            candidates = [c for c in range(cols) if can_place_horse(state, board, r, c)]
            if len(candidates) == 1:
                place_horse(state, board, r, candidates[0], depth)
                return f"【规则3】第 {r + 1} 行唯一候选 ({r + 1}, {candidates[0] + 1}) 放置 {icon}"

    if rules["cols_required"]:
        for c in range(cols):
            if any(state[r][c] > 0 for r in range(rows)):
                continue
            candidates = [r for r in range(rows) if can_place_horse(state, board, r, c)]
            if len(candidates) == 1:
                place_horse(state, board, candidates[0], c, depth)
                return f"【规则3】第 {c + 1} 列唯一候选 ({candidates[0] + 1}, {c + 1}) 放置 {icon}"

    # 规则4：每个颜色必须恰好一匹。
    candidates_by_color = _unplaced_color_candidates(state, board)
    for color, candidates in candidates_by_color.items():
        if len(candidates) == 1:
            r, c = candidates[0]
            place_horse(state, board, r, c, depth)
            return f"【规则4】颜色区块唯一候选 ({r + 1}, {c + 1}) 放置 {icon}"

    # 规则2：某颜色只能落在同一行/列，则该行/列其它颜色不能再放马。
    for color, candidates in candidates_by_color.items():
        if not candidates:
            continue
        candidate_rows = {r for r, _ in candidates}
        if len(candidate_rows) == 1:
            r = next(iter(candidate_rows))
            changed = False
            for c in range(cols):
                if board[r][c] != color and state[r][c] == 0:
                    state[r][c] = c_val
                    changed = True
            if changed:
                return f"【规则2】该颜色剩余候选均在第 {r + 1} 行，排除该行其他色"

        candidate_cols = {c for _, c in candidates}
        if len(candidate_cols) == 1:
            c = next(iter(candidate_cols))
            changed = False
            for r in range(rows):
                if board[r][c] != color and state[r][c] == 0:
                    state[r][c] = c_val
                    changed = True
            if changed:
                return f"【规则2】该颜色剩余候选均在第 {c + 1} 列，排除该列其他色"

    # 规则7：隐藏唯一色。仅当该行/列本身必须放一匹时成立。
    if rules["rows_required"]:
        for r in range(rows):
            if any(state[r][c] > 0 for c in range(cols)):
                continue
            candidates = [(r, c) for c in range(cols) if can_place_horse(state, board, r, c)]
            colors_in_row = {board[rr][cc] for rr, cc in candidates}
            if candidates and len(colors_in_row) == 1:
                target_color = next(iter(colors_in_row))
                changed = False
                for rr, cc in color_cells[target_color]:
                    if rr != r and state[rr][cc] == 0:
                        state[rr][cc] = c_val
                        changed = True
                if changed:
                    return f"【规则7】第 {r + 1} 行剩余候选全为同种颜色，已排除该色在其他行的可能"

    if rules["cols_required"]:
        for c in range(cols):
            if any(state[r][c] > 0 for r in range(rows)):
                continue
            candidates = [(r, c) for r in range(rows) if can_place_horse(state, board, r, c)]
            colors_in_col = {board[rr][cc] for rr, cc in candidates}
            if candidates and len(colors_in_col) == 1:
                target_color = next(iter(colors_in_col))
                changed = False
                for rr, cc in color_cells[target_color]:
                    if cc != c and state[rr][cc] == 0:
                        state[rr][cc] = c_val
                        changed = True
                if changed:
                    return f"【规则7】第 {c + 1} 列剩余候选全为同种颜色，已排除该色在其他列的可能"

    # 规则8：颜色 Hall 子集（最多看 4 色），对矩形棋盘同样成立。
    unplaced_colors = [color for color, candidates in candidates_by_color.items() if candidates]
    max_k = min(4, len(unplaced_colors))
    for k in range(2, max_k + 1):
        for combo in combinations(unplaced_colors, k):
            rows_used = {r for color in combo for r, _ in candidates_by_color[color]}
            if len(rows_used) == k:
                changed = False
                for r in rows_used:
                    for c in range(cols):
                        if board[r][c] not in combo and state[r][c] == 0:
                            state[r][c] = c_val
                            changed = True
                if changed:
                    return f"【规则8】高级互斥：发现 {k} 种颜色被封锁在 {k} 行中，排除对应干扰项"

            cols_used = {c for color in combo for _, c in candidates_by_color[color]}
            if len(cols_used) == k:
                changed = False
                for r in range(rows):
                    for c in cols_used:
                        if board[r][c] not in combo and state[r][c] == 0:
                            state[r][c] = c_val
                            changed = True
                if changed:
                    return f"【规则8】高级互斥：发现 {k} 种颜色被封锁在 {k} 列中，排除对应干扰项"

    # 规则6：单步反证。严格矛盾检查里已包含行/列匹配可行性剪枝。
    for r in range(rows):
        for c in range(cols):
            if not can_place_horse(state, board, r, c):
                continue
            test_state = copy.deepcopy(state)
            place_horse(test_state, board, r, c, 99)
            is_dead, reason = check_contradiction(test_state, board)
            if is_dead:
                state[r][c] = c_val
                return f"【规则6】反证：若在 ({r + 1}, {c + 1}) 放马必导致 [{reason}]，已画 ×"

    return None


def _state_signature(state):
    """忽略推演深度，只缓存逻辑上的 马/空/叉 三态。"""
    return tuple(tuple(1 if value > 0 else -1 if value < 0 else 0 for value in row) for row in state)


def _select_mrv_group(state, board):
    """MRV：同时比较颜色、必填行、必填列，优先探索候选最少的约束组。"""
    rows, cols = board_shape(board)
    rules = puzzle_rules(board)
    groups = []

    for color, cells in _unplaced_color_candidates(state, board).items():
        if cells:
            groups.append((len(cells), 0, f"颜色 {color}", cells))

    if rules["rows_required"]:
        for r in range(rows):
            if any(state[r][c] > 0 for c in range(cols)):
                continue
            cells = [(r, c) for c in range(cols) if can_place_horse(state, board, r, c)]
            if cells:
                groups.append((len(cells), 1, f"第 {r + 1} 行", cells))

    if rules["cols_required"]:
        for c in range(cols):
            if any(state[r][c] > 0 for r in range(rows)):
                continue
            cells = [(r, c) for r in range(rows) if can_place_horse(state, board, r, c)]
            if cells:
                groups.append((len(cells), 2, f"第 {c + 1} 列", cells))

    if not groups:
        return None, []

    _, _, label, cells = min(groups, key=lambda item: (item[0], item[1]))
    return label, cells


def run_deep_dfs(state, board, depth, history_log, dead_cache=None):
    """MRV + 失败状态缓存的完整 DFS。找到一个可行解即返回。"""
    if dead_cache is None:
        dead_cache = set()

    temp_state = copy.deepcopy(state)
    indent = "  " * depth

    signature = _state_signature(temp_state)
    if signature in dead_cache:
        return False, temp_state

    while True:
        is_dead, reason = check_contradiction(temp_state, board)
        if is_dead:
            history_log.append(f"{indent}└ 💥 [深度 {depth}] 推导过程崩溃：[{reason}]")
            dead_cache.add(_state_signature(temp_state))
            return False, temp_state

        if check_win(temp_state, board):
            return True, temp_state

        msg = logic_step(temp_state, board, depth)
        if not msg:
            break
        if "💥" in msg:
            history_log.append(f"{indent}└ 💥 [深度 {depth}] 推导过程死局！")
            dead_cache.add(_state_signature(temp_state))
            return False, temp_state
        if "🎉" in msg:
            return True, temp_state

    signature = _state_signature(temp_state)
    if signature in dead_cache:
        return False, temp_state

    group_label, target_cells = _select_mrv_group(temp_state, board)
    if not target_cells:
        dead_cache.add(signature)
        return False, temp_state

    modified = False
    history_log.append(f"{indent}└ 🧭 [深度 {depth}] MRV 选择 {group_label}，候选 {len(target_cells)} 个")

    for r, c in target_cells:
        if not can_place_horse(temp_state, board, r, c):
            continue

        next_state = copy.deepcopy(temp_state)
        place_horse(next_state, board, r, c, depth + 1)
        history_log.append(f"{indent}  └ 🔮 [深度 {depth + 1}] 开启探索分支：尝试 ({r + 1}, {c + 1}) 为 🦄")

        success, returned_state = run_deep_dfs(next_state, board, depth + 1, history_log, dead_cache)
        if success:
            return True, returned_state

        history_log.append(f"{indent}  └ 💥 [深度 {depth + 1}] 分支崩盘，反证 ({r + 1}, {c + 1}) 是死路，打 ×")
        temp_state[r][c] = -(depth + 1)
        modified = True

    if modified:
        is_dead, reason = check_contradiction(temp_state, board)
        if is_dead:
            history_log.append(f"{indent}└ 💥 [深度 {depth}] 所有候选覆灭：[{reason}]")
            dead_cache.add(_state_signature(temp_state))
            return False, temp_state

        history_log.append(f"{indent}└ ♻️ [深度 {depth}] 反证排除完成，基于新盘面继续推导...")
        return run_deep_dfs(temp_state, board, depth, history_log, dead_cache)

    dead_cache.add(_state_signature(temp_state))
    return False, temp_state


# --- 图像处理 ---

def normalize_image(source: Any) -> Image.Image:
    """把剪贴板图片和 Streamlit UploadedFile 统一为 PIL.Image.Image。"""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    return Image.open(source).convert("RGB")


def _sample_cell_color(img, r, c, rows, cols):
    """在格子内部多个偏心位置采样，避开边框和中央棋子/图标。"""
    width, height = img.size
    cell_w = width / cols
    cell_h = height / rows
    sample_points = (
        (0.28, 0.28), (0.50, 0.28), (0.72, 0.28),
        (0.28, 0.50),               (0.72, 0.50),
        (0.28, 0.72), (0.50, 0.72), (0.72, 0.72),
    )

    samples = []
    for fx, fy in sample_points:
        cx = int((c + fx) * cell_w)
        cy = int((r + fy) * cell_h)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x = min(max(cx + dx, 0), width - 1)
                y = min(max(cy + dy, 0), height - 1)
                samples.append(img.getpixel((x, y)))

    return tuple(np.median(np.asarray(samples, dtype=np.float32), axis=0).astype(np.uint8))


def process_image(img: Image.Image, rows: int, cols: int, color_count: int):
    """解析 rows × cols 截图，并聚类为指定数量的颜色区块。"""
    try:
        if rows <= 0 or cols <= 0 or color_count <= 0:
            raise ValueError("行数、列数和颜色数量都必须大于 0")
        if color_count > min(rows, cols):
            raise ValueError(f"当前规则下颜色数量不能超过 min(行, 列)={min(rows, cols)}")

        img = img.convert("RGB")
        arr = np.asarray(img)

        # 自动裁剪接近白色的外部背景。
        is_bg = np.all(arr > 220, axis=2)
        non_bg_rows = np.where(~is_bg.all(axis=1))[0]
        non_bg_cols = np.where(~is_bg.all(axis=0))[0]
        if len(non_bg_rows) > 0 and len(non_bg_cols) > 0:
            top, bottom = int(non_bg_rows[0]), int(non_bg_rows[-1])
            left, right = int(non_bg_cols[0]), int(non_bg_cols[-1])
            img = img.crop((left, top, right + 1, bottom + 1))

        pixels_list = [
            _sample_cell_color(img, r, c, rows, cols)
            for r in range(rows)
            for c in range(cols)
        ]
        pixels_arr = np.asarray(pixels_list, dtype=np.float64)
        unique_pixels, counts = np.unique(pixels_arr, axis=0, return_counts=True)

        if len(unique_pixels) < color_count:
            raise ValueError(f"截图中只检测到 {len(unique_pixels)} 种有效采样色，少于设定的 {color_count} 个颜色区块")

        # 只保存中心 + 权重，避免原实现不断复制/拼接大量重复 RGB tuple。
        centers = unique_pixels.astype(np.float64)
        weights = counts.astype(np.float64)

        while len(centers) > color_count:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diff ** 2, axis=-1)
            np.fill_diagonal(dists, np.inf)
            i, j = np.unravel_index(int(np.argmin(dists)), dists.shape)
            if i > j:
                i, j = j, i

            total_weight = weights[i] + weights[j]
            centers[i] = (centers[i] * weights[i] + centers[j] * weights[j]) / total_weight
            weights[i] = total_weight
            centers = np.delete(centers, j, axis=0)
            weights = np.delete(weights, j)

        board = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                pixel = pixels_arr[r * cols + c]
                closest_idx = int(np.argmin(np.sum((centers - pixel) ** 2, axis=1)))
                board[r][c] = closest_idx

        hex_palette = [
            f"#{int(np.clip(center[0], 0, 255)):02x}{int(np.clip(center[1], 0, 255)):02x}{int(np.clip(center[2], 0, 255)):02x}"
            for center in centers
        ]
        return board.tolist(), hex_palette

    except Exception as exc:
        st.error(f"图像识别失败: {exc}")
        return None, None


# --- 渲染 UI ---
st.title("🐎 小马数独 解题器")
if "board" not in st.session_state:
    init_state()

rows, cols = board_shape(st.session_state.board)
rules = puzzle_rules(st.session_state.board)
current_palette = st.session_state.get("palette", DEFAULT_PALETTE)
assumption_depth = sum(1 for snap in st.session_state.undo_stack if snap["action_type"] == "assume")

status_a, status_b, status_c = st.columns(3)
status_a.metric("棋盘", f"{rows} × {cols}")
status_b.metric("颜色区块", rules["color_count"])
status_c.metric("已放马", sum(1 for row in st.session_state.state for cell in row if cell > 0))

rule_notes = ["每个颜色恰好 1 匹", "每行至多 1 匹", "每列至多 1 匹", "相邻格不能同时放马"]
if rules["rows_required"]:
    rule_notes.append("本棋盘每行恰好 1 匹")
if rules["cols_required"]:
    rule_notes.append("本棋盘每列恰好 1 匹")
st.caption("当前规则：" + " · ".join(rule_notes))

with st.sidebar:
    st.header("📷 导入空白棋盘")

    dim_col1, dim_col2 = st.columns(2)
    with dim_col1:
        rows_input = int(st.number_input("行数", min_value=2, max_value=30, value=10, step=1))
    with dim_col2:
        cols_input = int(st.number_input("列数", min_value=2, max_value=30, value=10, step=1))

    color_count_input = int(st.number_input(
        "颜色区块数量",
        min_value=1,
        max_value=max(1, min(rows_input, cols_input)),
        value=min(10, rows_input, cols_input),
        step=1,
        help="每个颜色区块需要放 1 匹马，因此颜色数量不能超过行数和列数中的较小值。",
    ))

    paste_col, upload_col = st.columns([1, 1])

    with paste_col:
        paste_result = paste_image_button(
            label="📋 剪贴板",
            background_color="#4CAF50",
            hover_background_color="#45a049",
            errors="ignore",
        )

    with upload_col:
        uploaded_file = st.file_uploader("上传文件", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    # 统一为 PIL.Image.Image，避免 Pylance 把剪贴板对象误判成 PIL.Image 模块。
    img_to_process: Image.Image | None = None
    if paste_result is not None and paste_result.image_data is not None:
        try:
            img_to_process = normalize_image(paste_result.image_data)
            st.success("图片已从剪贴板加载！")
        except Exception as exc:
            st.error(f"剪贴板图片读取失败: {exc}")
    elif uploaded_file is not None:
        try:
            img_to_process = normalize_image(uploaded_file)
        except Exception as exc:
            st.error(f"上传图片读取失败: {exc}")

    if img_to_process is not None:
        with st.expander("🖼️ 预览待解析图片"):
            st.image(img_to_process, use_container_width=True)

        if st.button("解析并导入新棋盘", use_container_width=True, type="primary"):
            new_board, new_palette = process_image(img_to_process, rows_input, cols_input, color_count_input)
            if new_board:
                init_state(new_board, new_palette)
                st.rerun()

    st.header("⚙️ 应用游戏规则")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("▶️ 单步推导", use_container_width=True):
            save_snapshot("normal")
            msg = logic_step(st.session_state.state, st.session_state.board, assumption_depth)
            if msg:
                if assumption_depth == 0 and "【规则" in msg:
                    st.session_state.step_count += 1
                    msg = f"第 {st.session_state.step_count} 步：{msg}"
                st.session_state.history.append(msg)
                if "💥" in msg:
                    st.session_state.undo_stack.pop()
            else:
                st.session_state.history.append("⚠️ 确定性逻辑卡住。建议使用多层推演。")
                st.session_state.undo_stack.pop()

    with col_b:
        if st.button("⏩ 自动执行", use_container_width=True):
            save_snapshot("normal")
            while True:
                msg = logic_step(st.session_state.state, st.session_state.board, assumption_depth)
                if not msg:
                    st.session_state.history.append("⚠️ 自动引擎卡住：所有显式规则都无法适用了。")
                    break
                if assumption_depth == 0 and "【规则" in msg:
                    st.session_state.step_count += 1
                    msg = f"第 {st.session_state.step_count} 步：{msg}"
                st.session_state.history.append(msg)
                if "💥" in msg or "🎉" in msg:
                    break

    if st.session_state.undo_stack:
        last_snap = st.session_state.undo_stack[-1]
        if last_snap["action_type"] == "assume":
            r, c = last_snap["data"]
            st.write(f"上一步是手动假设 ({r + 1}, {c + 1}) 为🦄")
            if st.button(f"↩️ 撤回假设，认定 ({r + 1}, {c + 1}) 是死路", type="primary", use_container_width=True):
                undo()
                st.session_state.state[r][c] = -max(1, assumption_depth)
                if assumption_depth == 1:
                    st.session_state.step_count += 1
                    step_prefix = f"第 {st.session_state.step_count} 步："
                else:
                    step_prefix = ""
                st.session_state.history.append(
                    f"{step_prefix}⏪ 已回退并反证：当前组合下 ({r + 1}, {c + 1}) 是死路，已画 ×"
                )
                st.rerun()
            if st.button("↩️ 仅撤回推演，保持原样", use_container_width=True):
                undo()
                st.session_state.history.append("⏪ 已单纯撤回推演。")
                st.rerun()
        else:
            if st.button(f"↩️ 撤销普通操作 (剩余 {len(st.session_state.undo_stack)} 步)", use_container_width=True):
                undo()
                st.session_state.history.append("⏪ 撤销成功。")
                st.rerun()
    else:
        st.button("↩️ 撤销栈为空", disabled=True, use_container_width=True)

    if st.button("🔄 完全重置状态", use_container_width=True):
        init_state(st.session_state.board, current_palette)
        st.rerun()

    st.markdown("---")
    st.header("✍️ 手动干预")

    if assumption_depth > 0:
        st.info(f"🌌 当前已处在第 {assumption_depth} 层推演分支中")

    st.markdown("#### 🎯 单点定点假设")
    col1, col2 = st.columns(2)
    with col1:
        m_row = int(st.number_input("行 (Row)", min_value=1, max_value=rows, value=1, step=1))
    with col2:
        m_col = int(st.number_input("列 (Col)", min_value=1, max_value=cols, value=1, step=1))

    if st.button(f"在 ({m_row}, {m_col}) 放 🦄并自动应用规则", type="secondary", use_container_width=True):
        r, c = m_row - 1, m_col - 1
        if not can_place_horse(st.session_state.state, st.session_state.board, r, c):
            st.warning("这个位置当前不满足放马约束，请先检查同行、同列、相邻格或同颜色区块。")
        else:
            save_snapshot("assume", (r, c))
            new_depth = assumption_depth + 1
            st.session_state.history.append(
                f"======== 🔮 进入第 {new_depth} 层推演：假设 ({m_row}, {m_col}) 为 🦄 ========"
            )
            place_horse(st.session_state.state, st.session_state.board, r, c, new_depth)

            step_count = 0
            while True:
                msg = logic_step(st.session_state.state, st.session_state.board, new_depth)
                if not msg:
                    st.session_state.history.append("⚠️【推演卡住】你可以基于此状态，继续叠加新的假设。")
                    break
                step_count += 1
                st.session_state.history.append(f"  └ 触发步 {step_count}: {msg}")
                if "💥" in msg:
                    st.session_state.history.append("💥【推演死局】此假设失败！请在左侧撤回该假设。")
                    break
                if "🎉" in msg:
                    break
            st.rerun()

    m_action = st.radio("普通手动操作", ["× 排除 (打叉)", "🈳 清空该格", "🐎/🦄 强制放马"], horizontal=True)
    if st.button("确认手动操作", use_container_width=True):
        save_snapshot("normal")
        r, c = m_row - 1, m_col - 1
        if "×" in m_action:
            st.session_state.state[r][c] = -(assumption_depth + 1)
            msg_log = f"【手动】在 ({m_row}, {m_col}) 画 ×"
        elif "🈳" in m_action:
            st.session_state.state[r][c] = 0
            msg_log = f"【手动】清空了 ({m_row}, {m_col})"
        else:
            place_horse(st.session_state.state, st.session_state.board, r, c, assumption_depth)
            msg_log = f"【手动】在 ({m_row}, {m_col}) 强制放 🐎 并同步应用排除规则"

        if assumption_depth == 0:
            st.session_state.step_count += 1
            msg_log = f"第 {st.session_state.step_count} 步：{msg_log}"
        st.session_state.history.append(msg_log)
        st.rerun()

    st.markdown("### 彻底卡住了？")
    if st.button(" 干就完了 ", type="primary", use_container_width=True):
        save_snapshot("normal")
        st.session_state.history.append("======== 🤖 大力出奇迹 ========")
        history_log = []
        success, final_state = run_deep_dfs(
            st.session_state.state,
            st.session_state.board,
            assumption_depth,
            history_log,
        )

        for log in history_log:
            st.session_state.history.append(log)

        st.session_state.state = final_state
        if success:
            st.session_state.history.append("🎉 【推演通关】已找到一个满足全部约束的可行解！")
        else:
            st.session_state.history.append("⚠️ 【推演退栈】引擎已排除大量错误分支。请观察盘面上新增的 × 继续操作。")
        st.rerun()

    st.markdown("---")


# --- 带坐标系的 HTML 棋盘渲染 ---
cell_size = max(24, min(45, int(450 / max(cols, 10))))
header_size = max(24, min(30, cell_size))
font_size = max(16, int(cell_size * 0.53))

board_html = (
    f'<div style="display: grid; grid-template-columns: 30px repeat({cols}, {cell_size}px); '
    f'gap: 2px; justify-content: center; overflow-x: auto; padding: 4px 0;">'
)
board_html += f'<div style="width: 30px; height: {header_size}px;"></div>'

for c in range(cols):
    board_html += (
        f'<div style="width: {cell_size}px; height: {header_size}px; display: flex; align-items: flex-end; '
        f'justify-content: center; font-size: 14px; color: #7f8c8d; font-weight: bold;">{c + 1}</div>'
    )

for r in range(rows):
    board_html += (
        f'<div style="width: 30px; height: {cell_size}px; display: flex; align-items: center; '
        f'justify-content: flex-end; padding-right: 8px; font-size: 14px; color: #7f8c8d; '
        f'font-weight: bold;">{r + 1}</div>'
    )

    for c in range(cols):
        color_idx = st.session_state.board[r][c]
        bg_color = current_palette[color_idx % len(current_palette)]
        cell_val = st.session_state.state[r][c]
        content = ""
        badge = ""

        if cell_val > 0:
            depth = cell_val - 1
            if depth == 0:
                content = "🐎"
            else:
                content = "🦄"
                badge = (
                    f'<div style="position: absolute; top: -5px; right: -5px; background: {bg_color}; color: white; '
                    f'border-radius: 50%; width: 17px; height: 17px; font-size: 12px; display: flex; align-items: center; '
                    f'justify-content: center; font-weight: 900; box-shadow: 0 1px 4px rgba(0,0,0,0.6); '
                    f'border: 1.5px solid white; z-index: 10;">{depth}</div>'
                )
        elif cell_val < 0:
            depth = abs(cell_val) - 1
            content = "❌"
            if depth > 0:
                d_color = BADGE_COLORS[(depth - 1) % len(BADGE_COLORS)]
                badge = (
                    f'<div style="position: absolute; top: -5px; right: -5px; background: white; color: {d_color}; '
                    f'border-radius: 50%; width: 16px; height: 16px; font-size: 11px; display: flex; align-items: center; '
                    f'justify-content: center; font-weight: 900; box-shadow: 0 1px 4px rgba(0,0,0,0.5); '
                    f'border: 1.5px solid {d_color}; z-index: 10;">{depth}</div>'
                )

        board_html += (
            f'<div title="行: {r + 1}, 列: {c + 1}" style="position: relative; width: {cell_size}px; height: {cell_size}px; '
            f'background-color: {bg_color}; display: flex; align-items: center; justify-content: center; font-size: {font_size}px; '
            f'border-radius: 4px; box-shadow: inset 0 0 5px rgba(0,0,0,0.1); cursor: crosshair;">{content}{badge}</div>'
        )

board_html += "</div>"
st.markdown(board_html, unsafe_allow_html=True)


# --- 分类渲染逻辑日志 ---
st.markdown("### 📝 逻辑推导日志")
history_container = st.container(height=450)
with history_container:
    for idx, log in enumerate(reversed(st.session_state.history)):
        if "💥" in log:
            st.error(log)
        elif "⚠️" in log:
            st.warning(log)
        elif "🎉" in log or ("🐎" in log and "🦄" not in log and "推演" not in log and "假设" not in log):
            st.success(log)
        elif any(token in log for token in ("🦄", "└", "🔮", "推演", "撤回假设", "🤖", "🎯", "♻️", "🧭")):
            st.markdown(
                f'<div style="color: #7f8c8d; font-size: 14.5px; padding: 4px 0;">{log}</div>',
                unsafe_allow_html=True,
            )
        elif "第" in log and "步：" in log:
            st.markdown(
                f'<div style="color: #2c3e50; font-size: 15px; font-weight: 500; padding: 4px 0;">{log}</div>',
                unsafe_allow_html=True,
            )
        elif "⏪" in log or "【手动】" in log:
            st.info(log)
        else:
            st.write(log)
