import streamlit as st
import numpy as np
from PIL import Image
import io
import copy
from collections import Counter
from itertools import combinations
# 引入剪贴板组件
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

# --- 核心逻辑引擎 ---

def init_state(board_matrix=None, palette=None):
    if board_matrix is not None:
        st.session_state.board = copy.deepcopy(board_matrix)
        st.session_state.palette = palette if palette else DEFAULT_PALETTE
        size = len(board_matrix)
        st.session_state.state = [[0]*size for _ in range(size)]
        st.session_state.history = ["新棋盘及专属色彩已加载，等待操作。"]
    elif 'board' not in st.session_state:
        st.session_state.board = copy.deepcopy(DEFAULT_BOARD)
        st.session_state.palette = DEFAULT_PALETTE
        size = len(st.session_state.board)
        st.session_state.state = [[0]*size for _ in range(size)]
        st.session_state.history = ["默认棋盘已加载，等待操作。"]
        
    st.session_state.undo_stack = []
    st.session_state.step_count = 0

def save_snapshot(action_type="normal", data=None):
    st.session_state.undo_stack.append({
        'state': copy.deepcopy(st.session_state.state),
        'history': copy.deepcopy(st.session_state.history),
        'step_count': st.session_state.step_count,
        'action_type': action_type,
        'data': data
    })

def undo():
    if st.session_state.undo_stack:
        snapshot = st.session_state.undo_stack.pop()
        st.session_state.state = snapshot['state']
        st.session_state.history = snapshot['history']
        st.session_state.step_count = snapshot['step_count']
        return snapshot
    return None

def place_horse(state, board, r, c, depth=0):
    size = len(board)
    h_val = depth + 1
    c_val = -(depth + 1)
    
    state[r][c] = h_val
    for i in range(size):
        if state[r][i] == 0: state[r][i] = c_val
        if state[i][c] == 0: state[i][c] = c_val
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if 0 <= r+dr < size and 0 <= c+dc < size:
                if state[r+dr][c+dc] == 0: state[r+dr][c+dc] = c_val
    color = board[r][c]
    for i in range(size):
        for j in range(size):
            if board[i][j] == color and state[i][j] == 0:
                state[i][j] = c_val

def check_contradiction(state, board):
    size = len(board)
    for i in range(size):
        if all(state[i][j] < 0 for j in range(size)): 
            return True, f"第 {i+1} 行全部被排除"
    for j in range(size):
        if all(state[i][j] < 0 for i in range(size)): 
            return True, f"第 {j+1} 列全部被排除"
            
    unique_colors = set(c for row in board for c in row)
    for color in unique_colors:
        color_cells = [(i,j) for i in range(size) for j in range(size) if board[i][j] == color]
        if all(state[i][j] < 0 for i,j in color_cells): 
            return True, "某色块全是叉死局"
            
    return False, ""

def check_win(state):
    size = len(state)
    horses = sum(1 for row in state for cell in row if cell > 0)
    return horses == size

def logic_step(state, board, depth=0):
    is_dead, reason = check_contradiction(state, board)
    if is_dead:
        return f"💥【盘面死局】当前状态已崩溃：[{reason}]。此路不通！"
        
    if check_win(state):
        return "🎉【大吉大利】所有小马均已完美归位，成功破局！"

    size = len(board)
    c_val = -(depth + 1)
    icon = "🦄" if depth > 0 else "🐎"

    for i in range(size):
        empty = [j for j in range(size) if state[i][j] == 0]
        if len(empty) == 1 and not any(state[i][j] > 0 for j in range(size)):
            place_horse(state, board, i, empty[0], depth)
            return f"【规则3】第 {i+1} 行唯一空格 ({i+1}, {empty[0]+1}) 放置 {icon}"

    for j in range(size):
        empty = [i for i in range(size) if state[i][j] == 0]
        if len(empty) == 1 and not any(state[i][j] > 0 for i in range(size)):
            place_horse(state, board, empty[0], j, depth)
            return f"【规则3】第 {j+1} 列唯一空格 ({empty[0]+1}, {j+1}) 放置 {icon}"
            
    unique_colors = set(c for row in board for c in row)
    for color in unique_colors:
        empty = [(i,j) for i in range(size) for j in range(size) if board[i][j] == color and state[i][j] == 0]
        if len(empty) == 1 and not any(state[i][j] > 0 for i in range(size) for j in range(size) if board[i][j] == color):
            place_horse(state, board, empty[0][0], empty[0][1], depth)
            return f"【规则4】颜色区块唯一空格 ({empty[0][0]+1}, {empty[0][1]+1}) 放置 {icon}"

    for color in unique_colors:
        empty = [(i,j) for i in range(size) for j in range(size) if board[i][j] == color and state[i][j] == 0]
        if not empty: continue
        rows = set(r for r, c in empty)
        if len(rows) == 1:
            r = rows.pop()
            changed = False
            for j in range(size):
                if board[r][j] != color and state[r][j] == 0:
                    state[r][j] = c_val
                    changed = True
            if changed: return f"【规则2】该颜色剩余均在第 {r+1} 行，排除该行其他色"
        cols = set(c for r, c in empty)
        if len(cols) == 1:
            c = cols.pop()
            changed = False
            for i in range(size):
                if board[i][c] != color and state[i][c] == 0:
                    state[i][c] = c_val
                    changed = True
            if changed: return f"【规则2】该颜色剩余均在第 {c+1} 列，排除该列其他色"

    for i in range(size):
        empty_cols = [j for j in range(size) if state[i][j] == 0]
        if empty_cols:
            colors_in_row = set(board[i][j] for j in empty_cols)
            if len(colors_in_row) == 1:
                target_color = colors_in_row.pop()
                changed = False
                for r in range(size):
                    for c in range(size):
                        if r != i and board[r][c] == target_color and state[r][c] == 0:
                            state[r][c] = c_val
                            changed = True
                if changed: return f"【规则7】第 {i+1} 行剩余全为同种颜色，已排除该色在其他行的可能"
                
    for j in range(size):
        empty_rows = [i for i in range(size) if state[i][j] == 0]
        if empty_rows:
            colors_in_col = set(board[i][j] for i in empty_rows)
            if len(colors_in_col) == 1:
                target_color = colors_in_col.pop()
                changed = False
                for r in range(size):
                    for c in range(size):
                        if c != j and board[r][c] == target_color and state[r][c] == 0:
                            state[r][c] = c_val
                            changed = True
                if changed: return f"【规则7】第 {j+1} 列剩余全为同种颜色，已排除该色在其他列的可能"

    color_empty = {c: [] for c in unique_colors}
    color_placed = {c: False for c in unique_colors}
    for i in range(size):
        for j in range(size):
            col_c = board[i][j]
            if state[i][j] == 0:
                color_empty[col_c].append((i, j))
            elif state[i][j] > 0:
                color_placed[col_c] = True
                
    unplaced_colors = [c for c in unique_colors if not color_placed[c] and len(color_empty[c]) > 0]
    
    max_k = min(4, len(unplaced_colors))
    for k in range(2, max_k + 1):
        for combo in combinations(unplaced_colors, k):
            rows_used = set(r for c in combo for r, _ in color_empty[c])
            if len(rows_used) == k:
                changed = False
                for r in rows_used:
                    for j in range(size):
                        if board[r][j] not in combo and state[r][j] == 0:
                            state[r][j] = c_val
                            changed = True
                if changed:
                    return f"【规则8】高级互斥：发现 {k} 种颜色被封死在 {k} 行中，排除对应干扰项"
            
            cols_used = set(col for c in combo for _, col in color_empty[c])
            if len(cols_used) == k:
                changed = False
                for i in range(size):
                    for c_col in cols_used:
                        if board[i][c_col] not in combo and state[i][c_col] == 0:
                            state[i][c_col] = c_val
                            changed = True
                if changed:
                    return f"【规则8】高级互斥：发现 {k} 种颜色被封死在 {k} 列中，排除对应干扰项"

    for i in range(size):
        for j in range(size):
            if state[i][j] == 0:
                test_state = copy.deepcopy(state)
                place_horse(test_state, board, i, j, 99) 
                is_dead, reason = check_contradiction(test_state, board)
                if is_dead:
                    state[i][j] = c_val
                    return f"【规则6】反证：若在 ({i+1}, {j+1}) 放马必导致 [{reason}]，已画 ×"

    return None

def run_deep_dfs(state, board, depth, history_log):
    temp_state = copy.deepcopy(state)
    indent = " " * depth
    
    while True:
        is_dead, reason = check_contradiction(temp_state, board)
        if is_dead: 
            history_log.append(f"{indent} └ 💥 [深度 {depth}] 推导过程崩溃：[{reason}]")
            return False, temp_state
            
        if check_win(temp_state): 
            return True, temp_state
        
        msg = logic_step(temp_state, board, depth)
        if not msg:
            break
            
        if "💥" in msg:
            history_log.append(f"{indent} └ 💥 [深度 {depth}] 推导过程死局！")
            return False, temp_state
            
        if "🎉" in msg:
            return True, temp_state

    size = len(board)
    unique_colors = set(c for row in board for c in row)
    color_empty_counts = {}
    for color in unique_colors:
        empty_cells = [(i,j) for i in range(size) for j in range(size) if board[i][j] == color and temp_state[i][j] == 0]
        if empty_cells:
            color_empty_counts[color] = empty_cells
            
    if not color_empty_counts:
        return False, temp_state 
        
    sorted_colors = sorted(color_empty_counts.items(), key=lambda x: len(x[1]))
    _, target_cells = sorted_colors[0]
    
    modified = False 
    
    for r, c in target_cells:
        if temp_state[r][c] != 0: 
            continue
            
        next_state = copy.deepcopy(temp_state)
        place_horse(next_state, board, r, c, depth + 1)
        history_log.append(f"{indent} └ 🔮 [深度 {depth+1}] 开启探索分支：尝试 ({r+1}, {c+1}) 为 🦄")
        
        success, returned_state = run_deep_dfs(next_state, board, depth + 1, history_log)
        
        if success:
            return True, returned_state
        else:
            history_log.append(f"{indent} └ 💥 [深度 {depth+1}] 分支探索崩盘！退栈并反证 ({r+1}, {c+1}) 是死路，打 ×！")
            temp_state[r][c] = -(depth + 1) 
            modified = True
            
    if modified:
        is_dead, reason = check_contradiction(temp_state, board)
        if is_dead:
            history_log.append(f"{indent} └ 💥 [深度 {depth}] 尝试了所有分支均覆灭，导致本层崩溃：[{reason}]")
            return False, temp_state
        
        history_log.append(f"{indent} └ ♻️ [深度 {depth}] 本层成功通过反证法排除干扰，利用新盘面重新推导...")
        return run_deep_dfs(temp_state, board, depth, history_log)
        
    return False, temp_state


def process_image(img, grid_size):
    try:

        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
            
        arr = np.array(img)
        is_bg = np.all(arr > 220, axis=2)
        non_bg_rows = np.where(~is_bg.all(axis=1))[0]
        non_bg_cols = np.where(~is_bg.all(axis=0))[0]
        if len(non_bg_rows) > 0 and len(non_bg_cols) > 0:
            top, bottom = non_bg_rows[0], non_bg_rows[-1]
            left, right = non_bg_cols[0], non_bg_cols[-1]
            img = img.crop((left, top, right, bottom))
        width, height = img.size
        cell_w = width / grid_size
        cell_h = height / grid_size
        
        pixels_list = []
        for i in range(grid_size):
            for j in range(grid_size):
                cx = int((j + 0.5) * cell_w)
                cy = int((i + 0.5) * cell_h)
                pixels = []
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            pixels.append(img.getpixel((nx, ny)))
                most_common_pixel = Counter(pixels).most_common(1)[0][0]
                pixels_list.append(most_common_pixel)
                
        pixels_arr = np.array(pixels_list)
        unique_pixels, counts = np.unique(pixels_arr, axis=0, return_counts=True)
        clusters = [[p] * c for p, c in zip(unique_pixels, counts)] 
        
        while len(clusters) > grid_size:
            centers = np.array([np.mean(c, axis=0) for c in clusters])
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dists, np.inf) 
            min_idx = np.argmin(dists)
            i, j = np.unravel_index(min_idx, dists.shape)
            if i > j: i, j = j, i
            clusters[i].extend(clusters[j])
            del clusters[j]
            
        final_centers = np.array([np.mean(c, axis=0) for c in clusters])
        hex_palette = [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}" for c in final_centers]
        
        board = np.zeros((grid_size, grid_size), dtype=int)
        for i in range(grid_size):
            for j in range(grid_size):
                p = pixels_arr[i * grid_size + j]
                closest_idx = int(np.argmin(np.sum((final_centers - p)**2, axis=1)))
                board[i][j] = closest_idx
                
        return board.tolist(), hex_palette
    except Exception as e:
        st.error(f"图像识别失败: {e}")
        return None, None

# --- 渲染 UI ---
st.title("🐎 小马数独 解题器")
if 'board' not in st.session_state:
    init_state()

size = len(st.session_state.board)
current_palette = st.session_state.get('palette', DEFAULT_PALETTE)

assumption_depth = sum(1 for snap in st.session_state.undo_stack if snap['action_type'] == 'assume')

with st.sidebar:

    st.header("📷 导入空白棋盘")
    grid_size_input = st.number_input("输入棋盘网格大小（必须准确设置）", min_value=4, max_value=20, value=10)
        
        # 【新增剪贴板支持核心 UI】
    paste_col, upload_col = st.columns([1, 1])
        
    with paste_col:
            # 生成一个专门用来接受剪贴板图像的按钮组件
            paste_result = paste_image_button(
                label="📋 剪贴板",
                background_color="#4CAF50",
                hover_background_color="#45a049",
                errors="ignore"
            )
    with upload_col:
        uploaded_file = st.file_uploader("上传文件", type=['jpg', 'png'], label_visibility="collapsed")

    # 处理粘贴或上传的图片
    img_to_process = None
    if paste_result is not None and paste_result.image_data is not None:
        # 粘贴组件返回的是 base64 图像对象
        img_to_process = paste_result.image_data
        st.success("图片已从剪贴板加载！")
    elif uploaded_file is not None:
        img_to_process = uploaded_file

    if img_to_process is not None:
        if st.button("解析并导入新棋盘", use_container_width=True, type="primary"):
            new_board, new_palette = process_image(img_to_process, grid_size_input)
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
                if "💥" in msg: st.session_state.undo_stack.pop() 
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
                if "💥" in msg or "🎉" in msg: break


    if st.session_state.undo_stack:
        last_snap = st.session_state.undo_stack[-1]
        if last_snap['action_type'] == 'assume':
            r, c = last_snap['data']
            st.write(f"上一步是手动假设 ({r+1}, {c+1}) 为🦄")
            if st.button(f"↩️ 撤回假设，认定 ({r+1}, {c+1}) 是死路", type="primary", use_container_width=True):
                undo()
                st.session_state.state[r][c] = -(assumption_depth) 
                if assumption_depth == 1:
                    st.session_state.step_count += 1
                    step_prefix = f"第 {st.session_state.step_count} 步："
                else:
                    step_prefix = ""
                st.session_state.history.append(f"{step_prefix}⏪ 已回退并反证：当前组合下 ({r+1}, {c+1}) 是死路，已画 ×")
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
        m_row = st.number_input("行 (Row)", min_value=1, max_value=size, value=1)
    with col2:
        m_col = st.number_input("列 (Col)", min_value=1, max_value=size, value=1)
        
    if st.button(f"在 ({m_row}, {m_col}) 放 🦄并自动应用规则", type="secondary", use_container_width=True):
        r, c = m_row - 1, m_col - 1
        save_snapshot("assume", (r, c)) 
        new_depth = assumption_depth + 1
        st.session_state.history.append(f"======== 🔮 进入第 {new_depth} 层推演：假设 ({m_row}, {m_col}) 为 🦄 ========")
        place_horse(st.session_state.state, st.session_state.board, r, c, new_depth)
        
        step_count = 0
        while True:
            msg = logic_step(st.session_state.state, st.session_state.board, new_depth)
            if not msg:
                st.session_state.history.append(f"⚠️【推演卡住】你可以基于此状态，继续叠加新的假设。")
                break
            step_count += 1
            st.session_state.history.append(f"  └ 触发步 {step_count}: {msg}")
            if "💥" in msg:
                st.session_state.history.append("💥【推演死局】此假设失败！请在左侧【撤销中心】点击撤回。")
                break
            if "🎉" in msg: break
        st.rerun()

    m_action = st.radio("普通手动操作", ["× 排除 (打叉)", "🈳 清空该格", "🐎/🦄 仅放马"], horizontal=True)
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
            st.session_state.state[r][c] = (assumption_depth + 1)
            msg_log = f"【手动】仅在 ({m_row}, {m_col}) 强制放 🐎"
            
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
        
        success, final_state = run_deep_dfs(st.session_state.state, st.session_state.board, assumption_depth, history_log)
        
        for log in history_log:
            st.session_state.history.append(log)
            
        st.session_state.state = final_state
        
        if success:
            st.session_state.history.append("🎉 【推演通关】成功找到全局唯一解！")
        else:
            st.session_state.history.append("⚠️ 【推演退栈】引擎穷举排除了大量错误分支。请观察盘面上新增的 × 继续操作。")
        st.rerun()        

    st.markdown("---")
   
        


# --- 带坐标系的 HTML 棋盘渲染 (CSS Grid 重构) ---
BADGE_COLORS = ["#FF4757", "#1E90FF", "#2ED573", "#FFA502", "#9C88FF", "#FF6B81", "#3742FA", "#2F3542"]

board_html = f'<div style="display: grid; grid-template-columns: 30px repeat({size}, 45px); gap: 2px; justify-content: center;">'

board_html += '<div style="width: 30px; height: 30px;"></div>' 
for c in range(size):
    board_html += f'<div style="width: 45px; height: 30px; display: flex; align-items: flex-end; justify-content: center; font-size: 14px; color: #7f8c8d; font-weight: bold;">{c+1}</div>'

for i in range(size):
    board_html += f'<div style="width: 30px; height: 45px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; font-size: 14px; color: #7f8c8d; font-weight: bold;">{i+1}</div>'
    
    for j in range(size):
        color_idx = st.session_state.board[i][j]
        bg_color = current_palette[color_idx % len(current_palette)]
        cell_val = st.session_state.state[i][j]
        
        content = ""
        badge = ""
        
        
        if cell_val > 0:
            depth = cell_val - 1
            if depth == 0: 
                content = "🐎"
            else:
                content = "🦄"

                badge = f'<div style="position: absolute; top: -5px; right: -5px; background: {bg_color}; color: white; border-radius: 50%; width: 17px; height: 17px; font-size: 12px; display: flex; align-items: center; justify-content: center; font-weight: 900; box-shadow: 0 1px 4px rgba(0,0,0,0.6); border: 1.5px solid white; z-index: 10;">{depth}</div>'
                
        # 处理叉 (小于0)
        elif cell_val < 0:
            depth = abs(cell_val) - 1
            content = "❌"
            if depth > 0:
                d_color = BADGE_COLORS[(depth - 1) % len(BADGE_COLORS)]

                badge = f'<div style="position: absolute; top: -5px; right: -5px; background: white; color: {d_color}; border-radius: 50%; width: 16px; height: 16px; font-size: 11px; display: flex; align-items: center; justify-content: center; font-weight: 900; box-shadow: 0 1px 4px rgba(0,0,0,0.5); border: 1.5px solid {d_color}; z-index: 10;">{depth}</div>'
                
        board_html += f'<div title="行: {i+1}, 列: {j+1}" style="position: relative; width: 45px; height: 45px; background-color: {bg_color}; display: flex; align-items: center; justify-content: center; font-size: 24px; border-radius: 4px; box-shadow: inset 0 0 5px rgba(0,0,0,0.1); cursor: crosshair;">{content}{badge}</div>'

board_html += '</div>'
st.markdown(board_html, unsafe_allow_html=True)

# --- 完美分类渲染逻辑日志 ---
st.markdown("### 📝 逻辑推导日志")
history_container = st.container(height=450)
with history_container:
    for idx, log in enumerate(reversed(st.session_state.history)):
        if "💥" in log: st.error(log)
        elif "⚠️" in log: st.warning(log)
        elif "🎉" in log or ("🐎" in log and "🦄" not in log and "推演" not in log and "假设" not in log): st.success(log)
        elif "🦄" in log or "└" in log or "🔮" in log or "推演" in log or "撤回假设" in log or "🤖" in log or "🎯" in log or "♻️" in log: 
            st.markdown(f'<div style="color: #7f8c8d; font-size: 14.5px; padding: 4px 0;">{log}</div>', unsafe_allow_html=True)
        elif "第" in log and "步：" in log: 
            st.markdown(f'<div style="color: #2c3e50; font-size: 15px; font-weight: 500; padding: 4px 0;">{log}</div>', unsafe_allow_html=True)
        elif "⏪" in log or "【手动】" in log: st.info(log)
        else: st.write(log)