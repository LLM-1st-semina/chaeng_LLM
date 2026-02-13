import matplotlib.pyplot as plt
import numpy as np

# 1. 가상 VRAM 설정
TOTAL_BLOCKS = 32
BLOCKS_PER_ROW = 8
ROWS = TOTAL_BLOCKS // BLOCKS_PER_ROW

user_colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FFCC99', '#CC99FF', '#99FFFF']

def run_simulation():
    print("="*60)
    print("   chaeng_own: Interactive PagedAttention Simulator")
    print("="*60)
    print("1. Legacy 모드 | 2. PagedAttention 모드")
    choice = input("\n번호 입력: ")
    
    mode = "Legacy" if choice == '1' else "PagedAttention"
    
    # 그래프 초기화
    plt.ion() 
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 창이 뜨자마자 배경을 먼저 그려서 빈 화면 방지
    def draw_base():
        ax.clear()
        for r in range(ROWS):
            for c in range(BLOCKS_PER_ROW):
                ax.add_patch(plt.Rectangle((c, -r-1), 0.9, 0.9, fill=False, edgecolor='#CCCCCC', linewidth=1))
        ax.set_xlim(-0.2, BLOCKS_PER_ROW)
        ax.set_ylim(-ROWS-0.8, 0.2)
        ax.axis('off')

    current_users_data = []
    all_slots = [(r, c) for r in range(ROWS) for c in range(BLOCKS_PER_ROW)]
    if mode == "PagedAttention":
        np.random.seed(42)
        np.random.shuffle(all_slots)

    user_idx = 0
    draw_base()
    plt.show() # 창을 먼저 확실히 띄움

    while True:
        # 터미널 입력 받기 전 그래프가 멈추지 않게 살짝 멈춤
        plt.pause(0.1) 
        
        print(f"\n--- [User {chr(65+user_idx)}] 입력 ---")
        user_input = input(f"문장 입력 (종료 'q'): ")
        
        if user_input.lower() == 'q':
            break
            
        tokens = user_input.strip().split()
        if not tokens: continue
            
        current_users_data.append((f"User {chr(65+user_idx)}", tokens))
        draw_base() # 격자 다시 그리기

        used_count = 0
        if mode == "Legacy":
            ax.set_title(f"[Legacy Mode] - Static Reservation", fontsize=16, color='#D32F2F', pad=20)
            for u_i, (name, u_tokens) in enumerate(current_users_data):
                if u_i < ROWS:
                    for b_idx in range(BLOCKS_PER_ROW):
                        if b_idx < len(u_tokens):
                            ax.add_patch(plt.Rectangle((b_idx, -u_i-1), 0.9, 0.9, color=user_colors[u_i % len(user_colors)]))
                            ax.text(b_idx+0.45, -u_i-0.55, u_tokens[b_idx], ha='center', va='center', fontsize=9, fontweight='bold')
                        else:
                            ax.add_patch(plt.Rectangle((b_idx, -u_i-1), 0.9, 0.9, color='#E0E0E0', alpha=0.6, hatch='////', edgecolor='#BDBDBD'))
                    used_count += BLOCKS_PER_ROW
        else:
            ax.set_title(f"[PagedAttention Mode] - Dynamic Mapping", fontsize=16, color='#2E7D32', pad=20)
            ptr = 0
            for u_i, (name, u_tokens) in enumerate(current_users_data):
                for token in u_tokens:
                    if ptr < TOTAL_BLOCKS:
                        r, c = all_slots[ptr]
                        ax.add_patch(plt.Rectangle((c, -r-1), 0.9, 0.9, color=user_colors[u_i % len(user_colors)]))
                        ax.text(c+0.45, -r-0.55, token, ha='center', va='center', fontsize=9, fontweight='bold')
                        ptr += 1
            used_count = ptr

        ax.text(BLOCKS_PER_ROW/2, -ROWS-0.5, f"Used: {used_count}/{TOTAL_BLOCKS}", ha='center', fontsize=12, fontweight='bold')
        
        # ⭐ 이 세 줄이 실시간 갱신의 핵심입니다
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.1) 
        
        user_idx += 1

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()
