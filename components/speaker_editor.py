import streamlit as st

speaker_color = ["#e8520a", "#4da6e8", "#4ec94e", "#c97fe8", "#e8c14d"]
default_names = ["Điều tra viên", "Đối tượng", "Nhân chứng", "Người liên quan", "Khác"]

def speaker_editor(stats: dict, key_prefix: str = "spk") -> dict:
    if not stats:
        st.info("Chưa phát hiện người nói")
        return {}
    
    name_map = {}
    cols = st.columns(len(stats))

    for i, (spk, info) in enumerate(stats.items()):
        color = speaker_color[i % len(speaker_color)]
        default = default_names[i] if i < len(default_names) else f"Người {i+1}"
        pct = info.get("percent", 0)
        dur = f"{info['duration']:.0f}s"
        turns = info["turns"]

        with cols[i]:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;
                        border-top:3px solid {color};border-radius:8px;
                        padding:14px;margin-bottom:10px;">
                <div style="font-size:10px;font-weight:700;color:{color};
                            letter-spacing:1px;text-transform:uppercase;
                            margin-bottom:6px;">{spk}</div>
                <div style="font-size:22px;font-weight:800;color:#e6edf3;
                            margin-bottom:2px;">{pct:.1f}%</div>
                <div style="font-size:11px;color:#8b949e;">{dur} · {turns} lượt nói</div>
                <div style="height:3px;background:#30363d;border-radius:2px;margin-top:10px;">
                    <div style="height:3px;width:{min(pct,100):.0f}%;
                                background:{color};border-radius:2px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)
 
            name_map[spk] = st.text_input(
                label       = f"Tên cho {spk}",
                value       = default,
                key         = f"{key_prefix}_{spk}",
                placeholder = f"VD: {default}",
            )
 
    return name_map