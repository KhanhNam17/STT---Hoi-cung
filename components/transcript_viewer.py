import html
import streamlit as st 

speaker_color = ["#e8520a", "#4da6e8", "#4ec94e", "#c97fe8", "#e8c14d"]

def _color_map(turns):
    speakers = list(dict.fromkeys(t.speaker for t in turns))
    return {s: speaker_color[i % len(speaker_color)] for i, s in enumerate(speakers)}

def _ts(sec):
    return f"{int(sec//60):02d}:{int(sec%60):02d}"

def preview(turns, max_turns=4):
    """Hiển thị transcript mẫu cho Batch Mode."""
    if not turns:
        st.info("Chưa có transcript")
        return

    cmap      = _color_map(turns)
    displayed = turns[:max_turns]
    remaining = len(turns) - max_turns

    rows = ""
    for t in displayed:
        color = cmap.get(t.speaker, "#888")
        # html.escape() chuyển < > " & → &lt; &gt; &quot; &amp;
        # Bắt buộc để tránh nội dung transcript phá vỡ cấu trúc HTML
        safe_text    = html.escape(t.text[:140] + ("..." if len(t.text) > 140 else ""))
        safe_speaker = html.escape(t.speaker)

        rows += (
            f'<div style="display:flex;gap:10px;align-items:baseline;'
            f'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="font-family:monospace;font-size:11px;color:{color};'
            f'white-space:nowrap;min-width:50px;">[{_ts(t.start)}]</span>'
            f'<span style="font-size:11px;font-weight:700;color:{color};'
            f'min-width:120px;white-space:nowrap;">{safe_speaker}:</span>'
            f'<span style="font-size:13px;color:#e6edf3;line-height:1.5;">{safe_text}</span>'
            f'</div>'
        )

    more = (
        f'<div style="text-align:center;font-size:11px;color:#484f58;margin-top:8px;">'
        f'… và {remaining} lượt nói khác</div>'
    ) if remaining > 0 else ""

    html_block = (
        f'<div style="background:#0d1117;border:1px solid #30363d;'
        f'border-radius:8px;padding:14px 16px;">'
        f'<div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;'
        f'color:#484f58;margin-bottom:10px;">④ KẾT QUẢ – TRANSCRIPT MẪU</div>'
        f'{rows}{more}'
        f'</div>'
    )

    # Dùng st.html() — render HTML thuần, không qua Markdown parser
    # Tránh Markdown parser escape hoặc misinterpret nội dung transcript
    try:
        st.html(html_block)          # Streamlit >= 1.31
    except AttributeError:
        st.markdown(html_block, unsafe_allow_html=True)  # fallback phiên bản cũ

def full(turns, editable = False): # hiển thị cho Live Mode
    if not turns:
        st.info("Chưa có transcript")
        return 
    
    cmap = _color_map(turns)
    for i, t in enumerate(turns):
        color = cmap.get(t.speaker, "#888")
        col_l, col_r = st.columns([1, 5])
        with col_l:
            st.markdown(f"""
            <div style="text-align:right;padding-top:6px;">
                <span style="background:{color}22;color:{color};border:1px solid {color}55;
                             font-size:10px;font-weight:700;padding:2px 8px;
                             border-radius:4px;display:inline-block;margin-bottom:3px;">
                    {t.speaker}</span>
                <div style="font-family:monospace;font-size:10px;color:#484f58;">{_ts(t.start)}</div>
            </div>""", unsafe_allow_html=True)
        with col_r:
            if editable:
                num_lines = (len(t.text) // 75) + t.text.count('\n') + 1
                dynamic_height = max(68, num_lines * 25)
                t.text = st.text_area(f"Chỉnh sửa đoạn {i}", value=t.text, key=f"te_{i}",
                                      label_visibility="collapsed", height=dynamic_height)
            else:
                st.markdown(f"""
                <div style="padding:8px 12px;background:rgba(255,255,255,0.02);
                            border-left:2px solid {color};border-radius:4px;
                            font-size:14px;color:#e6edf3;line-height:1.65;
                            margin-bottom:4px;">{t.text}</div>""", unsafe_allow_html=True)