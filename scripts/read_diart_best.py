#!/usr/bin/env python
# scripts/read_diart_best.py
#
# Đọc tham số TỐT NHẤT mà diart.tune tìm được (lưu trong optuna sqlite DB ở
# thư mục output của diart.tune), rồi in ra đúng dạng dòng .env để dán vào.
#
# Cách dùng:
#   python scripts/read_diart_best.py tune_data/study
#
# (optuna đi kèm diart nên đã có sẵn trong env.)

import glob
import os
import sys


# diart param name → biến .env tương ứng trong streaming.py
_ENV_MAP = {
    "tau_active": "DIART_TAU_ACTIVE",
    "rho_update": "DIART_RHO_UPDATE",
    "delta_new":  "DIART_DELTA_NEW",
}


def main(study_dir: str) -> None:
    try:
        import optuna
    except ImportError:
        print("❌ Cần optuna (đi kèm diart): pip install optuna")
        sys.exit(1)

    dbs = glob.glob(os.path.join(study_dir, "**", "*.db"), recursive=True)
    dbs += glob.glob(os.path.join(study_dir, "*.db"))
    dbs = sorted(set(dbs))
    if not dbs:
        print(f"❌ Không tìm thấy file .db (optuna study) trong: {study_dir}")
        print("   Đảm bảo diart.tune đã chạy xong và --output trỏ đúng thư mục.")
        sys.exit(1)

    storage = f"sqlite:///{dbs[0]}"
    print(f"📂 Đọc study từ: {dbs[0]}")

    summaries = optuna.get_all_study_summaries(storage)
    if not summaries:
        print("❌ Không có study nào trong DB.")
        sys.exit(1)

    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)

    print(f"\n🏆 Best trial #{study.best_trial.number}")
    print(f"   Score (DER thấp = tốt): {study.best_value:.4f}")
    print(f"   Params: {study.best_params}")

    print("\n── DÁN VÀO .env ──")
    for k, v in study.best_params.items():
        env_key = _ENV_MAP.get(k, "DIART_" + k.upper())
        if isinstance(v, float):
            print(f"{env_key}={v:.4f}")
        else:
            print(f"{env_key}={v}")
    print("──────────────────")
    print("Sau khi dán, Live Mode & batch-diart sẽ dùng tham số đã tune.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dùng: python scripts/read_diart_best.py <thư_mục_output_của_diart.tune>")
        sys.exit(0)
    main(sys.argv[1])
