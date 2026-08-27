import os
import time
from huggingface_hub import HfApi


# ============================================================
# CONFIG
# ============================================================

REPO_ID = "Chillguy2026/AIC_2026_data"

LOCAL_ROOT = r"C:\MyD\University\Project\AIC\workspace_vector_anchor\keyframe_vector\L26_v2"

HF_ROOT = "keyframe_vector/Videos_L26"

REPO_TYPE = "dataset"

# Nghỉ giữa mỗi folder để giảm khả năng chạm rate limit
DELAY_BETWEEN_FOLDERS = 20

# Khi gặp 429, chờ hơn một cửa sổ rate-limit 5 phút
RATE_LIMIT_WAIT = 310

MAX_RETRIES = 5


# ============================================================
# UPLOAD
# ============================================================

api = HfApi()


def upload_one_folder(folder_name):
    local_folder = os.path.join(
        LOCAL_ROOT,
        folder_name,
    )

    path_in_repo = (
        f"{HF_ROOT}/{folder_name}"
    )

    print("\n" + "=" * 70)
    print(f"📁 Local : {local_folder}")
    print(f"☁️  HF    : {path_in_repo}")
    print("=" * 70)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api.upload_folder(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                folder_path=local_folder,
                path_in_repo=path_in_repo,
                commit_message=f"Upload {folder_name}",
            )

            print(f"✅ Upload thành công: {folder_name}")
            return True

        except Exception as e:
            error_text = str(e)

            # Rate limit
            if (
                "429" in error_text
                or "Too Many Requests" in error_text
            ):
                print(
                    f"⚠️ Rate limit khi upload {folder_name}."
                )

                print(
                    f"⏳ Chờ {RATE_LIMIT_WAIT} giây "
                    f"rồi thử lại..."
                )

                time.sleep(RATE_LIMIT_WAIT)
                continue

            print(
                f"❌ Lỗi upload {folder_name}:\n{e}"
            )

            return False

    print(
        f"❌ {folder_name} thất bại sau "
        f"{MAX_RETRIES} lần thử."
    )

    return False


def main():
    folders = [
        name
        for name in os.listdir(LOCAL_ROOT)
        if os.path.isdir(
            os.path.join(
                LOCAL_ROOT,
                name,
            )
        )
    ]

    # Đảm bảo:
    # L26_V001 -> L26_V002 -> L26_V003 -> ...
    folders.sort()

    print(f"📦 Tổng số folder: {len(folders)}")

    success = []
    failed = []

    for index, folder_name in enumerate(
        folders,
        start=1,
    ):
        print(
            f"\n🚀 [{index}/{len(folders)}] "
            f"{folder_name}"
        )

        result = upload_one_folder(
            folder_name
        )

        if result:
            success.append(folder_name)
        else:
            failed.append(folder_name)

        if index < len(folders):
            print(
                f"⏳ Nghỉ "
                f"{DELAY_BETWEEN_FOLDERS}s..."
            )

            time.sleep(
                DELAY_BETWEEN_FOLDERS
            )

    print("\n" + "=" * 70)
    print("HOÀN TẤT")
    print("=" * 70)

    print(
        f"✅ Thành công: "
        f"{len(success)}/{len(folders)}"
    )

    print(
        f"❌ Thất bại : "
        f"{len(failed)}/{len(folders)}"
    )

    if failed:
        print("\nCác folder thất bại:")

        for folder in failed:
            print(f"  - {folder}")


if __name__ == "__main__":
    main()