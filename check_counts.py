import os
from pathlib import Path

def main():
    # 1. 경로 설정
    input_dir = Path("HAI_Deepfake/test_data")
    output_dir = Path(r"D:\deepfake_Data\test_frame")
    
    if not output_dir.exists():
        print(f"❌ Error: Output directory not found: {output_dir}")
        return

    # 2. 원본 파일 목록 가져오기
    all_input_files = sorted(list(input_dir.glob('*')))
    total_expected = len(all_input_files)
    print(f"Total files in test_data: {total_expected}")

    # 3. 매칭 확인
    found_count = 0
    missing_files = []
    
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

    for f in all_input_files:
        ext = f.suffix.lower()
        if ext in video_exts:
            # 비디오는 _frames 폴더가 있어야 함
            target_path = output_dir / (f.name + "_frames")
            if target_path.exists() and target_path.is_dir():
                found_count += 1
            else:
                missing_files.append(f.name)
        else:
            # 이미지는 파일 그대로 있어야 함
            target_path = output_dir / f.name
            if target_path.exists() and target_path.is_file():
                found_count += 1
            else:
                missing_files.append(f.name)

    # 4. 결과 보고
    print("\n" + "="*40)
    print(f"📊 Processed Data Status Report")
    print("="*40)
    print(f"Expected: {total_expected}")
    print(f"Found   : {found_count}")
    print(f"Missing : {len(missing_files)}")
    
    if missing_files:
        print("\n🚫 Missing Files List:")
        for m in missing_files:
            print(f"  - {m}")
    else:
        print("\n✅ All files are perfectly processed!")
    print("="*40)

if __name__ == '__main__':
    main()
