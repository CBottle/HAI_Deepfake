import pandas as pd
import os

def main():
    # 파일명 수정
    input_file = "submission_final_srm.csv"
    output_file = "submission_inverted_srm.csv"

    # 로컬 경로 및 D드라이브 경로 체크
    paths_to_check = [
        input_file,
        os.path.join(r"D:\deepfake_Data", input_file),
        os.path.join(r"C:\Users\aunil\HAI_deepfake", input_file)
    ]
    
    target_path = None
    for p in paths_to_check:
        if os.path.exists(p):
            target_path = p
            break

    if target_path:
        print(f"📂 Found file at: {target_path}")
        df = pd.read_csv(target_path)
        # 확률 뒤집기
        df['prob'] = 1.0 - df['prob']
        # 현재 폴더에 저장
        df.to_csv(output_file, index=False)
        print(f"✅ Done! Created {output_file}")
        print(df.head())
    else:
        print(f"❌ Error: {input_file} not found in any known locations.")

if __name__ == '__main__':
    main()
