# !/bin/bash

conda activate aue8088
# 체크포인트 폴더 경로 설정
CKPT_DIR="ckpts"

# 체크포인트 폴더에 있는 모든 체크포인트 파일에 대해 테스트 실행
for ckpt_file in "$CKPT_DIR"/*.ckpt; do
    if [ -f "$ckpt_file" ]; then
        echo "Testing with checkpoint: $ckpt_file"
        python test.py --ckpt_file "$ckpt_file"
        echo "----------------------------------------"
    fi
done

# 체크포인트 파일이 없는 경우 메시지 출력
if [ ! "$(ls -A $CKPT_DIR 2>/dev/null)" ]; then
    echo "No checkpoint files found in $CKPT_DIR directory."
fi
