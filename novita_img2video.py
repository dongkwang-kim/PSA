import requests
import base64

url = "https://api.novita.ai/v3/async/img2video"

# 이미지 파일을 base64로 인코딩
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 실제 이미지 파일 경로 입력
image_base64 = encode_image_to_base64("your_image.jpg")

payload = {
    "extra": {
        "response_video_type": "mp4",
        "webhook": {
            "url": "https://your-webhook-url.com",  # 실제 웹훅 URL 입력
            "test_mode": {
                "enabled": True,
                "return_task_status": "TASK_STATUS_SUCCEED"
            }
        },
        "enterprise_plan": {"enabled": True}
    },
    "model_name": "SVD",  # 또는 "SVD-XT"
    "image_file": image_base64,
    "frames_num": 14,  # SVD는 14, SVD-XT는 25
    "frames_per_second": 6,
    "image_file_resize_mode": "ORIGINAL_RESOLUTION",  # 또는 "CROP_TO_ASPECT_RATIO"
    "steps": 10,
    "seed": -23,
    "cond_aug": 0.5,
    "enable_frame_interpolation": True
}

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"  # 실제 API 키 입력
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)


# Copy your referral link:
# https://novita.ai/referral?invited_code=NRAZGA
# Copy referral code: NRAZGA
