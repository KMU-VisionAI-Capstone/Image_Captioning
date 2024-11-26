import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, AutoProcessor
import numpy as np
import warnings

warnings.filterwarnings("ignore")

print("this is running")


@torch.no_grad()
def inference(model, image_array, processor, device):
    """
    이미지 ndarray를 입력받아 캡션을 생성.
    """
    model.eval()

    # numpy 배열을 PIL 이미지로 변환
    image = Image.fromarray(image_array).convert("RGB")

    # Processor로 입력 전처리
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"].to(device)

    # 모델로 캡션 생성
    generated_ids = model.generate(pixel_values=pixel_values, max_length=40)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 결과 반환 (캡션 문자열)
    return caption


def main(image_array):
    """
    np.ndarray 형태의 이미지를 받아 캡션을 생성하고 문자열로 반환.
    weight_path: Fine-tuned된 모델 가중치 파일 경로 (선택 사항)
    """

    weight_path = "/home/jmkim/dev/input-classification/Image_Captioning/epoch8_val_loss1.4723037481307983.pt"

    # 디바이스 설정
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()

    # 모델 및 Processor 로드
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large", use_cache=False
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    ).to(device)
    model.config.max_length = 40

    # Fine-tuned 가중치 로드 (선택 사항)
    if weight_path:
        print(f"Loading fine-tuned weights from: {weight_path}")
        trained_weight = torch.load(weight_path, map_location=device)
        model.load_state_dict(trained_weight)

    # 이미지에 대해 추론 수행
    caption = inference(model, image_array, processor, device)

    # 결과 캡션 출력
    print("Generated Caption:", caption)

    # 문자열로 반환
    return caption


# 테스트 코드 (사용 예)
if __name__ == "__main__":

    # 테스트용 이미지 불러오기
    test_image_path = "/home/jmkim/dev/input-classification/test_data/tower.jpg"  # 실제 이미지 경로 입력
    test_image = Image.open(test_image_path).convert("RGB")
    test_image_array = np.array(test_image)

    # 이미지 ndarray를 main에 전달
    caption = main(test_image_array)
