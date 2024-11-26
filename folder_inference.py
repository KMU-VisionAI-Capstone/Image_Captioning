import argparse
import os.path as osp
import os
from tqdm.auto import tqdm
import json
import torch
from PIL import Image
from glob import glob
from transformers import BlipForConditionalGeneration, AutoProcessor
import utils
import warnings

warnings.filterwarnings("ignore")


# def get_image_links(path):
#     images=[]
#     with open(path,'r') as json_file:
#         tour_json = json.load(json_file)
#     for id,values in tour_json['contents'].items():
#         if values['image'] != '':
#             img_format = values['image'].split('.')[-1]
#             url = values['image']
#             img_name = values['title']
#             if not os.path.exists("../tour_data/nature/자연_자연관광지_해안절경"):
#                 os.mkdir("../tour_data/nature/자연_자연관광지_해안절경")
#             save_path = f'../tour_data/nature/자연_자연관광지_해안절경/{img_name}.{img_format}'

#             image_read = urllib.request.urlopen(url).read()
#             image_open = open(save_path, 'wb')
#             image_open.write(image_read)
#             image_open.close()
#             images.append({values['title']:save_path})

#     return images


def get_images():
    train = glob("./final_dataset/train/*")
    train_imgs = []
    for train_folder in train:
        train_folder_img = glob(f"{train_folder}/*.jpg")
        train_imgs.append(train_folder_img)

    valid = glob("./final_dataset/val/*")
    valid_imgs = []
    for valid_folder in valid:
        valid_folder_img = glob(f"{valid_folder}/*.jpg")
        valid_imgs.append(valid_folder_img)

    return train_imgs, valid_imgs


@torch.no_grad()
def inference(model, image_links, processor):
    model.eval()

    results = []
    for element in tqdm(image_links, total=len(image_links)):
        image = Image.open(element).convert("RGB")
        title = element.split("/")[-1]

        inputs = processor(images=image, return_tensors="pt").to(args.device)
        pixel_values = inputs["pixel_values"].to(args.device)

        generated_ids = model.generate(pixel_values=pixel_values, max_length=40)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        results.append({"title": title, "caption": caption})

    return results


def main(args):
    torch.cuda.empty_cache()
    utils.set_seeds(args.seed)

    args.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Model
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large", use_cache=False
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large", device_map="auto"
    ).to(args.device)
    model.config.max_length = 40

    trained_weight_path = osp.join(args.weight)
    trained_weight = torch.load(trained_weight_path)

    model.load_state_dict(trained_weight)

    train_image_paths, valid_image_paths = get_images()

    # train_folder
    for folder in train_image_paths:
        folder_name = folder[0].split("/")[3]
        inference_results = inference(model, folder, processor, args.device)
        if not os.path.exists("./final_dataset/train_annotation"):
            os.mkdir("./final_dataset/train_annotation")

        results_file = f"./final_dataset/train_annotation/{folder_name}_inference_results_blip.json"
        json.dump(
            inference_results, open(results_file, "w"), ensure_ascii=False, indent=4
        )

    # valid_folder
    for folder in valid_image_paths:
        folder_name = folder[0].split("/")[3]
        inference_results = inference(model, folder, processor, args.device)
        if not os.path.exists("./final_dataset/val_annotation"):
            os.mkdir("./final_dataset/val_annotation")

        results_file = (
            f"./final_dataset/val_annotation/{folder_name}_inference_results_blip.json"
        )
        json.dump(
            inference_results, open(results_file, "w"), ensure_ascii=False, indent=4
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight", type=str)
    args = parser.parse_args()
    results_file = main(args)
