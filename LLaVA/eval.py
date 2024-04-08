import argparse
import json
import torch
from PIL import Image
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_avg_bleu(predicted_answers, true_answers):
    """
    Tính điểm avg. BLEU giữa các câu trả lời dự đoán và câu trả lời đúng.
    """
    true_answer_dict = {item['question']: item['answer'] for item in true_answers}

    references = []
    hypotheses = []

    for item in predicted_answers:
        question = item['question']
        answer = true_answer_dict.get(question, "")
        predicted = item['predicted_answer'][0] if item['predicted_answer'] else "tôi không có câu trả lời"

        reference = [answer.split()]
        hypothesis = predicted.split()

        references.append(reference)
        hypotheses.append(hypothesis)

    bleu_scores = [sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1) for ref, hyp in zip(references, hypotheses)]

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu

def main(args):
    model_path = args.model_path
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=args.model_base,
        model_name=get_model_name_from_path(model_path)
    )

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    results = {"results": {}}

    for ann_id, ann_details in tqdm(data['annotations'].items()):
        image_id = ann_details["image_id"]
        question = ann_details["question"]
        image_path = args.image_dir + data["images"][str(image_id)]
        image = Image.open(image_path).convert('RGB')
        image = image.resize((336, 336))
        image_tensor = process_images([image], image_processor, model.config)[0].to("cuda:0").half()

        if model.config.mm_use_im_start_end:
            query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            query = DEFAULT_IMAGE_TOKEN + '\n' + question

        query = "Cuộc trò chuyện giữa người dùng và trợ lý trí tuệ nhân tạo. Trợ lý hãy đưa ra câu trả lời hữu ích, chi tiết và lịch sự cho các câu hỏi của người dùng. USER: " + query
        input_ids = tokenizer.encode(query, return_tensors='pt').unsqueeze(0).to("cuda:0")

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0),
                image_sizes=[image.size],
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().split("\n")

        results["results"][ann_id] = {
            "image_id": image_id,
            "question": question,
            "predicted_answer": outputs
        }

    with open(args.output_json, 'w') as f:
        json.dump(results, f)

    print(f"Đã xử lý xong và lưu kết quả vào file: {args.output_json}")

    # Tính điểm avg. BLEU và in ra
    avg_bleu = calculate_avg_bleu(results["results"], data['annotations'].values())
    print(f"Avg. BLEU score: {avg_bleu}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images and questions using LLaVA model.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLaVA model checkpoint.")
    parser.add_argument("--model_base", type=str, required=True, help="Base path for the model.")

    args = parser.parse_args()
    main(args)
