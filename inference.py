import threading
from concurrent.futures import ThreadPoolExecutor
import os
import json
import asyncio
import argparse
from typing import Tuple, Dict, Any, List
from models.vlmevalkit_model_api import VLMEvalModel
from tqdm import tqdm
from datetime import datetime
import traceback
import logging

from typing import List, Any, Tuple


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenVLM:
    def __init__(self, model_type: str, max_new_token: int):
        self.model = VLMEvalModel(model_type=model_type,max_new_token=max_new_token)
    
    def ask_vlm(self, messages: List[Any], dataset_type: str) -> Tuple[bool, Any]:

        try:
            return self.model.infer(messages, dataset_type)
        except Exception as e:
            logger.error(f"VLM inference error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False, str(e)



def get_instruction_prompt(eval_type: str, question: str, context: str = None) -> str:

    current_date = datetime.now().strftime("%Y-%m-%d")
    
    time_agnostic = (
        "System Prompt: You are a knowledgeable assistant who can answer factual questions.\n"
        f"User Prompt: Given a question and image, you should answer it using your own knowledge based on today's date. Remember, your answer must contain only the name, with no other words.\n"
        f"QUESTION: {question}\n"
        "Your answer:"
    )
    
    temporal_interval = (
        "System Prompt: You are a knowledgeable assistant who can answer factual questions.\n"
        f"User Prompt: Given a question and image, you should answer it using your own knowledge based on the temporal interval. Remember, your answer must contain only the name, with no other words.\n"
        f"QUESTION: {question}\n"
        "Your answer:"
    )

    timestamp = (
        "System Prompt: You are a knowledgeable assistant who can answer factual questions.\n"
        f"User Prompt: Given a question and image, you should answer it using your own knowledge based on the timestamp. Remember, your answer must contain only the name, with no other words.\n"
        f"QUESTION: {question}\n"
        "Your answer:"
    )

    if context is None:
        context = ""
    
    awareness_future_past = (
        "System Prompt: You are a knowledgeable assistant who can answer factual questions.\n"
        f"User Prompt: Given a question and image and its relevant context, you should answer it using your own knowledge or the knowledge provided by the context. Remember, the provided context may not necessarily be up-to-date to answer the question, and your answer must contain only the name, with no other words.\n"
        f"CONTEXT: {context}\n"
        f"QUESTION: {question}\n"
        "Your answer:"
    )
    
    previous_future_unanswerable_date = (
        "System Prompt: You are a knowledgeable assistant who can answer factual questions.\n"
        f"User Prompt: Given a question and image, you should answer it using your own knowledge. Remember, please output 'Unknown' only if the answer does not exist. Otherwise, output the name only.\n"
        f"QUESTION: {question}\n"
        "Your answer:"
    )

    understanding = (
        "System Prompt: You are a knowledgeable assistant who can answer factual questions.\n"
        f"User Prompt: Given a question and image, you should answer the question using your knowledge and reasoning capacity. Remember, your answer must contain only the name, with no other words.\n"
        f"QUESTION: {question}\n"
        "Your answer:"
    )

    ranking_calculation = (
        "System Prompt: You are a knowledgeable assistant who can answer factual questions.\n"
        f"User Prompt: Given a question and image, you should answer the question using your knowledge and reasoning capacity. Remember, your answer must contain only the name, with no other words.\n"
        f"QUESTION: {question}\n"
        "Your answer:"
    )

    robustness = (
        "System Prompt: You are a knowledgeable assistant who can answer factual questions.\n"
        f"User Prompt: Given a question and image, you should answer the question using your knowledge and reasoning capacity. Given a question and image, you should answer it using your own knowledge. Remember, your answer must contain only 'Yes' or 'No'.\n"
        f"QUESTION: {question}\n"
        "Your answer:"
    )

    if eval_type == 'time_agnostic':
        return time_agnostic
    elif eval_type in ['temporal_interval', 'temporal_interval_new'] :
        return temporal_interval
    elif eval_type in ['timestamp', 'timestamp_new']:
        return timestamp
    elif eval_type in ['future_unanswerable_date', 'previous_unanswerable_date']:
        return previous_future_unanswerable_date
    elif eval_type in ['calculation', 'ranking']:
        return ranking_calculation
    elif eval_type in ['awareness_future', 'awareness_past','sufficient_context']:
        return awareness_future_past
    elif eval_type == 'understanding':
        return understanding
    elif eval_type in ['robustness_openqa', 'robustness_temporal_interval','robustness_timestamp']:
        return robustness
    else:
        raise ValueError(f"Invalid eval_type '{eval_type}'. Supported: 'time_agnostic', 'temporal_interval', 'timestamp', 'future_unanswerable_date', 'previous_unanswerable_date', 'calculation', 'ranking', 'awareness_future', 'awareness_past', 'understanding', 'robustness'.")


def process_item(item: Dict[str, Any]) -> Dict[str, List[List[str]]]:

    return {
        "image_mm_questions": [
            [ item['image'], item['mm_questions'] ]
        ],
        "paraphrase_image_mm_questions": [
            [ item['paraphrase_image'], item['mm_questions'] ]
        ],
        "image_mm_paraphrase_completion": [
            [ item['image'], item['mm_paraphrase_completion'] ]
        ],
        "paraphrase_image_mm_paraphrase_completion": [
            [ item['paraphrase_image'], item['mm_paraphrase_completion'] ]
        ],
    }


def inference_item(inference_data: List[Any], dataset_type: str, qa_agent: OpenVLM) -> Tuple[bool, Any]:
    try:
        return qa_agent.ask_vlm(inference_data, dataset_type=dataset_type)
    except Exception as e:
        logger.error(f"Inference item error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, str(e)


def get_dataset_config():

    return {
        'time_agnostic': 'mined_data\Dimension1_time_agnostic.json',
        'temporal_interval': 'mined_data\Dimension1_temporal_interval.json',
        'timestamp': 'mined_data\Dimension1_timestamp.json',
        'awareness_future': 'mined_data\Dimension2_awareness_future.json',
        'awareness_past': 'mined_data\Dimension2_awareness_past.json',
        'future_unanswerable_date': 'mined_data\Dimension3_future_unanswerable_date.json',
        'previous_unanswerable_date': 'mined_data\Dimension3_previous_unanswerable_date.json',
        'understanding': 'mined_data\Dimension4_understanding.json',
        'calculation': 'mined_data\Dimension5_calculation.json',
        'ranking': 'mined_data\Dimension5_ranking.json',
        'robustness': 'mined_data\Dimension6_robustness.json',

    }


def main(
    meta_save_path: str,
    model_name: str,
    check_done: bool,
    data_eval_type: str,
    dataset_type: str,
    max_new_token: int,
    image_path_prefix: str,
):
    try:
        # Get dataset configuration
        dataset_config = get_dataset_config()
        
        # Select dataset path according to data_eval_type
        if data_eval_type not in dataset_config:
            raise ValueError(f"Invalid data_eval_type '{data_eval_type}'. Supported: {list(dataset_config.keys())}")
        
        test_dataset = dataset_config[data_eval_type]
        
        logger.info(f"Loading dataset from: {test_dataset}")
        
        # 1) Load dataset
        with open(test_dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} items from dataset")

        # 2) Prepare output files
        output_dir = os.path.join(meta_save_path, data_eval_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"output_from_{model_name}.jsonl")
        error_output_path = os.path.join(output_dir, f"error_data_{model_name}.jsonl")
        
        logger.info(f"Output will be saved to: {output_path}")

        # 3) If check_done is set, skip items already completed
        if check_done and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as fin:
                done_ids = { json.loads(line)['id'] for line in fin }
            dataset = [it for it in dataset if it['id'] not in done_ids]
            logger.info(f"After filtering, {len(dataset)} items remaining")

        # 4) Initialize model
        logger.info(f"Initializing model: {model_name}")
        qa_agent = OpenVLM(model_type=model_name, max_new_token=max_new_token)
        logger.info("Model initialized successfully")

        # 5) Iterate over data and perform inference item by item
        with open(output_path, "a", encoding="utf-8") as fout:
            for idx, item in enumerate(tqdm(dataset, desc="Processing items")):
                try:
                    logger.info(f"Processing item {idx+1}/{len(dataset)}, ID: {item.get('id', 'unknown')}")
                    
                    data_dict = process_item(item)

                    for eval_name, entries in data_dict.items():
                        try:
                            # Each entries is a list of [image_path, text]
                            img_path, text = entries[0]

                            # Build instruction
                            if item['eval_type'] in ['awareness_future', 'awareness_past']:
                                prompt = get_instruction_prompt(eval_type=item['eval_type'], question=text, context=item['context'])
                            else:
                                prompt = get_instruction_prompt(eval_type=item['eval_type'], question=text)

                            # Check if image path exists
                            full_img_path = image_path_prefix + img_path
                            if not os.path.exists(full_img_path):
                                logger.warning(f"Image not found: {full_img_path}")
                                continue

                            inference_data = [
                                dict(type='image', value=full_img_path),
                                dict(type='text', value=prompt)
                            ]

                            success, answer = inference_item(inference_data, dataset_type, qa_agent)

                            if success:
                                item[f"{eval_name}_prediction"] = answer
                                logger.info(f"Successfully processed {eval_name}")
                            else:
                                logger.error(f"Failed to process {eval_name}: {answer}")
                                # Write to error file
                                with open(error_output_path, "a", encoding="utf-8") as errf:
                                    error_item = item.copy()
                                    error_item['error'] = answer
                                    errf.write(json.dumps(error_item, ensure_ascii=False) + "\n")

                        except Exception as e:
                            logger.error(f"Error processing {eval_name}: {str(e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            continue

                    # Write result
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    fout.flush()

                except Exception as e:
                    logger.error(f"Error processing item {idx}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

        logger.info("Processing completed successfully")

    except Exception as e:
        logger.error(f"Main function error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='VLM Inference Script')
    
    parser.add_argument('--meta_save_path', type=str, required=True,
                        help='Output save path')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name')
    parser.add_argument('--check_done', action='store_true', default=False,
                        help='Whether to check and skip completed items')
    parser.add_argument('--data_eval_type', type=str, required=True,
                        help='Data type (used to select dataset and save path)')
    parser.add_argument('--dataset_type', type=str, required=True,
                        help='Dataset type')
    parser.add_argument('--max_new_token', type=int, required=True,
                        help='Maximum number of new tokens')
    parser.add_argument('--image_path_prefix', type=str, required=True,
                        help='Image path prefix')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Call main function
    main(
        meta_save_path=args.meta_save_path,
        model_name=args.model_name,
        check_done=args.check_done,
        data_eval_type=args.data_eval_type,
        dataset_type=args.dataset_type,
        max_new_token=args.max_new_token,
        image_path_prefix=args.image_path_prefix
    )