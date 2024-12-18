import itertools
import os
import PyPDF2
import json
from util.s3util import S3Util
from config.config import get_config
from decimal import Decimal

configs = get_config()

S3_BUCKET = configs.s3_bucket
bedrock_price_df = S3Util().read_csv_from_s3(configs.bedrock_limit_csv_path, S3_BUCKET, as_dataframe=True)

def is_valid_combination(config, data):
    # Define your rules here
    regions = ["us-east-1", "us-west-2"]
    if config["region"] not in regions:
        return False
    if config["n_shot_prompts"] > 0 and data["n_shot_prompt_guide"] is None:
        return False
    if config['embedding']["service"] == "bedrock" and config["embedding"]["model"] == "amazon.titan-embed-image-v1":
        if config['vector_dimension'] != 1024 and config['vector_dimension'] != 384 and config['vector_dimension'] != 256:
            return False
    if config['embedding']["service"] == "bedrock" and config["embedding"]["model"] == "amazon.titan-embed-text-v2:0":
        if config['vector_dimension'] != 1024 and config['vector_dimension'] != 512 and config['vector_dimension'] != 256:
            return False
    if config['embedding']["service"] == "bedrock" and config["embedding"]["model"] == "amazon.titan-embed-text-v1":
        if config['vector_dimension'] != 1536:
            return False
    if (config['embedding']["service"] == "bedrock" and config["embedding"]["model"] == "cohere.embed-english-v3") or (
            config['embedding']["service"] == "bedrock" and config["embedding"][
        "model"] == "cohere.embed-multilingual-v3"):
        if config['vector_dimension'] != 1024:
            return False
    if (config['embedding']["service"] == "sagemaker" and config["embedding"]["model"] == "bge-large-en-v1.5") or (
            config['embedding']["service"] == "sagemaker" and config["embedding"]["model"] == "bge-m3"):
        if config['vector_dimension'] != 1024:
            return False
    valid_values = {Decimal('0.5'), Decimal('0.3'), Decimal('0.7'), Decimal('0')}
    if config['temp_retrieval_llm'] not in valid_values:
        return False
    if config['knn_num'] != 3 and config['knn_num'] != 5 and config['knn_num'] != 10 and config['knn_num'] != 15:
        return False
    if (config['embedding']["service"] == "bedrock" and config["embedding"]["model"] == "cohere.embed-english-v3"):
        if config['chunk_size'] != 512 and config['chunk_size'] != 256:
            return False
    return True


def parse_dynamodb(item):
    if isinstance(item, dict):
        if "M" in item:
            return {k: parse_dynamodb(v) for k, v in item["M"].items()}
        elif "L" in item:
            return [parse_dynamodb(v) for v in item["L"]]
        elif "S" in item:
            return item["S"]
        elif "N" in item:
            return float(item["N"]) if "." in item["N"] else int(item["N"])
    return item


def flatten_parameters(stage_data):
    flattened = {}
    for key, value in stage_data.items():
        if isinstance(value, list):
            flattened[key] = value
        else:
            flattened[key] = [value]
    return flattened


def restructure_combination(combination):
    # Define the expected structure
    result = {}

    # Prestep keys
    prestep_keys = ['region', 'gt_data', 'kb_data']
    for key in prestep_keys:
        if key in combination:
            result[key] = combination[key]

    # Indexing keys
    indexing_keys = [
        'chunking_strategy',
        'vector_dimension',
        'chunk_size',
        'chunk_overlap',
        'indexing_algorithm'
    ]
    for key in indexing_keys:
        if key in combination:
            result[key] = combination[key]

    # Embedding (nested)
    if 'service' in combination and 'model' in combination:
        result['embedding'] = {
            'service': combination['service'],
            'model': combination['model']
        }

    # Retrieval keys
    retrieval_keys = [
        'n_shot_prompts',
        'knn_num',
        'temp_retrieval_llm'
    ]
    for key in retrieval_keys:
        if key in combination:
            result[key] = combination[key]

    # Retrieval (nested)
    if 'retrieval_service' in combination and 'retrieval_model' in combination:
        result['retrieval'] = {
            'service': combination['retrieval_service'],
            'model': combination['retrieval_model']
        }

    return result


def estimate_embedding_model_bedrock_price(file_path, configuration, num_tokens_kb_data):
    try:
        df = file_path.copy()
        # return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None
    region = configuration["region"]
    chunk_size = configuration["chunk_size"]
    chunk_overlap = configuration["chunk_overlap"]
    embed_model = configuration["embedding_model"]

    eff_chunk_size = chunk_size * (1 - chunk_overlap / 100)

    num_chunks = num_tokens_kb_data / float(eff_chunk_size)
    num_tokens_with_overlap = num_chunks * float(eff_chunk_size)
    embed_model_price = df[(df['model'] == embed_model) & (df['Region'] == region)]['input_price']
    if embed_model_price.empty:
        return 0
    else:
        embed_model_price = float(embed_model_price.values[0])  # this price is in 1000s of tokens not millions
        embed_price = embed_model_price * num_tokens_with_overlap / 1000000
        return embed_price


def estimate_retrieval_model_bedrock_price(file_path, configuration, avg_prompt_length,
                                           num_prompts):
    try:
        df = file_path.copy()
        # return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None
    region = configuration["region"]
    chunk_size = configuration["chunk_size"]
    gen_model = configuration["retrieval_model"]
    n_shot_prompts = configuration["n_shot_prompts"]
    k = configuration["knn_num"]

    gen_model_price = df[(df['model'] == gen_model) & (df['Region'] == region)]['input_price']
    gen_model_price = float(gen_model_price.values[0])  # this price is in millions of tokens
    gen_model_out_price = df[(df['model'] == gen_model) & (df['Region'] == region)]['output_price']
    gen_model_out_price = float(gen_model_out_price.values[0])  # this price is in millions of tokens
    context_len = k * chunk_size
    prompt_len = (n_shot_prompts + 1) * avg_prompt_length
    total_input_tokens = (context_len + prompt_len) * num_prompts
    retrieval_input_price = gen_model_price * float(total_input_tokens) / 1000000
    total_output_tokens = avg_prompt_length
    retrieval_output_price = gen_model_out_price * float(total_output_tokens) / 1000000
    retrieval_price = retrieval_input_price + retrieval_output_price
    return retrieval_price




def estimate_opensearch_price():
    instance_price = .711 #per hour
    number_of_instances = 3
    number_of_hrs = 3
    overall_cost = instance_price * number_of_instances * number_of_hrs
    

    return overall_cost


def estimate_sagemaker_price():
    sagemaker_price = 1.515 #per hour g5.2xlarge per model
    number_of_instances = 1
    number_of_hrs = 3
    num_models=1
    overall_cost = sagemaker_price * number_of_instances * number_of_hrs*num_models
    
    return overall_cost



def count_characters_in_file(file_path):
    file_path = S3Util().download_file_from_s3(file_path)
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                content = file.read()
                return len(content)
        elif file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                for page in reader.pages:
                    content += page.extract_text()
                return len(content)
        else:
            return "Unsupported file format."
    except FileNotFoundError:
        return "File not found."


def read_gt_data(file_path):
    """
    Reads a GT-data file and outputs the number of JSON entries and the total number of characters in the file.

    Parameters:
    file_path (str): The path to the GT-data file.

    Returns:
    tuple: A tuple containing the number of JSON entries and the total number of characters in the file.
    """
    try:
        json_entries, data = S3Util().read_json_and_content_from_s3(file_path)
        num_entries = len(json_entries)
        total_characters = len(data)
        return num_entries, total_characters
    except Exception as e:
        print(f"Error reading the GT-data file: {e}")
        return None, None


def generate_all_combinations(data):
    # Parse the DynamoDB-style JSON
    parsed_data = {k: parse_dynamodb(v) for k, v in data.items()}


    parameters_all = parsed_data["prestep"]
    parameters_all.update(parsed_data["indexing"])
    parameters_all.update(parsed_data["retrieval"])
    parameters_all = {key: value if isinstance(value, list) else [value] for key, value in parameters_all.items()}

    keys = parameters_all.keys()
    combinations = [dict(zip(keys, values)) for values in itertools.product(*parameters_all.values())]

    gt_data = parameters_all["gt_data"][0]
    [num_prompts, num_chars] = read_gt_data(gt_data)

    avg_promopt_length = round(num_chars / num_prompts / 4)

    num_tokens_kb_data = count_characters_in_file(parameters_all["kb_data"][0]) / 4

    configurations = []
    valid_configurations = []
    os_price = estimate_opensearch_price()
    print("Total possible combinations:", len(combinations))
    for combination in combinations:
        # Generate a unique GUID
        configuration = {
            **combination
        }
        configurations.append(configuration)
        if is_valid_combination(configuration, data):
            
            configuration = {
                **{k: v for k, v in configuration.items() if k not in ["embedding", "retrieval", "gt_data", "kb_data"]},
                "embedding_service": configuration["embedding"]["service"],
                "embedding_model": configuration["embedding"]["model"],
                "retrieval_service": configuration["retrieval"]["service"],
                "retrieval_model": configuration["retrieval"]["model"]}
            valid_configurations.append(configuration)

            configuration["directional_pricing"] = 0
            if configuration['embedding_service'] == "bedrock" :
                embedding_price = estimate_embedding_model_bedrock_price(bedrock_price_df, configuration, num_tokens_kb_data)
                configuration["directional_pricing"] += embedding_price
            else:
                configuration["directional_pricing"] +=estimate_sagemaker_price()

            if configuration["retrieval_service"] == "bedrock":
                retrical_price = estimate_retrieval_model_bedrock_price(bedrock_price_df, configuration, avg_promopt_length, num_prompts)
                configuration["directional_pricing"] += retrical_price
            else:
                configuration["directional_pricing"] +=estimate_sagemaker_price()

            configuration["directional_pricing"] += os_price
            configuration["directional_pricing"] +=configuration["directional_pricing"]*0.2 #extra
            configuration["directional_pricing"] = round(configuration["directional_pricing"],2)
                    
                    
    print("valid_configurations: ",len(valid_configurations))
    

    return valid_configurations