"""
General copy/paste of JakeP's openai inference script. Some minor modifications/additions.
See: https://github.com/allenai/pdelfin/blob/main/pdelfin/data/runopenaibatch.py for pdelfin/jake's version

Extra features:
1. Configs: Will include config parsing to point to:
    - list of LOCAL files/directories to upload 
    - prompt to apply to text field of all files 
    - structured output templates

2. Experiment tracking: 
    - Some internal files to monitor/check some values of experiments 

3. Sandbox mode:
    - sandbox command to run a single file to make sure things work well

"""

import os
import time
import json
import datetime
import argparse
from openai import OpenAI
from tqdm.auto import tqdm
from batch_config import OpenAIBatchConfig
import gzip
from typing import List, Dict
from io import StringIO



MAX_OPENAI_DISK_SPACE = 100 * 1024 * 1024 * 1024 # Max is 100GB on openAI
UPLOAD_STATE_FILENAME = "SENDSILVER_DATA"
ALL_STATES = ["init", "processing", "completed", "errored_out", "could_not_upload"]
FINISHED_STATES = [ "completed", "errored_out" ]

OPENAI_ERROR_STATES = ['failed', 'expired', 'cancelled']
OPENAI_TERMINAL_STATES = ['completed'] + OPENAI_ERROR_STATES


# ===================================================================
# =                      CONFIG PARSING/SETUP                       =
# ===================================================================

def iter_jsonl(file_path: str) -> List[Dict]:
    data = open(file_path, 'rb').read()
    if file_path.endswith('.gz'):
        data = gzip.decompress(data)
    return [json.loads(_) for _ in data.splitlines()]



def format_request(file_path: str, config: OpenAIBatchConfig) -> StringIO:
    # Takes a regular jsonl and makes it openai request compatible using the config
    lines = iter_jsonl(file_path)
    custom_id_base = hashlib.sha256(file_path.encode('utf-8')).hexdigest()[:16]

    request_list = []
    for line_num, line in enumerate(lines):
        custom_id = '%s_%06d' % (custom_id_base, line_num)
        messages = [{"role": "system",
                     "content": config.system_prompt},
                    {"role": "user",
                     "content": '%s%s%s' % (config.prompt[0], line['text'], config.prompt[1])
                    }
                   ]    
        req = {'custom_id': custom_id,
               'method': "POST",
               'url': "/v1/chat/completions",
               'body': {
                  'model': config.model,
                  'messages': messages,     
                  'max_tokens': config.max_tokens,
                  'temperature': 0.1,
                  'log_probs': True,
                  'top_logprobs': 5,
                  'response_format': config.response_format
                }
              }
        request_list.append(json.dumps(req))

    output_str = '\n'.join(request_list)
    output = StringIO(output_str)
    output.seek(0)
    return output


# ===================================================================
# =                   OPENAI API INTERACTIONS                       =
# ===================================================================

# Function to upload a file to OpenAI and start batch processing
def upload_and_start_batch(file_path: str, config: OpenAIBatchConfig, experiment_description=None): # -> batch_id: str
    # Upload the file to OpenAI
    experiment_description = experiment_description or "(mj) OpenAI batch inference"
    formatted_sio = format_request(file_path, config)
    print(f"Uploading {file_path} to OpenAI Batch API...")
    upload_response = client.files.create(file=formatted_sio, purpose="batch")
    file_id = upload_response.id
    print(f"File uploaded successfully: {file_id}")

    # Create a batch job
    print(f"Creating batch job for {file_path}...")
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": experiment_description
        }
    )
    
    batch_id = batch_response.id
    print(f"Batch created successfully: {batch_id}")
    return batch_id


def download_batch_result(batch_id, config): # -> (batch_id: str, status: str)
    # Retrieve the batch result from OpenAI API
    output_dir = config.output_dir
    os.makedir(output_dir, exist_ok=True)

    batch_data = client.batches.retrieve(batch_id)
    status = batch_data.status

    if status != "completed":
        print(f"WARNING: {batch_id} is not completed, status: {batch_data.status}")
        return batch_id, status
    
    if batch_data.output_file_id is None:
        print(f"WARNING: {batch_id} is completed, but not output file was generated")
        return batch_id, status

    print(f"Downloading batch data for {batch_id}")

    file_response = client.files.content(batch_data.output_file_id)

    # Define output file path
    output_file = os.path.join(output_dir, f"{batch_id}.json")

    # Save the result to a file
    with open(output_file, 'w') as f:
        f.write(str(file_response.text))

    return batch_id, status


# =================================================
# =            MJ TRACKING TOOLs                  =
# =================================================

def get_time():
    return datetime.datetime.utcnow().isoformat()

def load_status_file(status_file: str) -> Dict:
    """ Initializes status file if doesn't exist, or loads it if it does """
    return json.loads(open(status_file, 'r').read())

def init_status_file(status_file: str, files: str) -> Dict:
    os.makedir(os.path.dirname(status_file), exist_ok=True)

    status_dict = {'last_updated': get_time(),
                   'total_files': len(files),
                   'processing_files': 0,
                   'completed_files': 0,
                   'errored_files': 0,
                   'files': {f: {'status': 'init'} for f in files}
                  }
    with open(status_file, 'w') as f:
        json.dump(status_dict, f, indent=2)
    return status_dict


def update_status_file(status_file: str, status_dict: Dict):
    status_dict['last_updated']  = get_time()
    with open(status_file, 'w') as f:
        json.dump(status_dict, f, indent=2)




# =================================================
# =                  COMMANDS                     =
# =================================================



def main_sandbox(config: str, status_file: str, max_gb: int, wait: bool, interval: int, experiment_description=None):
    main_upload(config, status_file, max_gb, experiment_description=experiment_description)


def main_upload(config: str, status_file: str, max_gb: int, experiment_description=None):
    # OpenAI setup checks
    starting_free_space = MAX_OPENAI_DISK_SPACE - get_total_space_usage()

    if starting_free_space < (max_gb * 1024**3) * 2:
        raise ValueError(f"Insufficient free space in OpenAI's file storage: Only {starting_free_space} GB left, but 2x{max_gb} GB are required (1x for your uploads, 1x for your results).")   

    # Setup internal tracking
    config = OpenAIBatchConfig.from_json(config)
    all_files = config.expand_files()
    assert len(set(all_files)) == len(all_files), "Need unique full filenames"
    status_dict = init_status_file(status_dict, all_files)


    # Upload all files:
    for f in tqdm(all_files):
        batch_id = upload_and_start_batch(f, config, experiment_description)
        status_dict['files'][f]['batch_id'] = batch_id
        status_dict['files'][f]['status'] = 'processing'
        status_dict['processing_files'] += 1
        update_status_file(status_file, status_dict)



def check(config: str, status_file: str, wait: bool, interval: int):
    config = OpenAIBatchConfig.from_json(config)
    status_dict = load_status_file(status_file)
    while True:
        for filename, file_status in tqdm(status_dict['files'].items()):
            if file_status['status'] in FINISHED_STATES:
                continue
            batch_id = file_status['batch_id']
            _, status = download_batch_result
            if status in OPENAI_ERROR_STATES:
                file_status['status'] == 'errored_out'
                status_dict['processing_files'] -= 1
                status_dict['errored_files'] += 1
            elif status == 'completed':
                file_status['status'] = 'completed'
                status_dict['processing_files'] -= 1
                status_dict['completed_files'] += 1
            else:
                pass
            update_status_file(status_file, status_dict)

        print("Report: %s files | %s processing | %s errored | %s completed" % 
             (status_dict['total_files'],
              status_dict['processing_files'],
              status_dict['errored_files'],
              status_dict['completed_files']))
        if status_dict['processing_files'] == 0:
            break
        if not wait:
            break
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload .jsonl files and process batches in OpenAI API.")

    # General args
    parser.add_argument('--command', type=str, choices=['sandbox', 'upload', 'check', 'clean'], required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--status-file', type=str, required=True, help="Some stuff to help with")
    parser.add_argument('--experiment', type=str)

    # Upload-specific args
    parser.add_argument('--max-gb', type=int, default=25, help="Max number of GB of batch processing files to upload at one time")

    # Sandbox/Check specific args
    parser.add_argument('--wait', type=bool, help="If present/true will wait and check periodically until done")
    parser.add_argument('--interval', type=int, default=10, help="If waiting, will wait this many secs between checks")

    # Main calls
    args = parser.parse_args()
    if args.command == 'sandbox':
        main_sandbox(config=args.config, status_file=args.status_file, max_gb=args.max_gb, wait=args.wait, interval=args.interval, experiment_description=args.experiment)
    elif args.command == 'upload':
        main_upload(config=args.config, status_file=args.status_file, max_gb=args.max_gb, experiment_description=args.experiment)
    elif args.command == 'check':
        main_check(config=args.config, status_file=args.status_file)
    elif args.command == 'clean':
        main_clean()
    else:
        raise NotImplementedError("Unknown command: %s" % args.command)



