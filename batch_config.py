"""
Config object for doing batch inference on OpenAI 
"""
from dataclasses import dataclass, field
from typing import List, Dict
import json
import glob
import os
from pathlib import Path



DEFAULT_SYSTEM_STR = "You are a helpful assistant."
DEFAULT_MODEL = "gpt-4o-2024-08-06"
@dataclass
class OpenAIBatchConfig:
    files: List[str]   # list of files or directories to expand and send out to batch inference
    prompt: List[str]  # PAIR of strings such that the thing that gets sent to openAI is (prompt[0] + doc['text'] + prompt[1])
    output_dir: str    # where the output files should live
    merge_dir: str     # where the merged input/output files live
    response_format: Dict    # structured output
    system_prompt: str = field(default=DEFAULT_SYSTEM_STR)
    max_tokens: int = field(default=1000)
    model: str = field(default=DEFAULT_MODEL)


    def expand_files(self) -> List[str]:
        expanded_files = []
        for file_path in self.files:
            path = Path(file_path)
            if path.is_dir():
                expanded_files.extend([
                    str(f) for f in path.rglob('*')
                    if f.is_file()
                ])
            else:
                expanded_files.extend([
                    str(f) for f in glob.glob(file_path)])
        return expanded_files


    @classmethod
    def from_json(cls, json_path: str) -> 'OpenAIBatchConfig':
        with open(json_path, 'rb') as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: str) -> None: 
        os.makedir(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump({
                'files': self.files,
                'prompt': self.prompt,
                'output_dir': self.output_dir,
                'response_format': self.structure
            }, f, indent=2)

