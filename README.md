# openai_batch

Some internal tooling to do OpenAI inference. Will ultimately put output data in the `merge_dir` with responses in the `openai_response` field.

## Usage:
- In the `experiments/` directory, make a config json
- Then either run with the `sandbox` command to do the full thing. Or do a flow of `upload`, `check`, `merge`



## Example
Use the interactive jupyter notebook to build a config file. Then run either:
```python openai_batch.py \
--command sandbox \
--config experiments/example/config.json \
--status-file experiments/example/status.json \
--experiment exp_name_goes_here \
--wait \
--interval 10
```

Or if you want to do this in several steps:
```python openai_batch.py \
--command upload \
--config experiments/example/config.json \
--status-file experiments/example/status.json \
--experiment-description exp_name_goes_here
```

then 
```python openai_batch.py \
--command check \
--config experiments/example/config.json \
--status-file experiments/example/status.json \
--wait \
--interval 10
```
and finally 
```python openai_batch.py \
--command merge \
--config experiments/example/config.json \
--status-file experiments/example/status.json
```