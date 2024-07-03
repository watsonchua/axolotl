from gradio_client import Client
from fire import Fire
from tqdm.auto import tqdm
import json


prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""


def main(api_path, input_path, output_path):
    client = Client(api_path)

    with open(input_path, 'r') as f:
        lines = f.readlines()
    json_lines = [json.loads(l) for l in lines]

    with open('/raid/watson/predictions/hansard/reply_by_hyd_alpaca_formatted_test_predictions_openllama3b_alpaca_format_lora.jsonl', 'w') as f:
	    for row in tqdm(json_lines):
		    prompt = prompt_template.format(instruction=row['instruction'], input=row['input'])
		    result = client.predict(
			    prompt,	# str  in 'instruction' Textbox component
				api_name="/predict"
		    )
		    response = result.rsplit("### Response:", maxsplit=1)[-1].strip()
		    output = {"generation": response, **row}
		    f.write(json.dumps(output) + '\n')
    

if __name__ == "__main__":
#    main(input_path="/raid/watson/input_data/hansard/reply_by_hyd_alpaca_formatted_test.jsonl", output_path="/raid/watson/predictions/hansard/llama3_8b_lora_alpaca.jsonl")
    Fire(main)
