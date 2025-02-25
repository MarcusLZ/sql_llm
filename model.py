import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model():    
    available_memory = torch.cuda.get_device_properties(0).total_memory
    
    model_name = "defog/sqlcoder-7b-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if available_memory > 20e9:
        # if you have atleast 20GB of GPU memory, run load the model in float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True,
        )
    else:
        # else, load in 4 bits – this is slower and less accurate
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            # torch_dtype=torch.float16,
            load_in_4bit=True,
            device_map="auto",
            use_cache=True,
        )
    return model, tokenizer

def prepare_prompt(question = "Give me the list of the most downloaded files?", db_schema='''CREATE TABLE files (file_id INT PRIMARY KEY, file_name VARCHAR(255), download_count INT);'''):
    

    prompt = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

    Generate a SQL query to answer this question: `{question}`

    DDL statements:

    {db_schema}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    The following SQL query best answers the question `{question}`:
    ```sql
    '''
    
    return prompt


def generate_query(prompt,model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
        temperature=0.0,
        top_p=1,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()


    return outputs[0].split("```sql")[1].split(";")[0]