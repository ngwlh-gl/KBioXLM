import transformers
import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
tokenizer = transformers.LlamaTokenizer.from_pretrained('/data1/gl/project/ner-relation/kbio-xlm/downstream/LLM_test/models/PMC-llama')
model = transformers.LlamaForCausalLM.from_pretrained('/data1/gl/project/ner-relation/kbio-xlm/downstream/LLM_test/models/PMC-llama', device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
# model.cuda()  # move the model to GPU

prompt_input = (
    'Below is an instruction that describes a task, paired with an input that provides further context.'
    'Write a response that appropriately completes the request.\n\n'
    '### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:'
)
text = 'We conclude that ( a ) cyclophosphamide is a human teratogen , ( b ) a distinct phenotype exists , and ( c ) the safety of CP in pregnancy is in serious question .'
example = {
    # "instruction": "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
    "instruction": "Imagine you are a Named Entity Recognition model, and you need to return the results according to the input I give you as required. ",
    # "input": (
    #     "###Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. "
    #     "She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. "
    #     "She otherwise feels well and is followed by a doctor for her pregnancy. "
    #     "Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air."
    #     "Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. "
    #     "Which of the following is the best treatment for this patient?"
    #     "###Options: A. Ampicillin B. Ceftriaxone C. Doxycycline D. Nitrofurantoin"
    # )
    "input": (
        '''###Question: The entity types you need to identify are as follows:｛“{}”，“{}”｝。
                    If there are entities with entity types mentioned above, the following table should be returned:
                    | Entity Type | Entity Name | 
                    | [Entity Type 1] | [Name1] | 
                    | [Entity Type 2] | [Name2] |
                    ...
                    | [Entity Type n] | [Namen] |
                    Please replace [Entity Type] and [Name] in the table with the specific entity type and name that you have identified.
                    Input: {} Please output the results:'''.format('Drug', 'Adverse Drug Event', text)
    )
}
input_str = [prompt_input.format_map(example)]

model_inputs = tokenizer(
    input_str,
    return_tensors='pt',
    padding=True,
)
print( f"\033[32mmodel_inputs\033[0m: { model_inputs }" )


topk_output = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1000,
    top_k=50
)
output_str = tokenizer.batch_decode(topk_output)
print('model predict: ', output_str[0])