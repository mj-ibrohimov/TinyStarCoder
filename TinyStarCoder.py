import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from sacrebleu import corpus_bleu

class CodeCompletionDataset:
    def __init__(self, directory):
        self.directory = directory
        self.dataset = []

    def split_code_snippet(self, code_snippet):
        cursor_position = random.randint(1, len(code_snippet) - 1)
        prefix = code_snippet[:cursor_position]
        middle = code_snippet[cursor_position:cursor_position + random.randint(1, 10)]
        suffix = code_snippet[cursor_position + len(middle):]
        return prefix, middle, suffix

    def generate_dataset(self):
        for filename in os.listdir(self.directory):
            if filename.endswith('.py'):
                with open(os.path.join(self.directory, filename), 'r') as file:
                    code = file.read()
                    snippets = code.split('\n')
                    for snippet in snippets:
                        if snippet.strip():
                            prefix, middle, suffix = self.split_code_snippet(snippet)
                            self.dataset.append((prefix, middle, suffix))
        return self.dataset

class TinyStarCoder:
    def __init__(self, checkpoint="bigcode/tiny_starcoder_py", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def generate_completion(self, prefix, suffix):
        input_text = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs)
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion.split("<fim_middle>")[-1].strip()

def evaluate_metrics(generated, actual):
    exact_match = generated.strip() == actual.strip()
    bleu_score = corpus_bleu([generated], [[actual]], force=True).score
    return exact_match, bleu_score

def main(directory):
    dataset_creator = CodeCompletionDataset(directory)
    dataset = dataset_creator.generate_dataset()
    
    model = TinyStarCoder()
    
    results = []
    
    for prefix, middle, suffix in dataset:
        generated = model.generate_completion(prefix, suffix)
        exact, bleu = evaluate_metrics(generated, middle)
        results.append({
            'prefix': prefix,
            'generated': generated,
            'actual': middle,
            'exact_match': exact,
            'bleu_score': bleu,
        })
    
    for result in results:
        print(f"Prefix: {result['prefix']}\nGenerated: {result['generated']}\nActual: {result['actual']}\nExact Match: {result['exact_match']}\nBLEU Score: {result['bleu_score']}\n")

if __name__ == "__main__":
    directory_path = 'path/to/your/code/files'
    main(directory_path)
