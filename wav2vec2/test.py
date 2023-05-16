from datasets import load_dataset
from transformers.models.wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import pdb
from multiprocessing import get_context

# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("models/wav2vec2-base-960h")

# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

#get the array for single row
def map_to_array(batch):
    print("process once")    
    batch["speech"] = [ item["array"] for item in batch["audio"] ]
    return batch

dataset = ds.map(map_to_array, batched=True, batch_size=10)
#dataset = ds.map(map_to_array, batched=True, batch_size=10, remove_columns=ds.column_names)
#dataset = ds.map(map_to_array)

#input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1

pdb.set_trace()
input_values = processor(dataset["speech"], return_tensors="pt", padding="longest").input_values  # Batch size 3

#to cuda normally only for transformer layers 
#inputs = {k: v.to("cuda") for k, v in inputs.items()}
#back to cpu
#inputs = {k: v.cpu() for k, v in inputs.items()}

pdb.set_trace()

# retrieve logits
with torch.no_grad():
    logits = model(input_values).logits


# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
