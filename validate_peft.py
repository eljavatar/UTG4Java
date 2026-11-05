from peft import LoraConfig, TaskType, AutoPeftModelForSeq2SeqLM, get_peft_model
from transformers import AutoModelForSeq2SeqLM


# https://huggingface.co/docs/peft/index
# https://huggingface.co/docs/peft/quicktour

# To train
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-220m", cache_dir="E:/000_Tesis/notebooks/huggingface_models/")

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# To inferences:
model = AutoPeftModelForSeq2SeqLM.from_pretrained("finetuned_model")


# Ejemplos:
# https://github.com/ZJU-ACES-ISE/chatunitest-core
# https://huggingface.co/zzzghttt/TestGen2-lora
