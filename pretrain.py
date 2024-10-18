from argparse import ArgumentParser
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
#from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from distutils.util import strtobool
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, EvalPrediction
from transformers import RobertaTokenizer, SchedulerType, DataCollatorForLanguageModeling, DataCollatorForPermutationLanguageModeling, DataCollatorForSeq2Seq
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union, Callable, Iterable
import torch
import transformers as t
from tqdm import tqdm
import json
import gc
import os
import time
import numpy as np
import math
import evaluate
from itertools import chain
#from torchtext.data.metrics import bleu_score
from codebleu import calc_codebleu
#import warnings

from t5_data_collator import compute_t5_input_and_target_lengths, DataCollatorForT5MLM, DataCollatorWithDynamicLengthForT5MLM

#warnings.filterwarnings("ignore", category=UserWarning)

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["TOKENIZERS_PARALLELISM"] = 'true'


'''
AuTomatic Learning of Assert Statements (Atlas)

Automatic Learning of Unit Test Cases for Java With Transformers and Dynamic Context

UTCG4Java
UTCGen4Java

UnitTestCaseGenForJava
Unit Test Case Generator for Java

UnitTestGenForJava -> UnitTestGen4Java
utg4java


T5 original usa span corruption (enmascarar un conjunto de palabras) (span_length=3, corrup_ratio=15%):
Original Sequence: Thank you for inviting me to your party last week
Masked sequence: Thank you <X> to <Y> week .
Target sequence: <X> for inviting me <Y> your party last <Z>

CodeT5 usa varios objetivos de preentrenamiento, entre los cualse se incluye Span Prediction (No usa MLM)
'''

parser = ArgumentParser()
parser.add_argument("-cdhf", "--cacheDirHuggingFace", dest="cacheDirHuggingFace", help="Directory where downloaded hugginface models are saved")
parser.add_argument("-cdds", "--cacheDirDatasets", dest="cacheDirDatasets", help="Directory where loaded datasets are saved")
parser.add_argument("-cddst", "--cacheDirDatasetTokenized", dest="cacheDirDatasetTokenized", help="Directory where tokenized loaded datasets are saved")
parser.add_argument("-a", "--accelerator", dest="accelerator", default="gpu", help="Device type to train the model. Options: cpu, gpu")
parser.add_argument("-u", "--hasUniqueDataFile", dest="hasUniqueDataFile", default=False, type=lambda x: bool(strtobool(x)), help="Indicates whether only a single file will be used for train|test|val or whether separate files will be used")
parser.add_argument("-i", "--trainInput", dest="trainInput", help="Training file for the Finetune model")
parser.add_argument("-v", "--valInput", dest="valInput", help="Validation Input file for the model accuracy")
parser.add_argument("-t", "--testInput", dest="testInput", help="Test Input file for the model accuracy")
parser.add_argument("-gas", "--gradientAccumuSteps", dest="gradientAccumuSteps", default=1, type=int, help="Number of steps to accumulate before update weighs")
parser.add_argument("-bst", "--batchSizeTrain", dest="batchSizeTrain", default=8, type=int, help="Number of samples per batch that will be loaded in training dataset")
parser.add_argument("-bsv", "--batchSizeValidation", dest="batchSizeValidation", default=16, type=int, help="Number of samples per batch that will be loaded in validation dataset")
parser.add_argument("-e", "--epochs", dest="epochs", type=int, help="Epochs for train the model")
parser.add_argument("-o", "--modelOutputDir", dest="outputPath", help="Output Directory Path for the model")
parser.add_argument("-logfr", "--loggerFrequencySteps", dest="loggerFrequencySteps", type=int, help="Frequency of how many N steps the results should be logged")
parser.add_argument("-warmup", "--lrWarmupSteps", dest="lrWarmupSteps", type=int, help="Warmup steps to update the learning rate")
parser.add_argument("-warmupRatio", "--lrWarmupRatio", dest="lrWarmupRatio", type=float, help="Warmup ratio to update the learning rate")
parser.add_argument("-fp16", "--useFp16", dest="useFp16", default=True, type=lambda x: bool(strtobool(x)), help="Indicates whether uses fp16 mixed precision")
parser.add_argument("-logsys", "--loggerListener", dest="loggerListener", help="System logging listener. Options: tensorboard. wandb")
parser.add_argument("-rn", "--runName", dest="runName", help="Run name of Execution (Used by WanDB)")
parser.add_argument("-doTrain", "--doTrain", dest="doTrain", default=True, type=lambda x: bool(strtobool(x)), help="Indicates whether performs training")
parser.add_argument("-doTest", "--doTest", dest="doTest", default=True, type=lambda x: bool(strtobool(x)), help="Indicates whether performs testing")
parser.add_argument("-inm", "--inputModel", dest="inputModel", help="Indicates the path of the input model that has been previously trained. If this attribute is passed and doTrain is False, no training is performed and only validation is performed on the test data set")


args = parser.parse_args()
cacheDirHuggingFace = args.cacheDirHuggingFace
cacheDirDatasets = args.cacheDirDatasets
cacheDirDatasetTokenized = args.cacheDirDatasetTokenized
accelerator = args.accelerator
hasUniqueDataFile = args.hasUniqueDataFile
trainInput = args.trainInput
valInput = args.valInput
testInput = args.testInput
outputPath = args.outputPath
gradientAccumuSteps = args.gradientAccumuSteps
batchSizeTrain = args.batchSizeTrain
batchSizeValidation = args.batchSizeValidation
epochs = args.epochs
loggerFrequencySteps = args.loggerFrequencySteps
lrWarmupSteps = args.lrWarmupSteps
lrWarmupRatio = args.lrWarmupRatio
useFp16 = args.useFp16
loggerListener = args.loggerListener
runName = args.runName
doTrain = args.doTrain
doTest = args.doTest
inputModel = args.inputModel

SEED = 42

os.environ["WANDB_PROJECT"] = "utg4java_final"
os.environ["WANDB_LOG_MODEL"] = "end" # end, or checkpoint for log all model checkpoints


# Utilities to Clean memory
# https://saturncloud.io/blog/how-to-clear-gpu-memory-after-pytorch-model-training-without-restarting-kernel/
def report_gpu():
    #print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def show_cache():
    print(f"Cache utilizada: {torch.cuda.memory_allocated()/(1024)} KB")


# Utilities to save files
def save_in_file_one_line(file_path: str, line: str):
    #Removing older version of the file
    if os.path.exists(file_path):
        os.remove(file_path)
    
    #Writing to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(line)

def save_in_file_list_lines(file_path: str, list_lines: List[str], msg_tqdm: str):
    #Removing older version of the file outputs
    if os.path.exists(file_path):
        os.remove(file_path)

    #Writing to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in tqdm(list_lines, desc=msg_tqdm):
            f.write(line + "\n")

def append_dict_to_list(dict_to_append: dict, list_base: list):
    list_base.append("{")

    for key, value in dict_to_append.items():
        list_base.append(f"  '{key}': {value.item() if isinstance(value, torch.Tensor) else value},")
    
    list_base.append("}")

def save_list_dicts_in_file_as_a_json(file_path: str, list_dicts: List[Dict], msg_tqdm: str):
    #Removing older version of the file
    if os.path.exists(file_path):
        os.remove(file_path)
    
    #Writing to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("Log History:\n")
        for dict in tqdm(list_dicts, desc=msg_tqdm):
            #json.dump(dict, f, indent=2) 
            json.dump(dict, f)
            f.write("\n")


def get_model_params_size(model):
    # sum(p.numel() for p in model.parameters() if p.requires_grad) # Only trainable params
    #print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")
    return sum(p.numel() for p in model.parameters())

def format_model_params_size(model_size):
    if model_size is None:
        return None

    if model_size < 1_000_000:
        value_format = "{:.2f} K".format(model_size / 1_000)
    elif model_size >= 1_000_000 and model_size < 1_000_000_000:
        value_format = "{:.2f} M".format(model_size / 1_000_000)
    elif model_size >= 1_000_000_000:
        value_format = "{:.2f} B".format(model_size / 1_000_000_000)
    else:
        value_format = str(model_size)
    
    return value_format


def get_model_disk_size(model):
    # Guardar el modelo en un archivo temporal que después será eliminado
    # temp = tempfile.NamedTemporaryFile(delete=False) # Se crea en la ruta temporal del sistema operativo
    temp_filename = "temp_model.pt"
    torch.save(model.state_dict(), temp_filename) # temp.name

    # Obtener el tamaño del archivo
    size = os.path.getsize(temp_filename)

    # Eliminar el archivo temporal
    os.remove(temp_filename)

    return size

def format_model_disk_size(size_in_bytes):
    if size_in_bytes is None:
        return None
    
    # Convertir el tamaño a KB, MB o GB
    if size_in_bytes < 1_000_000:  # Menos de 1 MB
        value_format = "{:.2f} KB".format(size_in_bytes / 1_000)
    elif size_in_bytes < 1_000_000_000:  # Menos de 1 GB
        value_format = "{:.2f} MB".format(size_in_bytes / 1_000_000)
    else:  # 1 GB o más
        value_format = "{:.2f} GB".format(size_in_bytes / 1_000_000_000)
    
    return value_format


def add_bos_token_to_inputs_ids(tokenizer, input_ids):
    new_inputs_ids = []
    split_inputs = [{'input_ids': ids} for ids in input_ids['input_ids']]
    for index in range(len(split_inputs)):
        new_input = [tokenizer.bos_token_id] + split_inputs[index]['input_ids']
        new_inputs_ids.append(new_input)

    return {
        "input_ids": new_inputs_ids
    }


def span_mask_input_ids(tokenizer, config, input_ids, max_length, mlm_probability = 0.15, mean_noise_span_length = 3.0):
    split_inputs = [{'input_ids': ids} for ids in input_ids['input_ids']]
    #print(split_inputs[0:2])

    #print(f"\npad_token_id: {config.pad_token_id}")
    #print(f"decoder_start_token_id: {config.decoder_start_token_id}")
    
    #for index in range(1):
    for index in range(len(split_inputs)):
        split_inputs[index]['input_ids'] = [tokenizer.bos_token_id] + split_inputs[index]['input_ids']
        #seq_length = len(split_inputs[index]['input_ids'])
        #print(f"\nSeq length: {seq_length}")

        #expanded_inputs_length, targets_length = compute_t5_input_and_target_lengths(
        #    inputs_length=seq_length,
        #    noise_density=mlm_probability,
        #    mean_noise_span_length=mean_noise_span_length,
        #)
        
        #print(f"\nExpanded list: {len(split_inputs[index]['input_ids'])}")
        #print(split_inputs[index])

    data_collator = DataCollatorWithDynamicLengthForT5MLM(
        tokenizer=tokenizer,
        noise_density=mlm_probability,
        mean_noise_span_length=mean_noise_span_length,
        max_length=max_length,
        pad_token_id=config.pad_token_id,
        decoder_start_token_id=config.decoder_start_token_id
    )
    
    masked = data_collator(split_inputs[0:2000])
    #print("\nShow masked samples")
    #print(masked)
    
    #print("\n\n\n\n")

    return ""


# Preprocess inputs using tokenizer. Solo tokenizamos, aún no retornamos tensores
def tokenize_examples(tokenizer, config, examples, max_length, span_mask: bool = True):
    # encode the text inputs
    text_input = examples["text"]
    #model_inputs = tokenizer(text_input, max_length=max_length, padding="max_length", truncation=True)
    #model_inputs = tokenizer(text_input, max_length=max_length, padding="max_length", truncation=True, return_attention_mask=False)
    tokenized_input = tokenizer(text_input, max_length=max_length, truncation=True, return_attention_mask=False, add_special_tokens=False)

    #if span_mask:
    #    span_mask_input_ids(tokenizer, config, tokenized_input, max_length)
    
    #print(tokenized_input.keys())
    # {input_ids: [[], []...]}
    #return tokenized_input
    return add_bos_token_to_inputs_ids(tokenizer, tokenized_input)


# Loads and splits the data into training and validationsets with a 80/10/10 split
def load_data(has_unique_data_file: bool, 
              type_datafiles: str, 
              cache_dir_datasets: str, 
              seed: int,
              whole_or_train_data_file: str, 
              validation_data_file: str,
              test_data_file: str):
    if has_unique_data_file:
        dataset_whole = load_dataset(type_datafiles, data_files=whole_or_train_data_file, cache_dir=cache_dir_datasets)
        # 85% train, 15% validation
        #train_valid = dataset_whole['train'].train_test_split(train_size=0.85, seed=seed)
        #train_valid_dataset = DatasetDict({
        #    'train': train_valid['train'],
        #    'valid': train_valid['test']
        #})
        # 80% train, 20% test + validation
        train_validtest = dataset_whole['train'].train_test_split(train_size=0.8, seed=seed)
        # Split the 20% test + valid in half test, half valid
        valid_test = train_validtest['test'].train_test_split(test_size=0.5, seed=seed)
        # gather everyone if you want to have a single DatasetDict
        train_valid_test_dataset = DatasetDict({
            'train': train_validtest['train'],
            'valid': valid_test['train'],
            'test': valid_test['test']
        })
    else:
        data_files = {
            "train": whole_or_train_data_file,
            "valid": validation_data_file,
            "test": test_data_file
        }
        train_valid_test_dataset = load_dataset(type_datafiles, data_files=data_files, cache_dir=cache_dir_datasets)

    return train_valid_test_dataset


# encode the sentences using the tokenizer
def tokenize_dataset(train_valid_test_dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
                     tokenizer,
                     config,
                     max_length,
                     cache_dir_tokenized_datasets):
    
    for dataset_name in ['train', 'valid', 'test']:
        train_valid_test_dataset[dataset_name] = train_valid_test_dataset[dataset_name].map(
            lambda examples: tokenize_examples(tokenizer, config, examples, max_length),
            batched=True,
            batch_size=2000,
            writer_batch_size=2000,
            remove_columns=["text"],
            #num_proc=4,
            load_from_cache_file=True,
            cache_file_name=f"{cache_dir_tokenized_datasets}{dataset_name}_pretrain_tokenized_cache.arrow"
        )
    
    return train_valid_test_dataset
    

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def test_predicts(tokenizer, decoder_start_token_id, trainer: Seq2SeqTrainer, test_dataset):
    test_outputs = trainer.predict(
        test_dataset=test_dataset,
        metric_key_prefix="test",
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=decoder_start_token_id
    ) # PredictionOutput
    
    labels = np.where(test_outputs.label_ids != -100, test_outputs.label_ids, tokenizer.pad_token_id)
    predictions = np.where(test_outputs.predictions != -100, test_outputs.predictions, tokenizer.pad_token_id)
    #logits = test_outputs.predictions.argmax(axis=-1)

    trainer.log_metrics("test", test_outputs.metrics)
    trainer.save_metrics("test", test_outputs.metrics)

    #print(f"\n\nPredictions: {predictions.shape}")
    #print(predictions)

    #print(f"\n\nTargets: {labels.shape}")
    #print(labels)

    predictions_decoded = tokenizer.batch_decode(
        predictions, skip_special_tokens=False #, clean_up_tokenization_spaces=True
    )
    list_predictions = lmap(str.strip, predictions_decoded)
    list_predictions = [s.replace('<pad>', '') for s in list_predictions]

    targets_decoded = tokenizer.batch_decode(
        labels, skip_special_tokens=False #, clean_up_tokenization_spaces=True
    )
    list_targets = lmap(str.strip, targets_decoded)
    list_targets = [s.replace('<pad>', '') for s in list_targets]

    # references_corpus_bleu = [[s.split(' ')] for s in list_targets]
    # candidate_corpus_bleu = [s.split(' ') for s in list_predictions]

    references_corpus_codebleu = [s for s in list_targets]
    candidate_corpus_codebleu = [s for s in list_predictions]

    bleu_metric = evaluate.load("bleu")

    #bleu = bleu_score(candidate_corpus_bleu, references_corpus_bleu)
    bleu = bleu_metric.compute(predictions=candidate_corpus_codebleu, references=references_corpus_codebleu)
    bleu_value = bleu["bleu"]

    codebleu = calc_codebleu(references_corpus_codebleu, candidate_corpus_codebleu, lang="java")
    codebleu_value = codebleu["codebleu"]

    print(f"\n\n\nMétricas BLEU:\nBleu = {bleu_value}\nCode_bleu = {codebleu_value}\n\n\n")

    print("\nSaving Predictions File...")
    output_predicts_path = outputPath + os.path.sep + "test_utg4java_pretrain_predicts.txt"
    save_in_file_list_lines(output_predicts_path, list_predictions, "Saving predictions in file")

    print("\nSaving Targets File...")
    output_targets_path = outputPath + os.path.sep + "test_utg4java_pretrain_targets.txt"
    save_in_file_list_lines(output_targets_path, list_targets, "Saving targets in file")

    """
    Only in linux (by code_bleu)
    metric_code_bleu = evaluate.load("dvitel/codebleu")
    metric_bleu = evaluate.load("bleu")
    #mt_metrics = evaluate.combine(["dvitel/codebleu"])
    results1 = metric_code_bleu.compute(predictions=list_predictions, references=list_targets, lang="java")
    results2 = metric_bleu.compute(predictions=list_predictions, references=list_targets)
    print("\n\n\nShowing other metrics:")
    print(results1)
    print()
    print(results2)
    print("\n\n")
    """

    return bleu, codebleu_value


# https://discuss.huggingface.co/t/what-does-evalprediction-predictions-contain-exactly/1691/3
# https://discuss.huggingface.co/t/how-to-accessing-the-input-ids-in-evalprediction-predictions-in-seq2seqtrainer/25372
# https://www.kaggle.com/code/alvations/huggingface-evaluate-for-mt-evaluations
# https://github.com/k4black/codebleu
def compute_metrics(eval_pred: EvalPrediction):
    print("\n\nCalculando metricas\n\n")
    #predictions, label_ids, inputs = eval_pred
    return {"test_metric_prueba": 128}


def generate_batch_splits(samples_idx: np.ndarray, batch_size: int, drop_last=True) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx


def group_texts(examples, expanded_inputs_length):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #print("\nShow concatenated_examples")
    #print(concatenated_examples)

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    print(f"\nShow total_length {total_length}")
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= expanded_inputs_length:
        total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
    
    print(f"\nShow new expanded_inputs_length {expanded_inputs_length}")
    print(f"\nShow new total_length {total_length}")
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
        for k, t in concatenated_examples.items()
    }
    return result


def example_data_collator_with_max_length(tokenizer, config, tokenized_datasets):
    max_seq_length = config.n_positions # max_position_embeddings
    mlm_probability = 0.15
    mean_noise_span_length = 3.0

    expanded_inputs_length, targets_length = compute_t5_input_and_target_lengths(
        inputs_length=max_seq_length,
        noise_density=mlm_probability,
        mean_noise_span_length=mean_noise_span_length,
    )

    tok_d = tokenized_datasets.map(
        lambda examples: group_texts(examples, expanded_inputs_length),
        batched=True
    )
    length_i = len(tok_d["train"][0]["input_ids"])
    print(f"\nLength inputs: {length_i}")
    print(tok_d)

    data_collator = DataCollatorForT5MLM(tokenizer=tokenizer,
                                         noise_density=mlm_probability,
                                         mean_noise_span_length=mean_noise_span_length,
                                         input_length=max_seq_length,
                                         target_length=targets_length,
                                         pad_token_id=config.pad_token_id,
                                         decoder_start_token_id=config.decoder_start_token_id)
    
    num_train_samples = len(tok_d["train"])
    print(f"\nLen train dataset: {num_train_samples}")
    train_samples_idx = np.random.permutation(np.arange(num_train_samples))
    train_batch_idx = generate_batch_splits(train_samples_idx, 1)

    for step, batch_idx in enumerate(tqdm(train_batch_idx, desc="Training...", position=1)):
        samples = [tok_d["train"][int(idx)] for idx in batch_idx]
        length = len(samples[0]["input_ids"])
        print(f"\nShow tokenized samples with lenght: {length}")
        print(samples)

        data_masked = data_collator(samples)
        print("\nShow masked samples")
        print(data_masked)
        if step == 0:
            break



if __name__ == "__main__":
    # Define device and accelerator
    if accelerator == "cpu":
        device = torch.device("cpu")
    elif accelerator == "gpu":
        # Si en el parámetro accelerator obtenemos como valor gpu, obtenemos el device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "cuda" not in str(device):
            # Si el device obtenido no es "cuda", quiere decir que es "cpu", por ende reasignamos el accelerator a "cpu"
            accelerator = "cpu"

    print("Cleaning memory...")
    report_gpu()
    show_cache()

    os.makedirs(outputPath, exist_ok=True)

    results = []

    print(f"Device: {device}  |  Accelerator: {accelerator}")

    #model_base = "Salesforce/codet5-small"
    model_base = "Salesforce/codet5p-220m"
    #model_base = "Salesforce/codet5p-770m"

    # Load tokenizer from existing one to re-use special tokens
    #tokenizer = RobertaTokenizer.from_pretrained(model_base, cache_dir=cacheDirHuggingFace)
    print("\nLoading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_base, cache_dir=cacheDirHuggingFace)
    tokenizer.model_max_length = 1024
    print(str(type(tokenizer)))
    
    print("\nLoading Configs...")
    config = T5Config.from_pretrained(model_base, cache_dir=cacheDirHuggingFace, vocab_size=len(tokenizer))

    # Load data
    print("\nLoading Data...")
    train_valid_test_dataset = load_data(has_unique_data_file=hasUniqueDataFile,
                                         type_datafiles="text",
                                         cache_dir_datasets=cacheDirDatasets,
                                         seed=SEED,
                                         whole_or_train_data_file=trainInput,
                                         validation_data_file=valInput,
                                         test_data_file=testInput)
    summary_loaded_dataset = str(train_valid_test_dataset)
    print(summary_loaded_dataset)
    #print(train_valid_dataset["train"][0])
    #print(train_valid_dataset["valid"][0])
    
    config.n_positions = tokenizer.model_max_length
    max_seq_length = config.n_positions # max_position_embeddings
    print("Max seq length: " + str(max_seq_length))

    mlm_probability = 0.15
    mean_noise_span_length = 3.0

    #print("\nMax_length: " + str(max_seq_length))
    #print("\nTokenizer Max_length: " + str(tokenizer.model_max_length)) 
    #print("\nModel config")
    #print(config)
    
    print("\nTokenizing Data...")
    tokenized_datasets = tokenize_dataset(train_valid_test_dataset=train_valid_test_dataset,
                                          tokenizer=tokenizer,
                                          config=config,
                                          max_length=max_seq_length,
                                          cache_dir_tokenized_datasets=cacheDirDatasetTokenized)
    summary_tokenized_dataset = str(tokenized_datasets)
    print(summary_tokenized_dataset)
    #isDataset = isinstance(tokenized_datasets["train"], Dataset)
    #print(f"\nIs instance of Dataset: {isDataset}") True

    print("\nShowing memory after load data...")
    show_cache()

    #print(tokenized_datasets["train"][0])
    #length_inputs = len(tokenized_datasets["train"][0]["input_ids"])
    
    
    #assert 128 == max_seq_length, "Return"
    

    # Load pretrained model
    print("\nLoading Model...")
    if doTrain:
        model = T5ForConditionalGeneration.from_pretrained(model_base, config=config, cache_dir=cacheDirHuggingFace)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
    else:
        assert inputModel is not None, "If doTrain is False you must indicate an input model"
        model = T5ForConditionalGeneration.from_pretrained(inputModel, cache_dir=cacheDirHuggingFace)
        #model.resize_token_embeddings(len(tokenizer))
        model.to(device)


    model_params_size = get_model_params_size(model)
    model_params_size_formatted = format_model_params_size(model_params_size)
    model_disk_size = get_model_disk_size(model)
    model_disk_size_formatted = format_model_disk_size(model_disk_size)

    print(f"\n==> Loaded model CodeT5:")
    print(f" - Model Params Size: {model_params_size}")
    print(f" - Model Params Size Formatted: {model_params_size_formatted}")
    print(f" - Model Disk Size: {model_disk_size}")
    print(f" - Model Disk Size Formatted: {model_disk_size_formatted}")

    output_configs_path = outputPath + os.path.sep + "codet5p_modelconfigs.txt"
    save_in_file_one_line(output_configs_path, str(model.config))

    print("\nShowing memory after load model...")
    show_cache()
    
    data_collator = DataCollatorWithDynamicLengthForT5MLM(
            tokenizer=tokenizer,
            noise_density=mlm_probability,
            mean_noise_span_length=mean_noise_span_length,
            max_length=max_seq_length,
            pad_token_id=config.pad_token_id,
            decoder_start_token_id=config.decoder_start_token_id
        )

    # CodeT5 para pretrain usa fp16, lr=2e-4, epochs=100
    # CodeT5 para fine-tuning usa lr=5e-5

    # CodeT5+ para pretrain (span denoising) usa fp16, lr=2e-4, steps=10.000 (batch_size=2048), weight_decay=0.1,
    # CodeT5+ para fine-tuning (Code summarization) usa lr=2e-5, epochs=10, batch_size=64, beam_size=5
    # CodeSearhNet (Java): train=164,923  val=10,955  inference=5,183

    learning_rate_pretraining = 2e-4
    #max_steps = 100
    gradient_accumulation_steps = gradientAccumuSteps
    weight_decay = 0.1
    lr_scheduler_type = "linear" # transformers.SchedulerType
    #save_steps = 100
    #eval_steps = 100

    effective_batch_size = batchSizeTrain * gradient_accumulation_steps
    num_train_steps = (len(tokenized_datasets["train"]) // effective_batch_size) * epochs

    warmup_ratio = lrWarmupRatio if lrWarmupRatio is not None else 0.05
    warmup_steps = lrWarmupSteps if lrWarmupSteps is not None else int(num_train_steps * warmup_ratio)

    print(f"num_train_steps={num_train_steps}, warmup_ratio={warmup_ratio}, warmup_steps={warmup_steps}")
    
    training_args = Seq2SeqTrainingArguments(
        use_cpu=accelerator == "cpu",
        output_dir=outputPath,
        overwrite_output_dir=True,
        seed=SEED,
        #data_seed=SEED,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        save_safetensors=False, # This avoid this warning: There were missing keys in the checkpoint model loaded: ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']

        # trainings parameters
        num_train_epochs=epochs,
        #max_steps=10,
        per_device_train_batch_size=batchSizeTrain,
        per_device_eval_batch_size=batchSizeValidation,

        # optimization parameters
        optim="adamw_torch",
        learning_rate=learning_rate_pretraining,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        fp16=useFp16,
        #fp16_full_eval=True, Más rápido, pero puede afectar las métricas

        # logging strategies
        disable_tqdm=False,
        log_level="info", # default: passive
        logging_dir=f"{outputPath}/logs",
        logging_strategy="steps",
        logging_steps=loggerFrequencySteps,
        #logging_nan_inf_filter=True,
        report_to=loggerListener, # wandb, tensorboard, etc
        run_name=runName, # used by wandb
        #skip_memory_metrics=False,

        # evaluation and saving strategies
        save_total_limit=1, # 2
        save_strategy="epoch", # steps, no
        #save_steps=save_steps,
        evaluation_strategy="epoch", # steps
        #eval_steps=eval_steps,
        load_best_model_at_end=True, # save_strategy and evaluation_strategy should be same
        metric_for_best_model="eval_loss",
        greater_is_better=False, # For loss then False, for accuracy then True
        #save_only_model=True, You can only load the model using `from_pretrained`
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        generation_max_length=max_seq_length
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        #compute_metrics=compute_metrics
        #callbacks=[]
    )
    trainer.create_model_card()
    
    
    # train the model
    if doTrain:
        print("\nInit Training...")
        start_time_train_and_val = time.time()
        train_results = trainer.train() # TrainOutput
        end_time_train_and_val = time.time()
        elapsed_time_train_and_val = end_time_train_and_val - start_time_train_and_val
        elapsed_time_train_and_val_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_train_and_val))

        #print(trainer.state)
        trainer.save_state()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)

        #eval_results = trainer.evaluate()
        #trainer.log_metrics("eval", eval_results)
        #trainer.save_metrics("eval", eval_results)

        results.append("SUMMARY MODEL:\n")
        results.append(f"Model Params Size: {model_params_size}")
        results.append(f"Model Params Size Formatted: {model_params_size_formatted}")
        results.append(f"Model Disk Size: {model_disk_size}")
        results.append(f"Model Disk Size Formatted: {model_disk_size_formatted}")
        results.append("\n")

        results.append("TRAINING AND VALIDATION RESULTS:\n")
        results.append(f"Training batch size: {batchSizeTrain}")
        results.append(f"Validation batch size: {batchSizeValidation}")
        results.append(f"Total expected epochs: {epochs}")
        results.append(f"Total expected trainig steps: {trainer.state.max_steps}")
        results.append(f"Total expected trainig steps 2: {num_train_steps}")
        results.append(f"Total trained epochs: {trainer.state.epoch}")
        results.append(f"Total trained steps: {trainer.state.global_step}")
        results.append(f"Elapsed time: {elapsed_time_train_and_val} seconds")
        results.append(f"Elapsed time (formatted): {elapsed_time_train_and_val_formatted}")
        results.append(f"Total flos: {trainer.state.total_flos}")
        results.append(f"Total flos (formatted): {'{:e}'.format(trainer.state.total_flos)}")
        #results.append(f"Training iterations by second: {format(train_itr_by_second, '.2f')} it/s")
        #results.append(f"Validation iterations by second: {format(val_itr_by_second, '.2f')} it/s")
        results.append(f"Best epoch val_loss: {trainer.state.best_metric}")
        results.append(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
        #results.append(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        results.append("\n\n")

        results.append("SUMMARY DATASETS:\n")
        results.append("Loaded Dataset:")
        results.append(str(summary_loaded_dataset))
        results.append("")
        results.append("Tokenized Dataset:")
        results.append(str(summary_tokenized_dataset))
        results.append("")

        output_results_path = outputPath + os.path.sep + "utg4java_pretrain_results.txt"
        print("\nSaving Results File...")
        save_in_file_list_lines(output_results_path, results, "Saving results in file")

        output_log_history_path = outputPath + os.path.sep + "utg4java_pretrain_log_history.txt"
        print("Saving Log History...")
        save_list_dicts_in_file_as_a_json(output_log_history_path, trainer.state.log_history, "Saving log history in file")

        print("Saving Model...")
        trainer.save_model(training_args.output_dir)
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        # This is already saved with trainer.save_model()
        #tokenizer.save_pretrained(training_args.output_dir)

    if doTest:
        bleu_metrics = []
        print("\nInit Testing Predictions...")
        start_time_test = time.time()
        partial_test_dataset = tokenized_datasets["test"]
        #partial_test_dataset = tokenized_datasets["test"].select(range(100))
        bleu, codebleu = test_predicts(tokenizer=tokenizer, decoder_start_token_id=config.decoder_start_token_id, trainer=trainer, test_dataset=partial_test_dataset)
        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test
        elapsed_time_test_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_test))

        bleu_metrics.append("TESTING RESULTS:\n")
        bleu_metrics.append(f"Elapsed time: {elapsed_time_test} seconds")
        bleu_metrics.append(f"Elapsed time (formatted): {elapsed_time_test_formatted}")
        bleu_metrics.append(f"BLEU: {bleu}")
        bleu_metrics.append(f"CodeBLEU: {codebleu}")

        output_bleu_metrics = outputPath + os.path.sep + "utg4java_pretrain_bleu_metrics.txt"
        print("\nSaving Bleu Metrics Results File...")
        save_in_file_list_lines(output_bleu_metrics, bleu_metrics, "Saving bleu metrics results in file")

    print("\nCleaning memory after finish...")
    show_cache()
    report_gpu()
    show_cache()


    
        

    