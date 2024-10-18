from argparse import ArgumentParser
from distutils.util import strtobool
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Config, Seq2SeqTrainer, Seq2SeqTrainingArguments, DefaultDataCollator, DataCollatorForSeq2Seq
#from transformers import SchedulerType, DataCollatorForLanguageModeling, DataCollatorForPermutationLanguageModeling, DataCollatorForSeq2Seq, DataCollatorWithPadding
#from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List, Union, Callable, Iterable
import evaluate
import datetime
import gc
import json
import numpy as np
import os
import time
import torch
#from torchtext.data.metrics import bleu_score
from codebleu import calc_codebleu
from parser_utils import ParserUtils

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

parser = ArgumentParser()
parser.add_argument("-m", "--tagModel", dest="tagModel", help="Model base to be used")
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
parser.add_argument("-sl", "--sourceLabel", dest="sourceLabel", default="src_fm_fc_dctx_priv", help="Source Label for the input.csv file")
parser.add_argument("-o", "--modelOutputDir", dest="outputPath", help="Output Directory Path for the model")
parser.add_argument("-logfr", "--loggerFrequencySteps", dest="loggerFrequencySteps", type=int, help="Frequency of how many N steps the results should be logged")
parser.add_argument("-warmup", "--lrWarmupSteps", dest="lrWarmupSteps", type=int, help="Warmup steps to update the learning rate")
parser.add_argument("-warmupRatio", "--lrWarmupRatio", dest="lrWarmupRatio", type=float, help="Warmup ratio to update the learning rate")
parser.add_argument("-fp16", "--useFp16", dest="useFp16", default=True, type=lambda x: bool(strtobool(x)), help="Indicates whether uses fp16 mixed precision")
parser.add_argument("-bf16", "--useBf16", dest="useBf16", default=False, type=lambda x: bool(strtobool(x)), help="Indicates whether uses bf16 mixed precision")
parser.add_argument("-logsys", "--loggerListener", dest="loggerListener", help="System logging listener. Options: tensorboard. wandb")
parser.add_argument("-rn", "--runName", dest="runName", help="Run name of Execution (Used by WanDB)")
parser.add_argument("-doTrain", "--doTrain", dest="doTrain", default=True, type=lambda x: bool(strtobool(x)), help="Indicates whether performs training")
parser.add_argument("-doTest", "--doTest", dest="doTest", default=True, type=lambda x: bool(strtobool(x)), help="Indicates whether performs testing")
parser.add_argument("-inm", "--inputModel", dest="inputModel", help="Indicates the path of the input model that has been previously trained. If this attribute is passed and doTrain is False, no training is performed and only validation is performed on the test data set")
parser.add_argument("-genStrategy", "--generationStrategy", default="greddy", dest="generationStrategy", help="Indicates the the strategy to generate outputs model")
parser.add_argument("-outSamples", "--outputSamples", dest="outputSamples", default=5, type=int, help="Number of the outputs to generate")
parser.add_argument("-numBeams", "--numBeamsPredictions", dest="numBeamsPredictions", default=5, type=int, help="Number of beams to generate (Apply only on beam_search)")
parser.add_argument("-fsnw", "--firstSampleNotWrong", dest="firstSampleNotWrong", default=True, type=lambda x: bool(strtobool(x)), help="Indicates if only returns first sample without errors")


args = parser.parse_args()
tagModel = args.tagModel
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
sourceLabel = args.sourceLabel
loggerFrequencySteps = args.loggerFrequencySteps
lrWarmupSteps = args.lrWarmupSteps
lrWarmupRatio = args.lrWarmupRatio
useFp16 = args.useFp16
useBf16 = args.useBf16
loggerListener = args.loggerListener
runName = args.runName
doTrain = args.doTrain
doTest = args.doTest
inputModel = args.inputModel
# https://huggingface.co/blog/how-to-generate
generationStrategy = args.generationStrategy
numReturnPredictions = args.outputSamples
numBeamsPredictions = args.numBeamsPredictions
firstSampleNotWrong = args.firstSampleNotWrong

NUM_CPUS = os.cpu_count() // 2
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

def save_in_file_list_lines(file_path: str, list_lines: list[str], msg_tqdm: str):
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

def save_list_dicts_in_file_as_a_json(file_path: str, list_dicts: list[dict], msg_tqdm: str):
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


# Utilities to get model parameters total
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


# Utilities to get model size in disk
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


def secs2timedelta(secs):
    """
    convert seconds to hh:mm:ss.msec, msecs rounded to 2 decimals
    """
    msec = int(abs(secs - int(secs)) * 100)
    return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"


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
                     max_length,
                     cache_dir_tokenized_datasets):
    
    for dataset_name in ['train', 'valid', 'test']:
        train_valid_test_dataset[dataset_name] = train_valid_test_dataset[dataset_name].map(
            lambda examples: tokenize_examples(tokenizer, dataset_name, examples, max_length),
            batched=True,
            batch_size=2000,
            writer_batch_size=2000,
            remove_columns=["target", "src_fm_fc_ms_ff", "src_fm_fc_dctx", "src_fm_fc_dctx_priv", "imports_focal_class", "imports_test_class"],
            #num_proc=4,
            load_from_cache_file=True,
            cache_file_name=f"{cache_dir_tokenized_datasets}{dataset_name}_tokenized_cache.arrow"
        )
    
    #train_valid_test_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    return train_valid_test_dataset


# Preprocess inputs using tokenizer. Solo tokenizamos, aún no retornamos tensores
def tokenize_examples(tokenizer, dataset_name, examples, max_seq_length):
    # encode the source-target pairs
    source = examples[sourceLabel]
    target = examples["target"]

    #print("\nLength Input: " + str(len(source)) + "\n")

    #inputs = ["generate Unit Test in Java: " + input_source for input_source in source]
    inputs = [input_source for input_source in source]
    # encode de inputs
    model_inputs = tokenizer(inputs, max_length=max_seq_length, padding="max_length", truncation=True)
    # With this encodes inputs and target in one sentence:
    #model_inputs = self.tokenizer(inputs, text_target=target, max_length=max_input_length, padding="max_length", truncation=True)

    # encode the expected outputs
    #expected_outputs = tokenizer(target, max_length=max_seq_length, padding="max_length", truncation=True).input_ids
    labels = tokenizer(target, max_length=max_seq_length, padding="max_length", truncation=True)
    
    model_inputs["labels"] = labels["input_ids"].copy()
    model_inputs["labels"] = [
        [(lab if lab != tokenizer.pad_token_id else -100) for lab in label] for label in model_inputs["labels"]
    ]

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    # https://huggingface.co/docs/transformers/main/en/model_doc/t5#training
    # expected_outputs[expected_outputs == self.tokenizer.pad_token_id] = -100
    #expected_outputs_with_ignore_index = []
    #for output_example in expected_outputs:
        # si algún token del target es igual a 0, lo reemplazamos por -100
        #output_example[output_example == tokenizer.pad_token_id] = -100
    #    output_example = [output if output != 0 else -100 for output in output_example]
    #    expected_outputs_with_ignore_index.append(output_example)

    #model_inputs["labels"] = expected_outputs_with_ignore_index

    # {input_ids: [[], []...], attention_mask: [[], []...], labels: [[], []...]}
    return model_inputs


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


# https://huggingface.co/blog/how-to-generate
# https://aclanthology.org/2023.emnlp-industry.73.pdf
# https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L119
def generate_predictions(tokenizer, decoder_start_token_id, trainer: Seq2SeqTrainer, test_dataset: Dataset, num_return_sequences: int):
    if generationStrategy == "greddy_search":
        test_outputs = trainer.predict(
            test_dataset=test_dataset, 
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id = decoder_start_token_id,
            #max_length=1024,
            metric_key_prefix="test"
        ) # PredictionOutput
    
    elif generationStrategy == "beam_search":
        test_outputs = trainer.predict(
            test_dataset=test_dataset, 
            metric_key_prefix="test", 
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id = decoder_start_token_id,
            do_sample=False, 
            num_beams=numBeamsPredictions,
            #no_repeat_ngram_size=2,
            early_stopping=True,
            num_return_sequences=num_return_sequences
        )

    elif generationStrategy == "sampling":
        test_outputs = trainer.predict(
            test_dataset=test_dataset, 
            metric_key_prefix="test", 
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id = decoder_start_token_id,
            do_sample=True, 
            top_k=0,
            temperature=0.8
            #no_repeat_ngram_size=2,
            #early_stopping=True,
            #num_return_sequences=num_return_sequences
        )
    
    elif generationStrategy == "top_k_sampling":
        test_outputs = trainer.predict(
            test_dataset=test_dataset, 
            metric_key_prefix="test", 
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id = decoder_start_token_id,
            do_sample=True, 
            top_k=50,
            temperature=0.6,
            #no_repeat_ngram_size=2,
            #early_stopping=True,
            num_return_sequences=num_return_sequences
        )
    
    elif generationStrategy == "top_p_sampling":
        test_outputs = trainer.predict(
            test_dataset=test_dataset, 
            metric_key_prefix="test", 
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id = decoder_start_token_id,
            do_sample=True, 
            top_p=0.95,
            #top_k=100, # 10, 50, 100
            temperature=0.8, # 0.4, 0.6, 0.8
            num_return_sequences=num_return_sequences
        )

    return test_outputs


def calculate_bleu_metrics(index: int, list_predictions: list, list_targets: list):
    #references_corpus_bleu = [[s.split(' ')] for s in list_targets]
    #candidate_corpus_bleu = [s.split(' ') for s in list_predictions]

    references_corpus_codebleu = [s for s in list_targets]
    candidate_corpus_codebleu = [s for s in list_predictions]

    bleu_metric = evaluate.load("bleu")

    #bleu = bleu_score(candidate_corpus_bleu, references_corpus_bleu)
    bleu = bleu_metric.compute(predictions=candidate_corpus_codebleu, references=references_corpus_codebleu)
    #print("BLEU results")
    #print(bleu)
    #print()
    bleu_value = bleu["bleu"]
    codebleu = calc_codebleu(references_corpus_codebleu, candidate_corpus_codebleu, lang="java")
    codebleu_value = codebleu["codebleu"]

    print(f"\n\n\nMétricas BLEU Nro {index}:\nBleu = {bleu_value}\nCode_bleu = {codebleu_value}\n\n\n")

    print(f"\nSaving Predictions File Nro {index}...")
    output_predicts_path = outputPath + os.path.sep + "test_" + tagModel + "_finetuning_predicts_" + str(index) + ".txt"
    save_in_file_list_lines(output_predicts_path, list_predictions, "Saving predictions in file Nro " + str(index))

    # Aqui no guardamos los targest, ya que estos no cambian y se guardan desde donde se llama este método
    # print("\nSaving Targets File...")
    # output_targets_path = outputPath + os.path.sep + "test_" + tagModel + " _finetuning_targets.txt"
    # save_in_file_list_lines(output_targets_path, list_targets, "Saving targets in file")

    return bleu_value, codebleu_value


def test_predicts(tokenizer, decoder_start_token_id, trainer: Seq2SeqTrainer, test_dataset: Dataset, num_return_sequences: int):
    #print("\n\n")
    #print("test_dataset")
    #print(test_dataset)
    #print("\n\n")

    #test_outputs = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test", num_beams=num_beans_predictions) # PredictionOutput
    test_outputs = generate_predictions(
        tokenizer=tokenizer,
        decoder_start_token_id=decoder_start_token_id,
        trainer=trainer,
        test_dataset=test_dataset, 
        num_return_sequences=num_return_sequences
    ) # PredictionOutput
    #print("\n\n")
    #print("test_outputs")
    #print(test_outputs)
    #print("\n\n")
    labels = np.where(test_outputs.label_ids != -100, test_outputs.label_ids, tokenizer.pad_token_id)
    predictions = np.where(test_outputs.predictions != -100, test_outputs.predictions, tokenizer.pad_token_id)
    #logits = test_outputs.predictions.argmax(axis=-1)

    print(f"\n\nPredictions: {predictions.shape}")
    #print(predictions)

    print(f"\n\nTargets: {labels.shape}")
    #print(labels)

    predictions_decoded = tokenizer.batch_decode(
        predictions, skip_special_tokens=True #, clean_up_tokenization_spaces=True
    )
    print(f"\n\nPredictions decoded: {len(predictions_decoded)}")
    #print(predictions_decoded)

    list_predictions = lmap(str.strip, predictions_decoded)
    #list_predictions = [s.replace('<pad>', '') for s in list_predictions]

    print(f"\n\nPredictions decoded formatted: {len(list_predictions)}")
    #for pred in list_predictions:
    #    print(pred)

    targets_decoded = tokenizer.batch_decode(
        labels, skip_special_tokens=True #, clean_up_tokenization_spaces=True
    )
    list_targets = lmap(str.strip, targets_decoded)
    #list_targets = [s.replace('<pad>', '') for s in list_targets]
    print(f"\n\nTargets decoded formatted: {len(list_targets)}")
    #for targ in list_targets:
    #    print(targ)

    predictions_with_error_in_all_beams = None
    corrects_by_pos_before_changes = None
    corrects_by_pos_after_changes = None

    if firstSampleNotWrong:
        parserUtils = ParserUtils("utf-8")

        new_list_predictions = []
        position = 0

        predictions_with_error_in_all_beams = 0
        corrects_by_pos_before_changes = []
        corrects_by_pos_after_changes = []
        for i in range(num_return_sequences):
            corrects_by_pos_before_changes.append([i, 0])
            corrects_by_pos_after_changes.append([i, 0])

        for i in range(0, len(list_predictions), num_return_sequences):
            first_predict_by_target = list_predictions[i]
            some_prediction_is_correct = False

            pos_beam = 0
            for j in range(i, i + num_return_sequences, 1):
                code_before_changes = list_predictions[j]
                code_before_changes_has_errors, code_without_missings = parserUtils.fix_missings_in_code(code_before_changes)
                code_after_changes_has_errors = parserUtils.validate_if_code_has_errors(code_without_missings)
                #print(f"Prediction ({str(position)}, {str(j)}) has error -> Before changes: {str(code_before_changes_has_errors)} - After changes: {str(code_after_changes_has_errors)}")

                if code_before_changes_has_errors == False:
                    #print("Add prediction without error (before changes): " + str(j))
                    corrects_by_pos_before_changes[pos_beam][1] = corrects_by_pos_before_changes[pos_beam][1] + 1
                    new_list_predictions.append(code_before_changes)
                    some_prediction_is_correct = True
                    break

                if code_after_changes_has_errors == False:
                    #print("Add prediction without error (before changes): " + str(j))
                    corrects_by_pos_after_changes[pos_beam][1] = corrects_by_pos_after_changes[pos_beam][1] + 1
                    new_list_predictions.append(code_without_missings)
                    some_prediction_is_correct = True
                    break
                
                pos_beam += 1
            
            if not some_prediction_is_correct:
                #print(f"Add first prediction by target: ({str(position)}, {str(i)})")
                new_list_predictions.append(first_predict_by_target)
                predictions_with_error_in_all_beams += 1
            
            position += 1
        
        list_predictions = new_list_predictions

        print(f"\n\nlist_predictions with fixed errors: {len(list_predictions)}")
        print(f"\ncorrects_by_pos_before_changes: {str(corrects_by_pos_before_changes)}")
        print(f"\ncorrects_by_pos_after_changes: {str(corrects_by_pos_after_changes)}")
        print(f"\npredictions_with_error_in_all_beams: {str(predictions_with_error_in_all_beams)}")

    segment_length = len(list_targets)
    count_segments = len(list_predictions) // segment_length
    #segmented_predictions = []
    #for i in range(count_segments):
    #    segment = []
    #    for j in range(0, (seg_length)):
    #        segment.append(list_predicts[i + (j * count_segments)])
    #    segmented_predictions.append(segment)
    segmented_predictions = [
        [list_predictions[i + (j * count_segments)] for j in range(segment_length)]
        for i in range(count_segments)
    ]

    print(f"\n\nSegments predictions: {len(segmented_predictions)}")

    list_bleu = []
    list_codebleu = []

    index = 1
    for list_segment_predicts in segmented_predictions:
        #print(list_segment_predicts)
        bleu, codebleu_value = calculate_bleu_metrics(index, list_segment_predicts, list_targets)
        test_outputs.metrics.update(
            {
                "test_bleu_" + str(index): bleu,
                "test_codebleu_ " + str(index): codebleu_value
            }
        )
        list_bleu.append(bleu)
        list_codebleu.append(codebleu_value)
        index += 1

    trainer.log_metrics("test", test_outputs.metrics)
    trainer.save_metrics("test", test_outputs.metrics)
    
    print("\nSaving Targets File...")
    output_targets_path = outputPath + os.path.sep + "test_" + tagModel + "_finetuning_targets.txt"
    save_in_file_list_lines(output_targets_path, list_targets, "Saving targets in file")

    return list_bleu, list_codebleu, corrects_by_pos_before_changes, corrects_by_pos_after_changes, predictions_with_error_in_all_beams


if __name__ == '__main__':
    start_total_time = time.time()
    # Define device and accelerator
    # https://gist.github.com/ronaldseoh/da4afaa1bb9eb34d32d167ba417a5199
    if accelerator == "cpu":
        device = torch.device("cpu")
    elif accelerator == "gpu":
        # Si en el parámetro accelerator obtenemos como valor gpu, obtenemos el device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "cuda" not in str(device):
            # Si el device obtenido no es "cuda", quiere decir que es "cpu", por ende reasignamos el accelerator a "cpu"
            accelerator = "cpu"
    else:
        raise "Accelerator is wrong"
    # elif accelerator == "colab_tpu":
    #     try:
    #         # assert os.environ['COLAB_TPU_ADDR']
    #         # https://github.com/pytorch/xla/blob/master/contrib/colab/getting-started.ipynb
    #         #import torch_xla
    #         #import torch_xla.core.xla_model as xm

    #         #device = xm.xla_device()
    #         pass
    #     except:
    #         accelerator = "cpu"
    #         device = torch.device("cpu")


    print("Cleaning memory...")
    report_gpu()
    show_cache()

    os.makedirs(outputPath, exist_ok=True)

    results = []

    print(f"Device: {device}  |  Accelerator: {accelerator}")

    if useFp16 and useBf16:
        raise "Debe indicar un solo tipo de precision: fp16 o bf16"
    
    half_precision_backend = "auto" # por defecto
    if useBf16:
        half_precision_backend = "cpu_amp" # bf16 solo se puede ejecutar sobre CPU o sobre NVIDIA con arquitectura AMP o superior

    print(f"Precision: useFp16={useFp16}, useBf16={useBf16}  |  Half Precision Backend: {half_precision_backend}")


    if tagModel == "utg4java":
        #model_base = "eljavatar/pretrain_utg4java_codet5_prueba"
        model_base = "eljavatar/pretrain_utg4java_codet5p" # 220m seq 512
        #model_base = "eljavatar/pretrain_utg4java_220m_seq1024"
    elif tagModel == "codet5p":
        #model_base = "Salesforce/codet5-small"
        model_base = "Salesforce/codet5p-220m"
        #model_base = "Salesforce/codet5p-770m"
    else:
        raise "Tag de Model_base no permitido"

    # Load tokenizer from existing one to re-use special tokens
    #tokenizer = RobertaTokenizer.from_pretrained(model_base, cache_dir=cacheDirHuggingFace)
    print("\nLoading Tokenizer...")
    special_tokens = ["<FCTX>", "</FCTX>", "<ECTX>", "</ECTX>", "<PRIVATE_FCTX>", "</PRIVATE_FCTX>"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_base, 
        cache_dir=cacheDirHuggingFace, 
        additional_special_tokens=special_tokens
    )
    tokenizer.model_max_length = 1024
    print("Type tokenizer: " + str(type(tokenizer)))
    print("Length tokenizer: " + str(len(tokenizer)))


    print("\nLoading Configs...")
    #config = T5Config.from_pretrained(model_base, cache_dir=cacheDirHuggingFace, vocab_size=len(tokenizer))
    # Load config from model without change the param vocab_size
    # Initial config vocab_size: 32100
    # Config with aditional special tokens: 32106
    config = T5Config.from_pretrained(model_base, cache_dir=cacheDirHuggingFace)


    # Load data
    print("\nLoading Data...")
    train_valid_test_dataset = load_data(has_unique_data_file=hasUniqueDataFile,
                                         type_datafiles="csv",
                                         cache_dir_datasets=cacheDirDatasets,
                                         seed=SEED,
                                         whole_or_train_data_file=trainInput,
                                         validation_data_file=valInput,
                                         test_data_file=testInput)
    summary_loaded_dataset = str(train_valid_test_dataset)
    print(summary_loaded_dataset)
    print()
    #print(train_valid_test_dataset["train"][0])
    #print(train_valid_test_dataset["valid"][0])

    config.n_positions = tokenizer.model_max_length
    max_seq_length = config.n_positions # max_position_embeddings
    print("Max seq length: " + str(max_seq_length))

    tokenized_datasets = tokenize_dataset(train_valid_test_dataset=train_valid_test_dataset,
                                          tokenizer=tokenizer,
                                          max_length=max_seq_length,
                                          cache_dir_tokenized_datasets=cacheDirDatasetTokenized)
    summary_tokenized_dataset = str(tokenized_datasets)
    print(summary_tokenized_dataset)
    print()
    #print(tokenized_datasets["train"][0])
    #print(tokenized_datasets["valid"][0])

    #print(f"EOS Token id: {str(tokenizer.eos_token_id)} -> {str(tokenizer.eos_token)}")
    #test_input_decoded = tokenizer.decode(
    #    tokenized_datasets["train"][0]["input_ids"], skip_special_tokens=False #, clean_up_tokenization_spaces=True
    #)
    #print()
    #print(test_input_decoded.replace('<pad>', ''))
    #print()

    #assert 128 == max_seq_length, "Return"

    print("\nShowing memory after load data...")
    show_cache()


    # Load pretrained model
    print("\nLoading Model...")
    if doTrain:
        model = T5ForConditionalGeneration.from_pretrained(model_base, config=config, cache_dir=cacheDirHuggingFace)
        #model = AutoModelForSeq2SeqLM.from_pretrained(model_base, config=config, cache_dir=cacheDirHuggingFace)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
    else:
        assert inputModel is not None, "If doTrain is False you must indicate an input model"
        model = T5ForConditionalGeneration.from_pretrained(inputModel, cache_dir=cacheDirHuggingFace)
        #model = AutoModelForSeq2SeqLM.from_pretrained(inputModel, cache_dir=cacheDirHuggingFace)
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

    output_configs_path = outputPath + os.path.sep + tagModel + "_finetuning_modelconfigs.txt"
    save_in_file_one_line(output_configs_path, str(model.config))

    print("\nShowing memory after load model...")
    show_cache()


    #assert 128 == max_seq_length, "Return"

    # Solamente necesitamos que el DataCollator convierta los inputs a tensores
    data_collator = DefaultDataCollator(
        return_tensors="pt"
    )

    # CodeT5 para fine-tuning usa lr=5e-5
    # CodeT5+ para fine-tuning (Code summarization) usa lr=2e-5, epochs=10, batch_size=64, beam_size=5

    learning_rate_finetuning = 2e-5
    #learning_rate_finetuning = 1e-5
    gradient_accumulation_steps = gradientAccumuSteps
    weight_decay = 0.05 # This is the recommended in source_code in CodeT5+
    #weight_decay = 0.001
    lr_scheduler_type = "linear" # transformers.SchedulerType

    effective_batch_size = batchSizeTrain * gradient_accumulation_steps
    num_train_steps = (len(tokenized_datasets["train"]) // effective_batch_size) * epochs

    warmup_ratio = lrWarmupRatio if lrWarmupRatio is not None else 0.05
    warmup_steps = lrWarmupSteps if lrWarmupSteps is not None else int(num_train_steps * warmup_ratio)

    print(f"\nNum_train_steps={num_train_steps}, warmup_ratio={warmup_ratio}, warmup_steps={warmup_steps}\n")

    # https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/tune_codet5p_seq2seq.py
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
        #max_steps=50,
        per_device_train_batch_size=batchSizeTrain,
        per_device_eval_batch_size=batchSizeValidation,

        # optimization parameters
        optim="adamw_torch",
        learning_rate=learning_rate_finetuning,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        fp16=useFp16,
        bf16=useBf16,
        half_precision_backend=half_precision_backend,
        #fp16_full_eval=True, Más rápido, pero puede afectar las métricas
        #bf16_full_eval=True, Estas opciones lo que hacen es setear el tipo de dato del modelo en precisión mixta

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
        load_best_model_at_end=False, # save_strategy and evaluation_strategy should be same
        metric_for_best_model="eval_loss",
        greater_is_better=False, # For loss then False, for accuracy then True
        #save_only_model=True, You can only load the model using `from_pretrained`
        #eval_accumulation_steps=args.eval_acc_steps, # set this lower, if testing or validation crashes
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        generation_max_length=max_seq_length,
        #generation_num_beams=numBeansPredictions
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
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.create_model_card()


    # train the model
    if doTrain:
        print("\nInit Training...")
        start_time_train_and_val = time.time()
        train_results = trainer.train() # TrainOutput
        end_time_train_and_val = time.time()
        elapsed_time_train_and_val = end_time_train_and_val - start_time_train_and_val
        #elapsed_time_train_and_val_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_train_and_val))
        elapsed_time_train_and_val_formatted = secs2timedelta(elapsed_time_train_and_val)

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
        results.append(f"Total expected trainig steps (calculated): {num_train_steps}")
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

        output_results_path = outputPath + os.path.sep + tagModel + "_finetuning_train_results.txt"
        print("\nSaving Results File...")
        save_in_file_list_lines(output_results_path, results, "Saving results in file")

        output_log_history_path = outputPath + os.path.sep + tagModel + "_finetuning_log_history.txt"
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
        #partial_test_dataset = tokenized_datasets["test"].select(range(10))
        #partial_test_dataset = tokenized_datasets["test"].select(range(128))
        partial_test_dataset = tokenized_datasets["test"].select(range(1024))
        #partial_test_dataset = tokenized_datasets["test"].select(range(3_200))
        #partial_test_dataset = tokenized_datasets["test"]
        list_bleu, list_codebleu, corrects_by_pos_before_changes, corrects_by_pos_after_changes, predictions_with_error_in_all_beams = test_predicts(
            tokenizer=tokenizer, 
            decoder_start_token_id=config.decoder_start_token_id, 
            trainer=trainer, 
            test_dataset=partial_test_dataset, 
            num_return_sequences=numReturnPredictions
        )
        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test
        #elapsed_time_test_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_test))
        elapsed_time_test_formatted = secs2timedelta(elapsed_time_test)

        bleu_metrics.append("TESTING RESULTS:\n")
        bleu_metrics.append(f"Elapsed time: {elapsed_time_test} seconds")
        bleu_metrics.append(f"Elapsed time (formatted): {elapsed_time_test_formatted}")

        for i in range(len(list_bleu)):
            bleu_metrics.append("========================================")
            bleu_metrics.append(f"BLEU_{i}:     {list_bleu[i]}")
            bleu_metrics.append(f"CodeBLEU_{i}: {list_codebleu[i]}")

        bleu_metrics.append("========================================")
        
        bleu_metrics.append(f"\ncorrects_by_pos_beam_before_changes: {str(corrects_by_pos_before_changes)}")
        bleu_metrics.append(f"corrects_by_pos_beam_after_changes: {str(corrects_by_pos_after_changes)}")
        bleu_metrics.append(f"predictions_with_error_in_all_beams: {str(predictions_with_error_in_all_beams)}")

        output_bleu_metrics = outputPath + os.path.sep + tagModel + "_finetuning_bleu_metrics.txt"
        print("\nSaving Bleu Metrics Results File...")
        save_in_file_list_lines(output_bleu_metrics, bleu_metrics, "Saving bleu metrics results in file")


    end_total_time = time.time()
    elapsed_total_time = end_total_time - start_total_time
    elapsed_total_time_formatted = secs2timedelta(elapsed_total_time)
    print("\n\nFINISH ALL")
    print(f"Elapsed time: {elapsed_total_time} seconds")
    print(f"Elapsed time (formatted): {elapsed_total_time_formatted}")

    print("\nCleaning memory after finish...")
    show_cache()
    report_gpu()
    show_cache()
