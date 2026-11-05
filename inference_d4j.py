from argparse import ArgumentParser
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from distutils.util import strtobool
from parser_utils import ParserUtils
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config, PLBartTokenizer, PLBartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DefaultDataCollator, DataCollatorForSeq2Seq, EvalPrediction, PreTrainedModel
# from transformers.trainer_pt_utils import DistributedTensorGatherer
from tqdm import tqdm
from typing import List, Union, Callable, Iterable
import csv
import datetime
import gc
import math
import numpy as np
import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

parser = ArgumentParser()
parser.add_argument("-cdhf", "--cacheDirHuggingFace", dest="cacheDirHuggingFace", help="Directory where downloaded hugginface models are saved")
parser.add_argument("-cdds", "--cacheDirDatasets", dest="cacheDirDatasets", help="Directory where loaded datasets are saved")
parser.add_argument("-cddst", "--cacheDirDatasetTokenized", dest="cacheDirDatasetTokenized", help="Directory where tokenized loaded datasets are saved")
parser.add_argument("-a", "--accelerator", dest="accelerator", default="gpu", help="Device type to train the model. Options: cpu, gpu")
parser.add_argument("-bsv", "--batchSizeValidation", dest="batchSizeValidation", default=16, type=int, help="Number of samples per batch that will be loaded in validation dataset")
parser.add_argument("-sl", "--sourceLabel", dest="sourceLabel", default="src_fm_fc_dctx_priv", help="Source Label for the input.csv file")
parser.add_argument("-o", "--modelOutputDir", dest="outputPath", help="Output Directory Path for the model")
#parser.add_argument("-logfr", "--loggerFrequencySteps", dest="loggerFrequencySteps", type=int, help="Frequency of how many N steps the results should be logged")
parser.add_argument("-fp16", "--useFp16", dest="useFp16", default=True, type=lambda x: bool(strtobool(x)), help="Indicates whether uses fp16 mixed precision")
parser.add_argument("-bf16", "--useBf16", dest="useBf16", default=False, type=lambda x: bool(strtobool(x)), help="Indicates whether uses bf16 mixed precision")
parser.add_argument("-inm", "--inputModel", dest="inputModel", help="Indicates the path of the input model that has been previously trained. If this attribute is passed and doTrain is False, no training is performed and only validation is performed on the test data set")
parser.add_argument("-genStrategy", "--generationStrategy", default="greddy", dest="generationStrategy", help="Indicates the the strategy to generate outputs model")
parser.add_argument("-outSamples", "--outputSamples", dest="outputSamples", default=5, type=int, help="Number of the outputs to generate")
parser.add_argument("-numBeams", "--numBeamsPredictions", dest="numBeamsPredictions", default=5, type=int, help="Number of beams to generate (Apply only on beam_search)")
parser.add_argument("-fsnw", "--firstSampleNotWrong", dest="firstSampleNotWrong", default=True, type=lambda x: bool(strtobool(x)), help="Indicates if only returns first sample without errors")
parser.add_argument("-d4jIn", "--defects4jInputs", dest="defects4jInputs", help="Indicates the path where the defects4j datasets are located")


args = parser.parse_args()
cacheDirHuggingFace = args.cacheDirHuggingFace
cacheDirDatasets = args.cacheDirDatasets
cacheDirDatasetTokenized = args.cacheDirDatasetTokenized
accelerator = args.accelerator
outputPath = args.outputPath
batchSizeValidation = args.batchSizeValidation
sourceLabel = args.sourceLabel
#loggerFrequencySteps = args.loggerFrequencySteps
useFp16 = args.useFp16
useBf16 = args.useBf16
inputModel = args.inputModel
# # https://huggingface.co/blog/how-to-generate
generationStrategy = args.generationStrategy
numReturnPredictions = args.outputSamples
numBeamsPredictions = args.numBeamsPredictions
firstSampleNotWrong = args.firstSampleNotWrong
defects4jInputs = args.defects4jInputs


NUM_CPUS = os.cpu_count() // 2
SEED = 42

DF4_PROJECTS = ['Closure', 'Cli', 'Codec', 'Collections', 'Compress', 'Csv', 'JxPath', 'Lang', 'Math', 'Gson', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Chart', 'Time', 'Jsoup', 'Mockito']
#DF4_PROJECTS = ["Lang", "Chart", "Cli", "Csv", "Gson"]
#DF4_PROJECTS = ["Gson", "Csv"]
#DF4_PROJECTS = ["Time"]
#DF4_PROJECTS = ["Gson"]

listPredictions: dict[dict] = {}
for project in DF4_PROJECTS:
    listPredictions[project] = {}

actualProject = ""


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
def save_in_file_list_lines(file_path: str, list_lines: list[str], msg_tqdm: str):
    #Removing older version of the file outputs
    if os.path.exists(file_path):
        os.remove(file_path)

    #Writing to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in tqdm(list_lines, desc=msg_tqdm):
            f.write(line + "\n")


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


def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_steps: number of steps processed
    - num_tokens: number of tokens processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if runtime == 0:
        return result
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
    return result


def save_prediction_results_in_csv(project_name: str, path_results: str, prediction_results: dict[list]):
    path_file_prediction_results = os.path.join(path_results, project_name + "_generated_tests.csv")
    columns_names = list(prediction_results.keys())
    #rows = zip(*prediction_results.values())
    rows = list(zip(*prediction_results.values()))
    with open(path_file_prediction_results, 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Escribimos los nombres de las columnas
        writer.writerow(columns_names)
        # Escribimos las filas
        for row in tqdm(rows, desc=f"Writting CSV results for project {project_name}"):
            writer.writerow(row)


def checkA3Test(expression):
    open_tup = tuple('({[')
    close_tup = tuple(')}]')
    map = dict(zip(open_tup, close_tup))
    queue = []

    for i in expression:
        if i in open_tup:
            queue.append(map[i])
        elif i in close_tup:
            if not queue or i != queue.pop():
                return "Unbalanced"
    if not queue:
        return "Balanced"
    else:
        return "Unbalanced"

# count = 0
def postProcessingA3Test(line):
    # global count
    # count+=1
    # print(count)
    words = line.split(" ")
    if(words[0]!="@Test"):
        words.insert(0,"@Test")

    #append test if not
    if "void" in words:
        id = words.index("void")
        if(words[id+1][:4] != "test"):
            words[id+1] = "test" + words[id+1]

    status = checkA3Test(line)
    if(status=="Unbalanced"):
        # print("Here")
        elem=None
        for word in words[::-1]:
            if(word[-1]==";"):
                elem = word
                break
        if(elem!=None):
            index_pos = len(words) - words[::-1].index(elem) - 1
            words = words[:index_pos+1]
            words.append("}")
        else:
            # with open(errLogs,"a+") as f:
            #     f.write(" ".join(words));
            #return None
            # retornamos el original
            return line

    output = " ".join(words)
    return output


# Loads and splits the data into training and validationsets with a 80/10/10 split
def load_data(type_datafiles: str, 
              cache_dir_datasets: str, 
              df4_path_inputs: str):
    
    data_files = {}
    for project in DF4_PROJECTS:
        path_input = os.path.join(df4_path_inputs, project + "_methods.csv")
        data_files[project] = path_input
    
    d4j_dataset = load_dataset(type_datafiles, data_files=data_files, cache_dir=cache_dir_datasets)

    return d4j_dataset


# encode the sentences using the tokenizer
def tokenize_dataset(d4j_dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
                     tokenizer,
                     source_label,
                     max_length,
                     cache_dir_tokenized_datasets):
    
    for dataset_name in DF4_PROJECTS:
        d4j_dataset[dataset_name] = d4j_dataset[dataset_name].map(
            lambda examples: tokenize_examples(tokenizer, dataset_name, examples, source_label, max_length),
            batched=True,
            batch_size=2000,
            writer_batch_size=2000,
            #remove_columns=["target", "src_fm_fc_ms_ff", "src_fm_fc_dctx", "src_fm_fc_dctx_priv", "imports_focal_class", "imports_test_class"],
            #num_proc=4,
            load_from_cache_file=False,
            cache_file_name=f"{cache_dir_tokenized_datasets}{dataset_name}_tokenized_cache.arrow"
        )

    return d4j_dataset


# Preprocess inputs using tokenizer. Solo tokenizamos, aún no retornamos tensores
def tokenize_examples(tokenizer, dataset_name, examples, source_label, max_seq_length):
    # encode the source
    source = examples[source_label]
    #print(f"max_seq_length: {str(max_seq_length)}")

    # print("\nLength Input: " + str(len(source)) + "\n")

    # encode de inputs
    model_inputs = tokenizer(source, max_length=max_seq_length, padding="max_length", truncation=True)
    
    # Para inferencias no se requiere de labels. Sin embargo si se desea que el trainer
    # llame al método compute_metrics custom, el input debe ir con labels. Por tanto, para
    # forzar que se llame a dicho método, en labels podríamos pasar una copia de los inputs:
    # model_inputs["labels"] = model_inputs["input_ids"].copy()
    # model_inputs["labels"] = [
    #     [(lab if lab != tokenizer.pad_token_id else -100) for lab in label] for label in model_inputs["labels"]
    # ]

    # {input_ids: [[], []...], attention_mask: [[], []...]}
    return model_inputs


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


# https://huggingface.co/blog/how-to-generate
# https://aclanthology.org/2023.emnlp-industry.73.pdf
# https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L119
def generate_predictions(tokenizer, 
                         decoder_start_token_id, 
                         max_length,
                         trainer: Seq2SeqTrainer, 
                         project_name: str,
                         d4j_dataset: Dataset, 
                         generation_strategy: str, 
                         num_return_sequences: int):
    """
    Sampling
    Características
    - Diversidad: Genera texto estocásticamente, seleccionando tokens basándose en su probabilidad. Esto puede producir salidas más creativas y variadas.
    - Riesgo de incoherencia: Mayor probabilidad de generar código sintácticamente incorrecto o menos relevante.
    - Recomendado para escenarios donde quieres explorar múltiples soluciones alternativas.
    Parámetros clave:
    - temperature: Valores bajos (ej. 0.3) generan texto más seguro pero menos diverso.
    - top_k/top_p: Limitan la selección de tokens para evitar opciones absurdas.

    Beam Search
    Características:
    - Coherencia: Mantiene múltiples secuencias candidato (beams) y selecciona las de mayor probabilidad acumulada. Ideal para generar código preciso y estructurado.
    - Menos diversidad: Tiende a converger en secuencias similares, especialmente con pocos beams.
    - Recomendado cuando la precisión sintáctica y lógica es crítica (ej. código Java válido).
    Parámetros clave:
    - num_beams: Usa valores entre 4 y 10 para equilibrio calidad/velocidad.
    - length_penalty: Ajusta para evitar salidas truncadas o redundantes.

    ¿Cuál estrategia usar?

    Beam Search es preferible para:
    - Generar código que debe compilar (ej. métodos bien estructurados).
    - Casos de prueba con requisitos específicos (ej. manejo de excepciones).

    Sampling es útil para:
    - Explorar múltiples enfoques (ej. diferentes assertions o escenarios).
    - Evitar la redundancia en casos de prueba repetitivos.

    En resumen, si se prioriza correctitud, usar beam search. Si se necesita diversidad, optar por sampling con parámetros conservadores (top_p=0.9, temperature=0.5)

    También se puede usar un enfoque híbrido si el modelo lo soporta:
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_beams=3,  # Combina beams con muestreo
        early_stopping=True,
    )
    """
    
    if generation_strategy == "greddy_search":
        predict_outputs = trainer.predict(
            test_dataset=d4j_dataset, 
            metric_key_prefix=project_name,
            eos_token_id=tokenizer.eos_token_id,
            #decoder_start_token_id = decoder_start_token_id,
            do_sample=False,
            max_length=max_length
        ) # PredictionOutput
    
    elif generation_strategy == "beam_search":
        predict_outputs = trainer.predict(
            test_dataset=d4j_dataset, 
            metric_key_prefix=project_name, 
            eos_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            decoder_start_token_id = decoder_start_token_id,
            do_sample=False,
            num_beams=numBeamsPredictions,   # Número de beams (más beams = mejor calidad, pero más lento)
            #length_penalty=0.8,             # Penaliza secuencias largas (valores < 1 acortan las salidas). Se ajusta para evitar salidas truncadas o redundantes
            early_stopping=True,            # Detiene la generación si todos los beams convergen
            num_return_sequences=num_return_sequences # Devuelve los mejores [num_return_sequences] beams
            # Para obtener las probabilidades se agregarían los siguientes parámetros:
            #output_scores=True,  # Habilitar el cálculo de las probabilidades
            #return_dict_in_generate=True,  # Devolver un diccionario con los resultados
        )
        # Para obtener las secuencias generadas y sus probabilidades, se haría de la siguiente forma:
        # sequences = output.sequences
        # scores = output.sequences_scores  # Probabilidades logarítmicas de cada secuencia
        # Se convierten las secuencias de nuevo a texto:
        # generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]
        # Se encuentra la secuencia con la probabilidad más alta:
        # best_index = scores.argmax().item()
        # best_sequence = generated_texts[best_index]
        # best_score = scores[best_index].exp().item()  # Convertir de log-prob a probabilidad
        # print(f"Secuencia con la probabilidad más alta: {best_sequence}")
        # print(f"Probabilidad: {best_score}")

        # Explicación:
        # - num_beams: Define el número de beams (haces) que se utilizarán en la búsqueda.
        # - num_return_sequences: Especifica cuántas secuencias quieres que se devuelvan.
        # - output_scores: Habilita el cálculo de las probabilidades logarítmicas de cada secuencia.
        # - return_dict_in_generate: Devuelve un diccionario con las secuencias y sus puntuaciones.
        # - scores: Contiene las probabilidades logarítmicas de cada secuencia. Puedes convertirlas a probabilidades usando .exp().

        # Nota:
        # - Las probabilidades devueltas son log-probabilidades, por lo que debes aplicar exp() para convertirlas a probabilidades normales.
        # - La secuencia con la probabilidad más alta es aquella cuyo score es el mayor.

        # Para obtener todas las secuencias ordenadas según la probabilidad más alta, se haría de la siguiente forma:
        # Acceder a las secuencias generadas y sus probabilidades
        # sequences = output.sequences
        # scores = output.sequences_scores  # Log-probabilidades de cada secuencia

        # # Convertir las secuencias de nuevo a texto
        # generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]

        # # Combinar las secuencias con sus probabilidades
        # sequence_prob_pairs = list(zip(generated_texts, scores))

        # # Ordenar las secuencias según su probabilidad (de mayor a menor)
        # sorted_sequence_prob_pairs = sorted(sequence_prob_pairs, key=lambda x: x[1], reverse=True)

        # # Mostrar las secuencias ordenadas
        # for i, (seq, log_prob) in enumerate(sorted_sequence_prob_pairs):
        #     prob = log_prob.exp().item()  # Convertir log-prob a probabilidad
        #     print(f"Secuencia {i + 1}: {seq}")
        #     print(f"Probabilidad: {prob}\n")

    elif generation_strategy == "sampling":
        # https://github.com/salesforce/CodeT5/blob/main/CodeT5%2B/humaneval/generate_codet5p.py#L122
        predict_outputs = trainer.predict(
            test_dataset=d4j_dataset, 
            metric_key_prefix=project_name, 
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id = decoder_start_token_id,
            do_sample=True,  # # Habilita el muestreo
            temperature=0.8, # Controla la aleatoriedad (0.1-1.0: valores bajos = más determinista)
            top_k=50,        # Filtra los 50 tokens más probables en cada paso
            top_p=0.95,      # Muestreo de núcleo (nucleus sampling): acumula hasta el 95% de probabilid
            num_return_sequences=num_return_sequences  # Generar múltiples opciones para comparar
        )
    
    elif generation_strategy == "top_k_sampling":
        predict_outputs = trainer.predict(
            test_dataset=d4j_dataset, 
            metric_key_prefix=project_name, 
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id = decoder_start_token_id,
            do_sample=True, 
            top_k=50,
            temperature=0.6,
            #no_repeat_ngram_size=2,
            #early_stopping=True,
            num_return_sequences=num_return_sequences
        )
    
    elif generation_strategy == "top_p_sampling":
        predict_outputs = trainer.predict(
            test_dataset=d4j_dataset, 
            metric_key_prefix=project_name, 
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id = decoder_start_token_id,
            do_sample=True, 
            top_p=0.95,
            #top_k=100, # 10, 50, 100
            temperature=0.4, # 0.4, 0.6, 0.8
            num_return_sequences=num_return_sequences
        )

        # Otra estrategia puede ser beam_group_search.
        # Sería similar a beam_search, pero agregando los siguientes parámetros:
        # - num_beam_groups: Cantidad de grupos de beams
        # - diversity_penalty: Controla la similitud en cada grupo de salidas
        # Esto podría llegar a ser útil, si por ejemplo se quieren generar diversos
        # métodos de test que validen diversos casos de uso (branches) que se dan en un método focal
        # https://huggingface.co/docs/transformers/generation_strategies

    return predict_outputs



def sorted_by_score_d4j_predicts(trainer: Seq2SeqTrainer, 
                                 # model: Union[PreTrainedModel, nn.Module], 
                                 tokenizer, 
                                 d4j_dataset: Dataset, 
                                 project_name: str, 
                                 max_length: int, 
                                 num_beams: int, 
                                 num_return_sequences: int):
    # Tokenizar el batch de entradas
    d4j_dataloader: DataLoader = trainer.get_test_dataloader(d4j_dataset)
    start_time = time.time()

    # batch_size = d4j_dataloader.batch_size
    batch_size = trainer.args.eval_batch_size
    num_examples = trainer.num_examples(d4j_dataloader)
    total_batch_size = trainer.args.eval_batch_size * trainer.args.world_size
    num_steps = math.ceil(num_examples / total_batch_size)

    print(f"***** Running predictions for {project_name} *****")
    print(f"  Num examples = {num_examples}")
    print(f"  Num steps = {num_steps}")
    print(f"  Batch size = {batch_size}\n")

    parserUtils = ParserUtils("utf-8")

    # world_size = max(1, trainer.args.world_size)
    # eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)

    trainer.model.eval()

    trainer.callback_handler.eval_dataloader = d4j_dataloader

    total_predictions = 0
    generated_tests_by_name_column: dict[str, list[str]] = {}
    # scores_by_name_column: dict[str, list[str]] = {}
    predictions_without_test_annotation = 0
    corrects_by_pos_after_changes = None
    predictions_with_error_in_all_segment = 0
    corrects_by_pos_before_changes = []
    for i in range(num_return_sequences):
        corrects_by_pos_before_changes.append([i, 0])
    
    count_steps = 1
    remaining_examples = num_examples
    # remaining_steps = num_steps

    with torch.no_grad():
        for step, inputs in enumerate(d4j_dataloader):
            # inputs = tokenizer(d4j_dataset, return_tensors="pt", padding=True, truncation=True)

            # print(f"\n\n\nStep: {step}")
            # print(inputs)

            step_batch_size = batch_size if count_steps < num_steps else remaining_examples
            count_steps += 1
            remaining_examples -= step_batch_size
            # remaining_steps -= 1

            # print(f"\n\nStep: {count_steps} -> step_batch_size: {step_batch_size} - remaining_examples: {remaining_examples}")
            output = trainer.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],  # Máscara de atención para el padding
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id = decoder_start_token_id,
                max_length=max_length,  # Longitud máxima de la secuencia generada
                num_beams=num_beams,    # Número de beams (más beams = mejor calidad, pero más lento)
                num_return_sequences=num_return_sequences,  # Número de secuencias a devolver por cada entrada
                do_sample=False,
                early_stopping=True,           # Detiene la generación si todos los beams convergen
                output_scores=True,            # Habilitar el cálculo de las probabilidades
                return_dict_in_generate=True,  # Devolver un diccionario con los resultados

                num_beam_groups=4,      # Divide los beams en grupos
                diversity_penalty=1.0,  # Fuerza diversidad entre grupos

                repetition_penalty=1.5,  # Valores típicos entre 1.0 y 2.0. Penaliza tokens ya generados, reduciendo la probabilidad de repeticiones.
                no_repeat_ngram_size=4,  # Evita que se repitan cuatrigramas
            )


            # print(f"\n\nData step {step}")
            sequences = output.sequences  # Todas las secuencias generadas
            # print(f"Sequences shape: {sequences.shape}")
            scores = output.sequences_scores  # Log-probabilidades de cada secuencia
            # print(f"Scores shape: {scores.shape}")

            # sequences.cpu() # Esto lo haríamos para usar Numpy
            # sequences = np.where(sequences != -100, sequences, tokenizer.pad_token_id)

            # Separar las secuencias y puntuaciones por entrada
            sequences = sequences.reshape(step_batch_size, num_return_sequences, -1)  # (batch_size, num_return_sequences, seq_len)
            scores = scores.reshape(step_batch_size, num_return_sequences)  # (batch_size, num_return_sequences)

            # Procesar cada entrada del batch
            for i in range(step_batch_size):
                # Obtener las secuencias y puntuaciones para esta entrada
                entry_sequences = sequences[i]
                entry_scores = scores[i]

                entry_sequences_decoded = [tokenizer.decode(seq, skip_special_tokens=True) for seq in entry_sequences]
                total_predictions += len(entry_sequences_decoded)

                # Combinar las secuencias con sus probabilidades
                sequence_prob_pairs = list(zip(entry_sequences_decoded, entry_scores))
                #sequence_prob_pairs = list(zip(entry_sequences, entry_scores))

                # Ordenar las secuencias según su probabilidad (de mayor a menor)
                sorted_sequence_prob_pairs = sorted(sequence_prob_pairs, key=lambda x: x[1], reverse=True)

                some_prediction_by_input_is_correct = False
                num_gen_test = 1
                for j, (seq, log_prob) in enumerate(sorted_sequence_prob_pairs):
                    if not seq.startswith("@Test"):
                        predictions_without_test_annotation += 1
                    
                    code_has_errors = parserUtils.validate_if_code_has_errors(seq)

                    if code_has_errors == False:
                        corrects_by_pos_before_changes[num_gen_test-1][1] += 1
                        some_prediction_by_input_is_correct = True


                    name_column = 'generated_test_' + str(num_gen_test)
                    if name_column in generated_tests_by_name_column:
                        generated_tests_by_name_column[name_column].append(seq)
                    else:
                        generated_tests_by_name_column[name_column] = [seq]

                    # log_prob is a Tensor with a float value
                    # name_column_score = 'score_test_' + str(num_gen_test)
                    # if name_column_score in scores_by_name_column:
                    #     scores_by_name_column[name_column_score].append(log_prob)
                    # else:
                    #     scores_by_name_column[name_column_score] = [log_prob]

                    num_gen_test += 1
                
                if not some_prediction_by_input_is_correct:
                    predictions_with_error_in_all_segment += 1


            trainer.control = trainer.callback_handler.on_prediction_step(trainer.args, trainer.state, trainer.control)

    # predictions = np.where(sequences != -100, sequences, tokenizer.pad_token_id)
    # print(f"\n\nPredictions: {predictions.shape}")
    #print(predictions)

    metrics = speed_metrics(
        project_name,
        start_time,
        num_samples=num_examples,
        num_steps=num_steps,
    )

    trainer.control = trainer.callback_handler.on_predict(trainer.args, trainer.state, trainer.control, metrics)

    total_outputs = len(d4j_dataset["id"])
    num_results = num_return_sequences
    prediction_results = {
        'id': d4j_dataset['id'], 
        'project': d4j_dataset['project'], 
        'version': d4j_dataset['version'], 
        'number_method_by_version': d4j_dataset['number_method_by_version'], 
        'modified_class': d4j_dataset['modified_class'], 
        'focal_method': d4j_dataset['focal_method'], 
        'focal_class': d4j_dataset['focal_class'], 
        'path_focal_class_in_project': d4j_dataset['path_focal_class_in_project'], 
        'test_class': d4j_dataset['test_class'], 
        'path_test_class': d4j_dataset['path_test_class'], 
        'package_focal_class': d4j_dataset['package_focal_class'], 
        'imports_focal_class': d4j_dataset['imports_focal_class'], 
        'num_results': [num_results] * total_outputs
    }
    # prediction_results.update(scores_by_name_column)
    prediction_results.update(generated_tests_by_name_column)


    print("")
    trainer.log_metrics(project_name, metrics)
    metrics.update(
        {
            f"{project_name}_runtime_formatted": secs2timedelta(metrics[f"{project_name}_runtime"]),
            f"{project_name}_samples": num_examples,
            f"{project_name}_steps": num_steps,
        }
    )
    trainer.save_metrics(project_name, metrics)

    print(f"\ncorrects_by_pos_before_changes: {str(corrects_by_pos_before_changes)}")
    print(f"\npredictions_with_error_in_all_segment: {str(predictions_with_error_in_all_segment)}")
    print(f"\npredictions_without_test_annotation: {str(predictions_without_test_annotation)}\n")

    return prediction_results, total_predictions, corrects_by_pos_before_changes, corrects_by_pos_after_changes, predictions_with_error_in_all_segment, predictions_without_test_annotation, metrics



def d4j_predicts(tokenizer, 
                 decoder_start_token_id,
                 max_length,
                 trainer: Seq2SeqTrainer,
                 project_name: str,
                 d4j_dataset: Dataset, 
                 generation_strategy: str, 
                 num_return_sequences: int):

    predict_outputs = generate_predictions(
        tokenizer=tokenizer,
        decoder_start_token_id=decoder_start_token_id,
        max_length=max_length,
        trainer=trainer,
        project_name=project_name,
        d4j_dataset=d4j_dataset, 
        generation_strategy=generation_strategy,
        num_return_sequences=num_return_sequences
    ) # PredictionOutput
    #print("\n\n")
    #print("test_outputs")
    #print(test_outputs)
    #print("\n\n")

    predictions = np.where(predict_outputs.predictions != -100, predict_outputs.predictions, tokenizer.pad_token_id)
    #logits = test_outputs.predictions.argmax(axis=-1)

    print(f"\n\nPredictions: {predictions.shape}")
    #print(predictions)

    predictions_decoded = tokenizer.batch_decode(
        predictions, skip_special_tokens=True #, clean_up_tokenization_spaces=True
    )
    #print(f"\n\nPredictions decoded: {len(predictions_decoded)}")
    #print(predictions_decoded)

    list_predictions = lmap(str.strip, predictions_decoded)
    #list_predictions = [s.replace('<pad>', '') for s in list_predictions]
    print(f"\n\nPredictions decoded formatted: {len(list_predictions)}")
    # for pred in list_predictions:
    #     print("\n" + pred)


    predictions_with_error_in_all_segment = None
    corrects_by_pos_before_changes = None
    corrects_by_pos_after_changes = None
    parserUtils = ParserUtils("utf-8")

    if firstSampleNotWrong:
        (list_predictions, 
         corrects_by_pos_before_changes, 
         corrects_by_pos_after_changes, 
         predictions_with_error_in_all_segment) = list_predictions_getting_first_not_wrong(
                parserUtils, 
                list_predictions, 
                num_return_sequences
        )
    else:
        (corrects_by_pos_before_changes, 
         predictions_with_error_in_all_segment) = only_validate_if_predictions_has_errors(
                parserUtils, 
                list_predictions, 
                num_return_sequences
         )

    total_predictions = len(list_predictions)

    segment_length = len(d4j_dataset["id"]) # input_length
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

    print(f"\n\nSegments predictions: {len(segmented_predictions)}\n")

    num_results = len(segmented_predictions)
    prediction_results = {
        'id': d4j_dataset['id'], 
        'project': d4j_dataset['project'], 
        'version': d4j_dataset['version'], 
        'number_method_by_version': d4j_dataset['number_method_by_version'], 
        'modified_class': d4j_dataset['modified_class'], 
        'focal_method': d4j_dataset['focal_method'], 
        'focal_class': d4j_dataset['focal_class'], 
        'path_focal_class_in_project': d4j_dataset['path_focal_class_in_project'], 
        'test_class': d4j_dataset['test_class'], 
        'path_test_class': d4j_dataset['path_test_class'], 
        'package_focal_class': d4j_dataset['package_focal_class'], 
        'imports_focal_class': d4j_dataset['imports_focal_class'], 
        #'num_results': [num_results for i in range(segment_length)],
        'num_results': [num_results] * segment_length,
    }

    index = 1
    predictions_without_test_annotation = 0
    for list_segment_generated_test_i_predicts in segmented_predictions:
        #print(list_segment_predicts)
        for generated_tets in list_segment_generated_test_i_predicts:
            if not generated_tets.startswith("@Test"):
                predictions_without_test_annotation += 1
        prediction_results['generated_test_' + str(index)] = list_segment_generated_test_i_predicts
        index += 1

    trainer.log_metrics(project_name, predict_outputs.metrics)
    trainer.save_metrics(project_name, predict_outputs.metrics)
    
    print(f"\npredictions_without_test_annotation: {str(predictions_without_test_annotation)}\n")

    # print("\n\n")
    # print("Results:")
    # print(prediction_results)
    # print("\n\n")

    return prediction_results, total_predictions, corrects_by_pos_before_changes, corrects_by_pos_after_changes, predictions_with_error_in_all_segment, predictions_without_test_annotation, predict_outputs.metrics


def list_predictions_getting_first_not_wrong(parserUtils: ParserUtils, list_predictions: list, num_return_sequences: int):
    new_list_predictions = []
    #position = 0

    predictions_with_error_in_all_segment = 0
    corrects_by_pos_before_changes = []
    corrects_by_pos_after_changes = []
    for i in range(num_return_sequences):
        corrects_by_pos_before_changes.append([i, 0])
        corrects_by_pos_after_changes.append([i, 0])

    #segmented_predictions = []
    #for i in range(count_segments):
    #    segment = []
    #    for j in range(0, (seg_length)):
    #        segment.append(list_predicts[i + (j * count_segments)])
    #    segmented_predictions.append(segment)

    for i in range(0, len(list_predictions), num_return_sequences):
        first_predict_by_target = list_predictions[i]
        some_prediction_by_input_is_correct = False

        pos_in_segment = 0
        for j in range(i, i + num_return_sequences, 1):
            code_before_changes = list_predictions[j]

            if "A3Test" in inputModel:
                code_before_fixed_has_errors = parserUtils.validate_if_code_has_errors(code_before_changes)
                code_without_missings = postProcessingA3Test(code_before_changes)
                # print("\n\nBefore:")
                # print(code_before_changes)

                # print("\n\nAfter:")
                # print(code_without_missings)
                # print("")
            else:
                code_before_fixed_has_errors, code_without_missings = parserUtils.fix_missings_and_errors_in_code(code_before_changes)

            code_after_fixed_has_errors = parserUtils.validate_if_code_has_errors(code_without_missings)
            
            #print("\n\n")
            #print(code_before_changes)

            # if not code_after_changes_has_errors:
            #     #print()
            #     code_without_missings = parserUtils.adapt_name_and_modifiers(code_without_missings)
            #     #print(code_without_missings)
            
            #print("\n\n")
            #print(f"Prediction ({str(position)}, {str(j)}) has error -> Before changes: {str(code_before_changes_has_errors)} - After changes: {str(code_after_changes_has_errors)}")

            if not code_before_fixed_has_errors:
                #print("Add prediction without error (before changes): " + str(j))
                corrects_by_pos_before_changes[pos_in_segment][1] = corrects_by_pos_before_changes[pos_in_segment][1] + 1
                if "A3Test" not in inputModel:
                    code_before_changes = parserUtils.adapt_name_and_modifiers(code_before_changes)
                new_list_predictions.append(code_before_changes)
                some_prediction_by_input_is_correct = True
                break

            if not code_after_fixed_has_errors:
                #print("Add prediction without error (before changes): " + str(j))
                corrects_by_pos_after_changes[pos_in_segment][1] = corrects_by_pos_after_changes[pos_in_segment][1] + 1
                if "A3Test" not in inputModel:
                    code_without_missings = parserUtils.adapt_name_and_modifiers(code_without_missings)
                new_list_predictions.append(code_without_missings)
                some_prediction_by_input_is_correct = True
                break
            
            pos_in_segment += 1
        
        if not some_prediction_by_input_is_correct: # Todos los candidatos para el input tienen error de sintaxis
            #print(f"Add first prediction by target: ({str(position)}, {str(i)})")
            new_list_predictions.append(first_predict_by_target)
            predictions_with_error_in_all_segment += 1
        
        #position += 1
    
    list_predictions = new_list_predictions

    print(f"\n\nlist_predictions with fixed errors: {len(list_predictions)}")
    print(f"\ncorrects_by_pos_before_changes: {str(corrects_by_pos_before_changes)}")
    print(f"\ncorrects_by_pos_after_changes: {str(corrects_by_pos_after_changes)}")
    # predictions_with_error_in_all_returned_sequences_by_input
    print(f"\npredictions_with_error_in_all_segment: {str(predictions_with_error_in_all_segment)}")

    return list_predictions, corrects_by_pos_before_changes, corrects_by_pos_after_changes, predictions_with_error_in_all_segment


def only_validate_if_predictions_has_errors(parserUtils: ParserUtils, list_predictions: list, num_return_sequences: int):
    """
    Con esta función, se valida por cada output si tiene error de sintaxis o no
    """
    predictions_with_error_in_all_segment = 0
    corrects_by_pos_before_changes = []
    for i in range(num_return_sequences):
        corrects_by_pos_before_changes.append([i, 0])

    #segmented_predictions = []
    #segment_length = len(targets)
    #count_segments = len(list_predictions) // segment_length (= num_return_sequences)
    #for i in range(count_segments):
    #    segment = []
    #    for j in range(0, (segment_length)):
    #        segment.append(list_predicts[i + (j * count_segments)])
    #    segmented_predictions.append(segment)

    # 5 targets, 3 results by target, then output are 15 predicts
    # count_segments = 15 // 5 = 3 => Obtengo 3 listas de predicts, cada una con 5 items
    # 
    # for i = 0 (hasta 3):
    #   segment 0
    #   for j = 0 (hasta 5):
    #       get list[0 + (0 * 3)]
    #       segment 0 => get list[0]
    #   for j = 1 (hasta 5):
    #       get list[0 + (1 * 3)]
    #       segment 0 => get list[3]
    #   for j = 2 (hasta 5):
    #       get list[0 + (2 * 3)]
    #       segment 0 => get list[6]
    #   for j = 3 (hasta 5):
    #       get list[0 + (3 * 3)]
    #       segment 0 => get list[9]
    #   for j = 4 (hasta 5):
    #       get list[0 + (4 * 3)]
    #       segment 0 => get list[12]
    #
    # for i = 1 (hasta 3):
    #   segment 1
    #   for j = 0 (hasta 5):
    #       get list[1 + (0 * 3)]
    #       segment 1 => get list[1]
    #   for j = 1 (hasta 5):
    #       get list[1 + (1 * 3)]
    #       segment 1 => get list[4]
    #   for j = 2 (hasta 5):
    #       get list[1 + (2 * 3)]
    #       segment 1 => get list[7]
    #   for j = 3 (hasta 5):
    #       get list[1 + (3 * 3)]
    #       segment 1 => get list[10]
    #   for j = 4 (hasta 5):
    #       get list[1 + (4 * 3)]
    #       segment 1 => get list[13]
    #
    # for i = 2 (hasta 3):
    #   segment 2
    #   for j = 0 (hasta 5):
    #       get list[2 + (0 * 3)]
    #       segment 2 => get list[2]
    #   for j = 1 (hasta 5):
    #       get list[2 + (1 * 3)]
    #       segment 2 => get list[5]
    #   for j = 2 (hasta 5):
    #       get list[2 + (2 * 3)]
    #       segment 2 => get list[8]
    #   for j = 3 (hasta 5):
    #       get list[2 + (3 * 3)]
    #       segment 2 => get list[11]
    #   for j = 4 (hasta 5):
    #       get list[2 + (4 * 3)]
    #       segment 2 => get list[14]


    # 5 targets, 3 results by target, then output are 15 predicts
    # len(list_predictions) = 15
    # num_return_sequences = 3
    #
    # for i = 0 (hasta 15, con saltos de 3):
    #   for j = 0 (hasta 3):
    #       get list[0]
    #   for j = 1 (hasta 3):
    #       get list[1]
    #   for j = 2 (hasta 3):
    #       get list[2]
    #  
    # for i = 3 (hasta 15, con saltos de 3):
    #   for j = 3 (hasta 6):
    #       get list[3]
    #   for j = 4 (hasta 6):
    #       get list[4]
    #   for j = 5 (hasta 6):
    #       get list[5]
    #
    # for i = 6 (hasta 15, con saltos de 3):
    #   for j = 6 (hasta 9):
    #       get list[6]
    #   for j = 7 (hasta 9):
    #       get list[7]
    #   for j = 8 (hasta 9):
    #       get list[8]

    for i in range(0, len(list_predictions), num_return_sequences):
        some_prediction_by_input_is_correct = False

        pos_in_segment = 0
        for j in range(i, i + num_return_sequences, 1):
            code_without_changes = list_predictions[j]
            code_has_errors = parserUtils.validate_if_code_has_errors(code_without_changes)

            if code_has_errors == False:
                corrects_by_pos_before_changes[pos_in_segment][1] = corrects_by_pos_before_changes[pos_in_segment][1] + 1
                some_prediction_by_input_is_correct = True
                #break
            
            pos_in_segment += 1
        
        if not some_prediction_by_input_is_correct:
            predictions_with_error_in_all_segment += 1

    print(f"\ncorrects_by_pos_before_changes: {str(corrects_by_pos_before_changes)}")
    # predictions_with_error_in_all_returned_sequences_by_input
    print(f"\npredictions_with_error_in_all_segment: {str(predictions_with_error_in_all_segment)}")

    return corrects_by_pos_before_changes, predictions_with_error_in_all_segment



def compute_metrics(eval: EvalPrediction):
    # print("\nActual project: " + actualProject)
    # predictions, label_ids, inputs
    listPredictions[actualProject] = {
        "inputs": eval.inputs,
        "predictions": eval.predictions
    }
    # print(f"\n\nlistPredictions INNER: {str(len(listPredictions[actualProject]))}")
    return {}


if __name__ == '__main__':
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
    
    print("Cleaning memory...")
    report_gpu()
    show_cache()

    os.makedirs(outputPath, exist_ok=True)

    results = []

    print(f"Device: {device}  |  Accelerator: {accelerator}")

    if useFp16 and useBf16:
        raise "Debe indicar un solo tipo de precision: fp16 o bf16. Si no indica ninguno, por defecto será fp32"
    
    half_precision_backend = "auto" # por defecto
    if useBf16:
        half_precision_backend = "cpu_amp" # bf16 solo se puede ejecutar sobre CPU o sobre NVIDIA con arquitectura AMP o superior

    precision = "fp32"
    if useFp16:
        precision = "fp16"
    elif useBf16:
        precision = "bf16"

    print(f"Precision: {precision}  |  Half Precision Backend: {half_precision_backend}")


    # Load tokenizer from existing one to re-use special tokens
    #tokenizer = RobertaTokenizer.from_pretrained(model_base, cache_dir=cacheDirHuggingFace)
    print("\nLoading Tokenizer...")
    tokenizer = None
    max_seq_length = None
    #if inputModel == "SaranyaAakas/A3Test":
    #tokenizer = AutoTokenizer.from_pretrained(inputModel, cache_dir=cacheDirHuggingFace)
    if "A3Test" in inputModel:
        tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-large", cache_dir=cacheDirHuggingFace)
        max_seq_length = 1024
        #tokenizer.model_max_length = max_seq_length
    else:
        tokenizer = AutoTokenizer.from_pretrained(inputModel, cache_dir=cacheDirHuggingFace)
        max_seq_length = tokenizer.model_max_length
    print("Type tokenizer: " + str(type(tokenizer)))
    print("Length tokenizer: " + str(len(tokenizer)))

    #max_seq_length = tokenizer.model_max_length
    # print("\nLoading Configs...")
    # config = T5Config.from_pretrained(inputModel, cache_dir=cacheDirHuggingFace)
    # max_seq_length = config.n_positions
    print("Max seq length: " + str(max_seq_length))


    # Load data
    print("\nLoading Data...")
    loaded_d4j_dataset = load_data(type_datafiles="csv",
                                   cache_dir_datasets=cacheDirDatasets,
                                   df4_path_inputs=defects4jInputs)
    summary_loaded_dataset = str(loaded_d4j_dataset)
    print(summary_loaded_dataset)
    print()
    #print(loaded_d4j_dataset["train"][0])
    #print(loaded_d4j_dataset["valid"][0])

    tokenized_datasets = tokenize_dataset(d4j_dataset=loaded_d4j_dataset,
                                          tokenizer=tokenizer,
                                          source_label=sourceLabel,
                                          max_length=max_seq_length,
                                          cache_dir_tokenized_datasets=cacheDirDatasetTokenized)
    summary_tokenized_dataset = str(tokenized_datasets)
    print(summary_tokenized_dataset)
    print()
    #print(tokenized_datasets["train"][0])
    #print(tokenized_datasets["valid"][0])

    #assert 128 == max_seq_length, "Return forced"

    print("\nShowing memory after load data...")
    show_cache()


    # Load pretrained model
    print("\nLoading Model...")
    assert inputModel is not None, "You must indicate an input model"

    #if inputModel == "SaranyaAakas/A3Test":
    model = None
    decoder_start_token_id = None
    if "A3Test" in inputModel:
        model = PLBartForConditionalGeneration.from_pretrained(inputModel, cache_dir=cacheDirHuggingFace)
        decoder_start_token_id = tokenizer.bos_token_id #Probar usando tokenizer.pad_token_id
        model.config.max_position_embeddings = max_seq_length
        model.max_seq_length = max_seq_length
    else:
        model = T5ForConditionalGeneration.from_pretrained(inputModel, cache_dir=cacheDirHuggingFace)
        config = T5Config.from_pretrained(inputModel, cache_dir=cacheDirHuggingFace)
        decoder_start_token_id = config.decoder_start_token_id
    model.to(device)


    print("\nShowing memory after load model...")
    show_cache()


    # Solamente necesitamos que el DataCollator convierta los inputs a tensores
    data_collator = DefaultDataCollator(
        return_tensors="pt"
    )


    training_args = Seq2SeqTrainingArguments(
        use_cpu=accelerator == "cpu",
        output_dir=outputPath,
        overwrite_output_dir=True,
        seed=SEED,
        per_device_eval_batch_size=batchSizeValidation,

        #fp16=useFp16,
        #bf16=useBf16,
        #half_precision_backend=half_precision_backend,
        #fp16_full_eval=True, Más rápido, pero puede afectar las métricas
        #bf16_full_eval=True, Estas opciones lo que hacen es setear el tipo de dato del modelo en precisión mixta

        # logging strategies
        disable_tqdm=False,
        log_level="info", # default: passive
        #skip_memory_metrics=False,

        # prediction
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        generation_max_length=max_seq_length
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    predict_metrics = []
    print("\nInit Predictions Generation...")
    start_time_all_predicts = time.time()

    for project in DF4_PROJECTS:
        print(f"\n\n\n\nGenerating predictions for project {project}...\n")
        #partial_predict_dataset = tokenized_datasets[project].select(range(10))
        partial_predict_dataset = tokenized_datasets[project]
        actualProject = project

        if generationStrategy == 'beam_search' and "A3Test" not in inputModel:
            (prediction_results, 
            total_predictions,
            corrects_by_pos_before_changes, 
            corrects_by_pos_after_changes, 
            predictions_with_error_in_all_segment, 
            predictions_without_test_annotation,
            predict_outputs_metrics) = sorted_by_score_d4j_predicts(
                    trainer=trainer,
                    tokenizer=tokenizer, 
                    d4j_dataset=partial_predict_dataset, 
                    project_name=project, 
                    max_length=max_seq_length, 
                    num_beams=numBeamsPredictions, 
                    num_return_sequences=numReturnPredictions
            )
        else:
            (prediction_results, 
            total_predictions,
            corrects_by_pos_before_changes, 
            corrects_by_pos_after_changes, 
            predictions_with_error_in_all_segment, 
            predictions_without_test_annotation,
            predict_outputs_metrics) = d4j_predicts(
                    tokenizer=tokenizer, 
                    decoder_start_token_id=decoder_start_token_id,
                    max_length=max_seq_length,
                    trainer=trainer, 
                    project_name=project,
                    d4j_dataset=partial_predict_dataset, 
                    generation_strategy=generationStrategy,
                    num_return_sequences=numReturnPredictions
            )
        total_outputs_records = len(prediction_results['id'])

        save_prediction_results_in_csv(project, outputPath, prediction_results)

        project_predicts_runtime = predict_outputs_metrics[project + '_runtime']
        project_samples_per_second = predict_outputs_metrics[project + '_samples_per_second']
        elapsed_time_per_sample = 1.0 / project_samples_per_second
        elapsed_time_per_sample_formatted = secs2timedelta(elapsed_time_per_sample)

        project_predicts_runtime_formatted = secs2timedelta(project_predicts_runtime)
        predict_metrics.append("==================================================")
        predict_metrics.append(f"Predictions Results Project {project}:\n")
        predict_metrics.append(f"Elapsed time: {str(project_predicts_runtime)} seconds")
        predict_metrics.append(f"Elapsed time (formatted): {project_predicts_runtime_formatted}")
        predict_metrics.append(f"Samples per second: {str(project_samples_per_second)}")
        predict_metrics.append(f"Elapsed time per sample: {str(elapsed_time_per_sample)} seconds")
        predict_metrics.append(f"Elapsed time per sample (formatted): {elapsed_time_per_sample_formatted}\n")
        predict_metrics.append(f"total_outputs_records: {str(total_outputs_records)}")
        predict_metrics.append(f"total_predictions: {str(total_predictions)}")
        predict_metrics.append(f"predictions_without_test_annotation: {str(predictions_without_test_annotation)}")
        predict_metrics.append(f"corrects_by_pos_beam_before_changes: {str(corrects_by_pos_before_changes)}")
        predict_metrics.append(f"corrects_by_pos_beam_after_changes: {str(corrects_by_pos_after_changes)}")
        predict_metrics.append(f"predictions_with_error_in_all_segment: {str(predictions_with_error_in_all_segment)}\n\n")
    
    predict_metrics.append("==================================================")
    
    end_time_all_predicts = time.time()
    elapsed_time_all_predicts = end_time_all_predicts - start_time_all_predicts
    elapsed_time_all_predicts_formatted = secs2timedelta(elapsed_time_all_predicts)
    predict_metrics.append("\n\nALL PREDICTIONS RESULTS:\n")
    predict_metrics.append(f"Model: {inputModel}")
    # predict_metrics.append(f"Precision: {precision}")
    predict_metrics.append(f"Elapsed time: {elapsed_time_all_predicts} seconds")
    predict_metrics.append(f"Elapsed time (formatted): {elapsed_time_all_predicts_formatted}")
    
    output_predict_metrics = os.path.join(outputPath, "predict_metrics.txt")
    print("\n\n\nSaving Predict Metrics Results File...")
    save_in_file_list_lines(output_predict_metrics, predict_metrics, "Saving predict metrics results in file")

    print("\n\nFINISH PREDICTIONS")
    print(f"Elapsed time: {elapsed_time_all_predicts} seconds")
    print(f"Elapsed time (formatted): {elapsed_time_all_predicts_formatted}")

    print("\nCleaning memory after finish...")
    show_cache()
    report_gpu()
    show_cache()

