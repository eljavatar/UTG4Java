import argparse
import copy
import json
import os
import pandas as pd
import re
import time
from tqdm import tqdm
from parser_utils import ParserUtils


summary_by_project = {}
list_summary_syntax_report: list = []

all_total_tests = 0
total_first_original_syntax_corrects = 0
total_first_original_syntax_errors = 0
total_first_a3test_strategy_syntax_corrects = 0
total_first_a3test_strategy_syntax_errors = 0
total_first_ast_strategy_syntax_corrects = 0
total_first_ast_strategy_syntax_errors = 0


def checkErrorsWithA3TestStrategy(expression):
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
def postProcessingWithA3TestStrategy(line):
    # global count
    # count+=1
    # print(count)
    words = line.split(" ")
    if (words[0]!="@Test"):
        words.insert(0,"@Test")

    #append test if not
    if "void" in words:
        id = words.index("void")
        if (id + 1) >= len(words):
            # La palabra void es la última
            return line

        if (len(words[id+1]) < 4 or words[id+1][:4] != "test"):
            words[id+1] = "test" + words[id+1]
        # try:
        #     if (not words[id+1].startswith("test")):
        #         words[id+1] = "test" + words[id+1]
        # except Exception as ex:
        #     print(f"Original line: \n{line}")
        #     print(f"\n\nWords updated (id void = {id}): \n{words}")
        #     raise ex

    status = checkErrorsWithA3TestStrategy(line)
    if (status=="Unbalanced"):
        # print("Here")
        elem = None
        for word in words[::-1]: # Recorre la lista de palabras en orden inverso
            try:
                if (word[-1]==";"):
                    elem = word
                    break
            except Exception as ex:
                # print(f"Original line: \n{line}")
                # print(f"\n\nWords updated (word = {word}): \n{words}")
                # raise ex
                return line
        
        if (elem!=None):
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


def post_processing_project(input_generated_tests: str, file_generated_tests: str, output: str):
    path_generated_tests_project = os.path.join(input_generated_tests, file_generated_tests + ".csv")
    path_generated_tests_first_origin = os.path.join(output, file_generated_tests + "_first_origin.csv")
    path_generated_tests_all_fixed_ast_strategy = os.path.join(output, file_generated_tests + "_all_fixed_with_ast_strategy.csv")
    path_generated_tests_first_not_wrong_ast_strategy = os.path.join(output, file_generated_tests + "_first_not_wrong_with_ast_strategy.csv")
    path_generated_tests_all_adapted_a3test_strategy = os.path.join(output, file_generated_tests + "_all_adapted_with_a3test_strategy.csv")
    path_generated_tests_first_adapted_a3test_strategy = os.path.join(output, file_generated_tests + "_first_adapted_with_a3test_strategy.csv")

    if not os.path.exists(path_generated_tests_project):
        print(f"Not exists file {path_generated_tests_project}")
        return

    df = pd.read_csv(path_generated_tests_project, encoding="utf-8")
    df.replace({pd.NA: None}, inplace=True) # Reemplaza los NaN de pandas por None
    list_generated_tests = df.to_dict(orient='records')

    # print(f"len_list_generated_tests: {len(list_generated_tests)}\n")
    #print(list_generated_tests[0])

    parserUtils = ParserUtils("utf-8")

    total_empty_tests = 0

    list_generated_tests_first_origin: list[dict] = []

    count_with_fixed_tests_ast_strategy = 0
    errors_after_adapt_name_ast_strategy = 0
    list_generated_tests_all_fixed_ast_strategy: list[dict] = []
    list_generated_tests_first_not_wrong_ast_strategy: list[dict] = []

    count_with_first_adapted_test_correct_a3test_strategy = 0
    count_with_some_correct_adapted_test_a3test_strategy = 0
    errors_after_adapt_test_a3test_strategy = 0
    list_generated_tests_all_adapted_a3test_strategy: list[dict] = []
    list_generated_tests_first_adapted_a3test_strategy: list[dict] = []

    total = 0

    first_original_syntax_corrects = 0

    for generated_tests in tqdm(list_generated_tests, desc=f"PostProcessing file {file_generated_tests}"):
        num_results = generated_tests['num_results']

        generated_tests_first_origin = copy.deepcopy(generated_tests)
        generated_tests_first_origin['num_results'] = 1

        generated_tests_all_fixed_ast_strategy = copy.deepcopy(generated_tests)
        generated_tests_first_not_wrong_ast_strategy = copy.deepcopy(generated_tests)
        generated_tests_all_adapted_a3test_strategy = copy.deepcopy(generated_tests)
        generated_tests_first_adapted_a3test_strategy = copy.deepcopy(generated_tests)

        list_tests_to_analize = []

        first_original_generated_test = generated_tests['generated_test_1']
        first_original_generated_test_has_errors = parserUtils.validate_if_code_has_errors(first_original_generated_test)
        if not first_original_generated_test_has_errors:
            first_original_syntax_corrects += 1

        for i in range(1, (num_results + 1)):
            key_generated_test = 'generated_test_' + str(i)
            # print(f"Analyzing test: {key_generated_test}")

            # if i == 20:
            #     print(f"{key_generated_test}: {generated_tests[key_generated_test]}")

            # if generated_tests[key_generated_test] is None:
            #     print(f"El test {i} is None")

            list_tests_to_analize.append(str(generated_tests[key_generated_test]))

            if i > 1:
                generated_tests_first_origin.pop(key_generated_test)
            
            # Removemos todos los tests_generados de los resultados, ya que
            # los reemplazaremos con los resultados del post procesamiento
            generated_tests_all_fixed_ast_strategy.pop(key_generated_test)
            generated_tests_first_not_wrong_ast_strategy.pop(key_generated_test)
            generated_tests_all_adapted_a3test_strategy.pop(key_generated_test)
            generated_tests_first_adapted_a3test_strategy.pop(key_generated_test)
        

        (count_empty_tests,
         list_all_fixed_tests_ast_strategy, 
         first_test_case_not_wrong_ast_strategy, 
         is_there_a_fixed_test_ast_strategy, 
         some_test_has_error_after_adapt_name_ast_strategy,
         list_all_adapted_tests_a3test_strategy,
         first_adapted_test_is_correct_a3test_strategy,
         is_there_already_a_fixed_test_a3test_strategy,
         some_test_has_error_after_adapt_a3test_strategy) = get_fixed_tests(parserUtils, list_tests_to_analize)

        try:
            for i in range(1, (num_results + 1)):
                generated_tests_all_fixed_ast_strategy['generated_test_' + str(i)] = list_all_fixed_tests_ast_strategy[i - 1]
                generated_tests_all_adapted_a3test_strategy['generated_test_' + str(i)] = list_all_adapted_tests_a3test_strategy[i -1]
        except Exception as ex:
            print(f"\n\n\n\nId: {generated_tests['id']} - Empty tests: {count_empty_tests}  - Length AST: {len(list_all_fixed_tests_ast_strategy)}  - Length A3Test: {len(list_all_adapted_tests_a3test_strategy)}\n\n\n\n")
            raise ex

        
        generated_tests_first_not_wrong_ast_strategy['num_results'] = 1
        generated_tests_first_not_wrong_ast_strategy['generated_test_1'] = first_test_case_not_wrong_ast_strategy

        generated_tests_first_adapted_a3test_strategy['num_results'] = 1
        generated_tests_first_adapted_a3test_strategy['generated_test_1'] = list_all_adapted_tests_a3test_strategy[0]

        total_empty_tests += count_empty_tests

        if is_there_a_fixed_test_ast_strategy:
            count_with_fixed_tests_ast_strategy += 1
        if some_test_has_error_after_adapt_name_ast_strategy:
            errors_after_adapt_name_ast_strategy += 1
        
        if first_adapted_test_is_correct_a3test_strategy:
            count_with_first_adapted_test_correct_a3test_strategy += 1
        if is_there_already_a_fixed_test_a3test_strategy:
            count_with_some_correct_adapted_test_a3test_strategy += 1
        if some_test_has_error_after_adapt_a3test_strategy:
            errors_after_adapt_test_a3test_strategy += 1

        
        list_generated_tests_first_origin.append(generated_tests_first_origin)
        list_generated_tests_all_fixed_ast_strategy.append(generated_tests_all_fixed_ast_strategy)
        list_generated_tests_first_not_wrong_ast_strategy.append(generated_tests_first_not_wrong_ast_strategy)
        list_generated_tests_all_adapted_a3test_strategy.append(generated_tests_all_adapted_a3test_strategy)
        list_generated_tests_first_adapted_a3test_strategy.append(generated_tests_first_adapted_a3test_strategy)

        total += 1
        # if total > 0:
        #     break

    
    summary_by_project[file_generated_tests] = {
        'total': total,
        'total_empty_tests': total_empty_tests,
        'count_corrects_first_original': first_original_syntax_corrects,
        'count_with_fixed_tests_ast_strategy': count_with_fixed_tests_ast_strategy,
        'errors_after_adapt_name_ast_strategy': errors_after_adapt_name_ast_strategy,
        'count_with_first_adapted_test_correct_a3test_strategy': count_with_first_adapted_test_correct_a3test_strategy,
        'count_with_some_correct_adapted_test_a3test_strategy': count_with_some_correct_adapted_test_a3test_strategy,
        'errors_after_adapt_test_a3test_strategy': errors_after_adapt_test_a3test_strategy
    }


    first_original_syntax_errors = total - first_original_syntax_corrects
    first_a3test_strategy_syntax_errors = total - count_with_first_adapted_test_correct_a3test_strategy
    first_ast_strategy_syntax_errors = total - count_with_fixed_tests_ast_strategy

    syntax_summary = {
        'project': file_generated_tests.split("_")[0],
        'total_tests': total,
        'first_original_syntax_corrects': first_original_syntax_corrects,
        'first_original_rate_syntax_corrects': "{:.2f}".format((first_original_syntax_corrects / total) * 100),
        'first_original_syntax_errors': first_original_syntax_errors,
        'first_original_rate_syntax_errors': "{:.2f}".format((first_original_syntax_errors / total) * 100),
        'first_a3test_strategy_syntax_corrects': count_with_first_adapted_test_correct_a3test_strategy,
        'first_a3test_strategy_rate_syntax_corrects': "{:.2f}".format((count_with_first_adapted_test_correct_a3test_strategy / total) * 100),
        'first_a3test_strategy_syntax_errors': first_a3test_strategy_syntax_errors,
        'first_a3test_strategy_rate_syntax_errors': "{:.2f}".format((first_a3test_strategy_syntax_errors / total) * 100),
        'first_ast_strategy_syntax_corrects': count_with_fixed_tests_ast_strategy,
        'first_ast_strategy_rate_syntax_corrects': "{:.2f}".format((count_with_fixed_tests_ast_strategy / total) * 100),
        'first_ast_strategy_syntax_errors': first_ast_strategy_syntax_errors,
        'first_ast_strategy_rate_syntax_errors': "{:.2f}".format((first_ast_strategy_syntax_errors / total) * 100)
    }
    
    list_summary_syntax_report.append(syntax_summary)

    global all_total_tests
    global total_first_original_syntax_corrects
    global total_first_original_syntax_errors
    global total_first_a3test_strategy_syntax_corrects
    global total_first_a3test_strategy_syntax_errors
    global total_first_ast_strategy_syntax_corrects
    global total_first_ast_strategy_syntax_errors

    all_total_tests += total
    total_first_original_syntax_corrects += first_original_syntax_corrects
    total_first_original_syntax_errors += first_original_syntax_errors
    total_first_a3test_strategy_syntax_corrects += count_with_first_adapted_test_correct_a3test_strategy
    total_first_a3test_strategy_syntax_errors += first_a3test_strategy_syntax_errors
    total_first_ast_strategy_syntax_corrects += count_with_fixed_tests_ast_strategy
    total_first_ast_strategy_syntax_errors += first_ast_strategy_syntax_errors

    
    df_generated_tests_first_origin = pd.DataFrame(list_generated_tests_first_origin)
    df_generated_tests_all_fixed_ast_strategy = pd.DataFrame(list_generated_tests_all_fixed_ast_strategy)
    df_generated_tests_first_not_wrong_ast_strategy = pd.DataFrame(list_generated_tests_first_not_wrong_ast_strategy)
    df_generated_tests_all_adapted_a3test_strategy = pd.DataFrame(list_generated_tests_all_adapted_a3test_strategy)
    df_generated_tests_first_adapted_a3test_strategy = pd.DataFrame(list_generated_tests_first_adapted_a3test_strategy)

    os.makedirs(output, exist_ok=True)

    df_generated_tests_first_origin.to_csv(path_generated_tests_first_origin, index=False, encoding="utf-8")
    df_generated_tests_all_fixed_ast_strategy.to_csv(path_generated_tests_all_fixed_ast_strategy, index=False, encoding="utf-8")
    df_generated_tests_first_not_wrong_ast_strategy.to_csv(path_generated_tests_first_not_wrong_ast_strategy, index=False, encoding="utf-8")
    df_generated_tests_all_adapted_a3test_strategy.to_csv(path_generated_tests_all_adapted_a3test_strategy, index=False, encoding="utf-8")
    df_generated_tests_first_adapted_a3test_strategy.to_csv(path_generated_tests_first_adapted_a3test_strategy, index=False, encoding="utf-8")


# def get_all_tests_fixed(parserUtils: ParserUtils, list_tests_to_analize: list[str]):
#     list_tests_fixed = []
#     for test_to_analize in list_tests_to_analize:
#         if test_to_analize is None:
#             list_tests_fixed.append("")
#             continue

#         code_before_changes = test_to_analize
#         code_before_changes_has_errors, code_fixed = parserUtils.fix_missings_in_code(code_before_changes)

#         if not code_before_changes_has_errors:
#             list_tests_fixed.append(parserUtils.adapt_name_and_modifiers(code_before_changes))
#             continue

#         code_after_fixed_has_errors = parserUtils.validate_if_code_has_errors(code_before_changes)
#         if not code_after_fixed_has_errors:
#             list_tests_fixed.append(parserUtils.adapt_name_and_modifiers(code_fixed))
#             continue
    
#     return list_tests_fixed


def get_fixed_tests(parserUtils: ParserUtils, list_tests_to_analize: list[str]):
    count_empty_tests = 0
    
    list_all_fixed_tests_ast_strategy = []
    first_test_case_not_wrong_ast_strategy = list_tests_to_analize[0]
    is_there_already_a_fixed_test_ast_strategy = False
    some_test_has_error_after_adapt_name_ast_strategy = False

    list_all_adapted_tests_a3test_strategy = []
    first_adapted_test_is_correct_a3test_strategy = False
    is_there_already_a_fixed_test_a3test_strategy = False
    some_test_has_error_after_adapt_a3test_strategy = False

    # print("\n\n\n")
    # print(f"Size: {len(list_tests_to_analize)}")
    # print(list_tests_to_analize[-1])
    # print("\n\n\n")

    i = 0
    for test_to_analize in list_tests_to_analize:
        i += 1

        try:
            if test_to_analize is None or test_to_analize.strip() == "" or test_to_analize.strip() == "nan":
                count_empty_tests += 1
                list_all_fixed_tests_ast_strategy.append("")
                list_all_adapted_tests_a3test_strategy.append("")
                continue
        except Exception as ex:
            print(f"\nError empty generated_test {i}:")
            print(f"|{test_to_analize}|\n")
            raise ex

        code_before_changes = test_to_analize

        code_adpated_a3test_strategy = postProcessingWithA3TestStrategy(copy.deepcopy(code_before_changes))
        list_all_adapted_tests_a3test_strategy.append(code_adpated_a3test_strategy)

        code_after_adapted_has_errors_a3test_strategy = parserUtils.validate_if_code_has_errors(code_adpated_a3test_strategy)
        if code_after_adapted_has_errors_a3test_strategy:
            some_test_has_error_after_adapt_a3test_strategy = True
        else:
            is_there_already_a_fixed_test_a3test_strategy = True
            if i == 1:
                first_adapted_test_is_correct_a3test_strategy = True

        
        code_before_changes_has_errors = parserUtils.validate_if_code_has_errors(code_before_changes)
        # Para facilitar el recorrido de bytes, eliminamos tabs y saltos de línea primero
        # code_before_changes_normalized = re.sub(r'[\n\t]+', ' ', copy.deepcopy(code_before_changes))
        code_before_changes_normalized = code_before_changes.strip()
        code_before_changes_normalized = get_utf8_decoded_text(code_before_changes_normalized)


        if not code_before_changes_has_errors:
            test_with_adapted_name_ast_strategy = parserUtils.adapt_name_and_modifiers(code_before_changes_normalized)
            some_test_has_error_after_adapt_name_ast_strategy = parserUtils.validate_if_code_has_errors(test_with_adapted_name_ast_strategy)
            if some_test_has_error_after_adapt_name_ast_strategy:
                print(f"\n\n\nCode before changes (ast_strategy) {i}:")
                print(code_before_changes_normalized)
                print("\n\nCode before changes with name adapted using ast_strategy:")
                print(test_with_adapted_name_ast_strategy)
                print("\n")
                raise "Error after adapt name without fix using ast_strategy"

            list_all_fixed_tests_ast_strategy.append(test_with_adapted_name_ast_strategy)

            if not is_there_already_a_fixed_test_ast_strategy:
                first_test_case_not_wrong_ast_strategy = test_with_adapted_name_ast_strategy
                is_there_already_a_fixed_test_ast_strategy = True
            continue
        

        code_fixed_ast_strategy = parserUtils.fix_missings_and_errors_in_code(code_before_changes_normalized)
        try:
            code_after_fixed_has_errors_ast_strategy = parserUtils.validate_if_code_has_errors(code_fixed_ast_strategy)
        except Exception as ex:
            print(f"\nError validando:")
            print(f"code_before_changes_normalized: \n{code_before_changes_normalized}")
            print(f"\ncode_fixed_ast_strategy: \n{code_fixed_ast_strategy}\n\n")
            raise ex

        if not code_after_fixed_has_errors_ast_strategy:
            test_with_adapted_name_ast_strategy = parserUtils.adapt_name_and_modifiers(code_fixed_ast_strategy)
            some_test_has_error_after_adapt_name_ast_strategy = parserUtils.validate_if_code_has_errors(test_with_adapted_name_ast_strategy)
            if some_test_has_error_after_adapt_name_ast_strategy:
                print(f"\n\n\nCode fixed (ast_strategy) {i}:")
                print(code_fixed_ast_strategy)
                print("\n\nCode fixed with name adapted using ast_strategy:")
                print(test_with_adapted_name_ast_strategy)
                print("\n")
                raise "Error after adapt name after fix using ast_strategy"

            list_all_fixed_tests_ast_strategy.append(test_with_adapted_name_ast_strategy)

            if not is_there_already_a_fixed_test_ast_strategy:
                first_test_case_not_wrong_ast_strategy = test_with_adapted_name_ast_strategy
                if len(first_test_case_not_wrong_ast_strategy) == 0:
                    print(f"\n\nEmpty prev ({i}): {code_fixed_ast_strategy}")
                    print(f"\n\ncode_before_changes: \n{code_before_changes}")
                    print(f"\n\ncode_before_changes_normalized: \n{code_before_changes_normalized}")
                    raise "Error after adapt name after fix using ast_strategy is empty"
                is_there_already_a_fixed_test_ast_strategy = True
            continue


        list_all_fixed_tests_ast_strategy.append(code_before_changes)
    
    # if (not first_adapted_test_is_correct_a3test_strategy
    #     and is_there_already_a_fixed_test_ast_strategy):
    #     print(f"\n\n\nOriginal: \n{list_tests_to_analize[0]}")
    #     print(f"\nAST Strategy: \n{first_test_case_not_wrong_ast_strategy}")
    #     print(f"\nA3Test Strategy: \n{list_all_adapted_tests_a3test_strategy[0]}")

    # if (first_adapted_test_is_correct_a3test_strategy
    #     and not is_there_already_a_fixed_test_ast_strategy):
    #     print(f"\n\n\nOriginal: \n{list_tests_to_analize[0]}")
    #     print(f"\nAST Strategy: \n{first_test_case_not_wrong_ast_strategy}")
    #     print(f"\nA3Test Strategy: \n{list_all_adapted_tests_a3test_strategy[0]}")
    
    # if (len(first_test_case_not_wrong_ast_strategy) == 0):
    #     print(f"\n\n\nOriginal: \n|{list_tests_to_analize[0]}|")
    #     print(f"\nA3Test Strategy: \n|{list_all_adapted_tests_a3test_strategy[0]}|")
    #     print(f"\nAST first: \n|{list_all_fixed_tests_ast_strategy[0]}|")
    #     print(f"\nFirst not wrong: \n|{first_test_case_not_wrong_ast_strategy}|")
    #     print(f"\nFixed AST: \n|{is_there_already_a_fixed_test_ast_strategy}|")
    #     print(f"\nCounts empty: \n|{count_empty_tests}|\n\n\n")
    
    return (count_empty_tests, 
            list_all_fixed_tests_ast_strategy, 
            first_test_case_not_wrong_ast_strategy, 
            is_there_already_a_fixed_test_ast_strategy, 
            some_test_has_error_after_adapt_name_ast_strategy,
            list_all_adapted_tests_a3test_strategy,
            first_adapted_test_is_correct_a3test_strategy,
            is_there_already_a_fixed_test_a3test_strategy,
            some_test_has_error_after_adapt_a3test_strategy)



def get_utf8_decoded_text(code_before_decode: str):
    # try:
    #     # Vamos ignorando caracteres especiales
    #     # https://docs.python.org/es/3.13/howto/unicode.html
    #     # https://docs.python.org/es/3.13/library/codecs.html#standard-encodings
    #     # Codificación Latin-1 con \uXXXX y \UXXXXXXXX para otros puntos de código.
    #     code_before_changes_normalized = code_before_changes_normalized.encode("raw_unicode_escape").decode("utf-8", "ignore")
    # except Exception as ex:
    #     try:
    #         # Europa Occidental
    #         code_before_changes_normalized = code_before_changes_normalized.encode("cp1252").decode("utf-8", "ignore")
    #     except Exception as ex2:
    #         try:
    #             # Chino tradicional
    #             code_before_changes_normalized = code_before_changes_normalized.encode("big5").decode("utf-8", "ignore")
    #         except Exception as ex3:
    #             # Chino simplificado
    #             code_before_changes_normalized = code_before_changes_normalized.encode("hz").decode("utf-8", "ignore")

    code_before_changes_normalized = code_before_decode
    has_encoded_problem = False
    try:
        # Vamos ignorando caracteres especiales
        # https://docs.python.org/es/3.13/howto/unicode.html
        # https://docs.python.org/es/3.13/library/codecs.html#standard-encodings

        # Europa Occidental
        code_before_changes_normalized = code_before_changes_normalized.encode("cp1252").decode("utf-8", "ignore")
    except Exception as ex1:
        has_encoded_problem = True
        try:
            # Chino tradicional
            code_before_changes_normalized = code_before_changes_normalized.encode("big5").decode("utf-8", "ignore")
        except Exception as ex2:
            try:
                # Chino simplificado
                code_before_changes_normalized = code_before_changes_normalized.encode("hz").decode("utf-8", "ignore")
            except Exception as ex3:
                # Codificación Latin-1 con \uXXXX y \UXXXXXXXX para otros puntos de código.
                code_before_changes_normalized = code_before_changes_normalized.encode("raw_unicode_escape").decode("utf-8", "ignore")
    
    return code_before_changes_normalized if has_encoded_problem else code_before_decode


def export_results_json(data, file_path: str):
    """
    Exports data as json file
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        data_json = json.dumps(data, indent = 4)
        f.write(data_json)


def parse_args():
    """
    Parse the args passed from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_generated_tests", 
        type=str, 
        default="E:/000_Tesis/d4j_results_benchmark/d4j_generated_tests_only_public_bs/finetuning_01_codet5p_src_fm_fc_dctx_no_apply_fix_multiple_results_10",
        help="Folder where files generated_tests are located",
    )
    parser.add_argument(
        "--project_id",
        type=str,
        default="all",
        help="Project Id to Post Processing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="E:/000_Tesis/d4j_results_benchmark/d4j_generated_tests_only_public_bs/finetuning_01_codet5p_src_fm_fc_dctx_no_apply_fix_multiple_results_10",
        help="Path to the output folder",
    )

    return vars(parser.parse_args())



def main():
    args = parse_args()
    input_generated_tests = args['input_generated_tests']
    project_id = args['project_id']
    output = args['output']

    # list_projects_d4j = ['Csv', 'Cli', 'Lang', 'Chart', 'Gson']
    list_projects_d4j = ['Closure', 'Cli', 'Codec', 'Collections', 'Compress', 'Csv', 'JxPath', 'Lang', 'Math', 'Gson', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Chart', 'Time', 'Jsoup', 'Mockito']

    list_files_generated_tests = []
    if project_id == 'all':
        for project_df4 in list_projects_d4j:
            list_files_generated_tests.append(project_df4 + "_generated_tests")
    elif project_id in list_projects_d4j:
        list_files_generated_tests.append(project_id + "_generated_tests")
    else:
        raise 'Project_id not defined'
    

    start_time = time.time()

    for file_generated_tests in list_files_generated_tests:
        post_processing_project(input_generated_tests, file_generated_tests, output)
    
    end_time = time.time()
    
    print("\n\nSummary by Project:")
    for key_project, values_map in summary_by_project.items():
        print(f"\n{key_project}: {values_map}")
    
    output_file_summary = os.path.join(output, "all_summary_post_processing.json")
    export_results_json(summary_by_project, output_file_summary)


    syntax_summary_totals = {
        'project': 'Total',
        'total_tests': all_total_tests,
        'first_original_syntax_corrects': total_first_original_syntax_corrects,
        'first_original_rate_syntax_corrects': "{:.2f}".format((total_first_original_syntax_corrects / all_total_tests) * 100),
        'first_original_syntax_errors': total_first_original_syntax_errors,
        'first_original_rate_syntax_errors': "{:.2f}".format((total_first_original_syntax_errors / all_total_tests) * 100),
        'first_a3test_strategy_syntax_corrects': total_first_a3test_strategy_syntax_corrects,
        'first_a3test_strategy_rate_syntax_corrects': "{:.2f}".format((total_first_a3test_strategy_syntax_corrects / all_total_tests) * 100),
        'first_a3test_strategy_syntax_errors': total_first_a3test_strategy_syntax_errors,
        'first_a3test_strategy_rate_syntax_errors': "{:.2f}".format((total_first_a3test_strategy_syntax_errors / all_total_tests) * 100),
        'first_ast_strategy_syntax_corrects': total_first_ast_strategy_syntax_corrects,
        'first_ast_strategy_rate_syntax_corrects': "{:.2f}".format((total_first_ast_strategy_syntax_corrects / all_total_tests) * 100),
        'first_ast_strategy_syntax_errors': total_first_ast_strategy_syntax_errors,
        'first_ast_strategy_rate_syntax_errors': "{:.2f}".format((total_first_ast_strategy_syntax_errors / all_total_tests) * 100)
    }

    list_summary_syntax_report.append(syntax_summary_totals)
    df_summary_syntax_report = pd.DataFrame(list_summary_syntax_report)

    path_summary_syntax_report = os.path.join(output, "all_summary_syntax_report.csv")
    df_summary_syntax_report.to_csv(path_summary_syntax_report, index=False)
    
    
    elapsed_time = end_time - start_time
    elapsed_timeformatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("\n\n")
    print(f"Total time seconds: {elapsed_time}")
    print(f"Total time formatted: {elapsed_timeformatted}\n")


if __name__ == '__main__':
    main()
