import pandas as pd
import re
from tqdm import tqdm
#import javalang

#df_asserts = pd.read_csv("assertLines.txt")
#df_methods = pd.read_csv("testMethods.txt")


def format_code(code):
    #formatted_method = method.replace(' (', '(').replace(' )', ')').replace(' ,', ',')
    formatted_code = code.replace(' ++ ', '++').replace(' -- ', '--').replace(' ** ', '**').replace(' . ', '.').replace(' :: ', '::').replace('( ', '(').replace(' (', '(').replace(' )', ')').replace(' ;', ';').replace(' ,', ',').replace('[ ', '[').replace(' [', '[').replace(' ]', ']')
    formatted_code = formatted_code.replace('catch(', 'catch (').replace('try(', 'try (').replace('if(', 'if (').replace('for(', 'for (').replace('while(', 'while (').replace(' < ? > ', '<?> ').replace(' @ ', ' @')
    formatted_code = formatted_code.replace('}(', '} (').replace(')(', ') (').replace('=(', '= (').replace('+(', '+ (').replace('-(', '- (').replace('*(', '* (').replace('/(', '/ (').replace(';(', '; (').replace(':(', ': (').replace('?(', '? (').replace('>(', '> (').replace('<(', '< (').replace('&&(', '&& (').replace('||(', '|| (')
    return formatted_code


def combine_files():
    with open('testMethods.txt', 'r', encoding='utf-8') as methods_file, open('assertLines.txt', 'r', encoding='utf-8') as asserts_file:
        methods = methods_file.readlines()
        asserts = asserts_file.readlines()

    if len(methods) != len(asserts):
        print("Los archivos no tienen la misma cantidad de líneas.")
        return

    with open('combined.txt', 'w', encoding='utf-8') as output_file:
        for method, assert_ in tqdm(zip(methods, asserts), total=len(methods), desc="Combinando archivos"):
            replaced_placeholder = method.replace('"<AssertPlaceHolder>"', assert_.strip())
            formatted_code = format_code(replaced_placeholder)
            output_file.write(formatted_code)
            #output_file.write(method.replace('"<AssertPlaceHolder>"', assert_.strip()))

'''
def format_java_code():
    with open('combined.txt', 'r') as input_file:
        lines = input_file.readlines()

    with open('formatted.txt', 'w') as output_file:
        for line in tqdm(lines, desc="Formateando datos"):
            #methods = re.findall(r'(\w+ \(.+?\) \{.+?\})', line)
            #methods = re.findall(r'(\w+ \(.+?\) \{[^{}]*\})', line)
            method_tuples = re.findall(r'(\b(?!(if|for|while|switch|try|catch)\b)\w+ \(.+?\) \{[^{}]*\})', line)
            #method_tuples = re.findall(r'(\b(?!(if|for|while|switch|try|catch)\b)\w+ \(.+?\) \{.+?\})', line)
            print("Len methods: " + str(len(method_tuples)))
            for i, method_tuple in enumerate(method_tuples):
                method = method_tuple[0]
                formatted_method = format_code(method)
                tag = 'testMethod' if i == 0 else 'focalMethod'
                output_file.write('<' + tag + '>') # output_file.write('<' + tag + '>\n')
                if tag == 'testMethod':
                    output_file.write('@Test public void ') # output_file.write('@Test\n')
                output_file.write(formatted_method) # output_file.write('public void ' + formatted_method + '\n')
                output_file.write('</' + tag + '>') # output_file.write('</' + tag + '>\n')
            output_file.write('\n')
'''

combine_files()
#format_java_code()












