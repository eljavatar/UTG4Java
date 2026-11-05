import copy
import re
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node
from typing import List, Dict, Any, Set, Optional
from packaging import version
# https://setuptools.pypa.io/en/latest/pkg_resources.html
import pkg_resources


LIST_ERRORS = [
    "modifiers",
    "identifier",
    "type_identifier",
    "type_parameters",
    "formal_parameters",
    "field_access",
    "method_invocation",
    "assignment_expression",
    "binary_expression",
    "string_literal",
    "decimal_integer_literal",
    "decimal_floating_point_literal",
    "scoped_type_identifier",
    "null_literal",
    "class_literal",
    "true",
    "false",
    "final",
    "new",
    "for",
    "do",
    "while",
    "if",
    "switch",
    "assert",
    '"',
    '.',
    ",",
    ";",
    ":",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "...",
    "+",
    "-",
    "*",
    "/",
    "=",
    "++",
    "--",
    "+=",
    "-=",
    "*=",
    "/=",
    "==",
    "&",
    "&&",
    "|",
    "||",
    "<=",
    ">=",
    "!",
    "->",
    "%"
]


class ParserUtils():

    def __init__(self, input_encoding):
        #print(f"\n\n Version tree-sitter: {str(tree_sitter.__version__)} \n\n")
        tree_sitter_version = pkg_resources.get_distribution("tree_sitter").version
        #print(f"\n\n Version tree-sitter: '{str(tree_sitter_version)}' \n\n")

        if version.parse(str(tree_sitter_version)) < version.parse("0.22.0"):
            JAVA_LANGUAGE = Language(tsjava.language(), "java")
        else:
            JAVA_LANGUAGE = Language(tsjava.language())
        
        if version.parse(str(tree_sitter_version)) < version.parse("0.22.1"):
            self.parser = Parser()
            self.parser.set_language(JAVA_LANGUAGE)
        else:
            self.parser = Parser(JAVA_LANGUAGE)

        self.input_encoding = input_encoding


    # def fix_missings_and_errors_in_code(self, src_code: str):
    #     # # Para facilitar el recorrido de bytes, eliminamos tabs y saltos de línea primero
    #     # src_code_normalized = re.sub(r'[\n\t]+', ' ', src_code)

    #     src_code_decorated = "public class ClassTest { " + src_code + " }"

    #     code_without_changes_has_errors, code_missings_fixed = self.fix_missings_in_code(src_code_decorated)

    #     if not code_without_changes_has_errors:
    #         return False, src_code
        
    #     code_missings_fixed_has_errors, code_errors_fixed = self.fix_errors_in_code(code_missings_fixed)

    #     if not code_missings_fixed_has_errors:
    #         list_body_methods: set[str] = self.get_body_methods_from_class(code_missings_fixed)
    #         return True, " ".join(list_body_methods) # A pesar de que fue fixeado, el código inicial era erróneo, por eso devuelve True
    #         # return True, code_missings_fixed # A pesar de que fue fixeado, el código inicial era erróneo, por eso devuelve True
        
    #     list_body_methods: set[str] = self.get_body_methods_from_class(code_errors_fixed)
    #     return True, " ".join(list_body_methods)
    #     # return True, code_errors_fixed
    

    def clean_multiple_spaces(self, src_code):
        src_code = re.sub(r'\s{2,}', ' ', src_code)
        return src_code
    

    def get_body_methods_from_class_str(self, src_code_fixed, src_code_default):
        list_body_methods: set[str] = self.get_body_methods_from_class(src_code_fixed)
        if len(list_body_methods) == 0:
            return src_code_default
        # print(list_body_methods)
        src_fixed_errors = " ".join(list_body_methods)
        return self.clean_multiple_spaces(src_fixed_errors)
    

    def fix_missings_and_errors_in_code(self, src_code: str):
        # print(f"\nWithout decorate: \n{src_code}")

        src_code_decorated = "public class ClassTest {\n " + src_code + " \n}"

        # code_before_apply_fix_error_has_only_missings = False
        # code_before_apply_fix_error_already_has_syntax_error = False

        # print(f"\nBefore changes: \n{src_code_decorated}\n\n")


        for i in range(3): # Cantidad de veces que intentamos corregir errores antes de corregir missings
            # print(f"\n\n\nIteration {i}:")
            (_, 
             code_before_apply_fix_error_has_missings, 
             code_before_apply_fix_error_has_only_errors_no_missings, 
             code_errors_fixed) = self.fix_errors_in_code(src_code_decorated)

            if len(code_errors_fixed.strip()) == 0:
                code_errors_fixed = src_code_decorated
                # break
            
            code_after_errors_fixed_has_syntax_error = self.validate_if_code_has_errors(code_errors_fixed)

            if not code_after_errors_fixed_has_syntax_error:
                # print(f"\nAfter fix {i} error (fixed): \n{code_errors_fixed}")
                return self.get_body_methods_from_class_str(code_errors_fixed, src_code)
                # list_body_methods: set[str] = self.get_body_methods_from_class(code_errors_fixed)
                # print(list_body_methods)
                # src_fixed_errors = " ".join(list_body_methods)
                # return self.clean_multiple_spaces(src_fixed_errors)

            # print(f"\nAfter fix {i} error (not fixed): \n{code_errors_fixed}")
            
            # code_before_apply_fix_error_already_has_syntax_error = True
            src_code_decorated = src_code_decorated if len(code_errors_fixed.strip()) == 0 else code_errors_fixed
        

        # Si llega hasta aquí, es porque el código aún tiene errores
        # Intentamos solucionar los missings
        # src_code_decorated = re.sub(r'\s{2,}', ' ', src_code_decorated)

        _, code_missings_fixed = self.fix_missings_in_code(src_code_decorated)
        code_after_missings_fixed_has_syntax_error = self.validate_if_code_has_errors(code_missings_fixed)

        if not code_after_missings_fixed_has_syntax_error:
            # print(f"\nAfter fix missings: {code_missings_fixed}")
            return self.get_body_methods_from_class_str(code_missings_fixed, src_code)
            # list_body_methods: set[str] = self.get_body_methods_from_class(code_missings_fixed)
            # src_fixed_missings = " ".join(list_body_methods)
            # return self.clean_multiple_spaces(src_fixed_missings)
        
        # print(f"\nNot apply fix: {src_code}")

        # Si llega hasta aquí, es porque ninguno de los fix aplicados solucionaron
        # los errores de sintaxis, por tanto devlvemos el c'digo original

        # list_body_methods: set[str] = self.get_body_methods_from_class(code_missings_fixed)
        # No fue posible aplicar fix_error, ni fix_missings, devolvemos el código original
        # return True, " ".join(list_body_methods)
        return src_code


    def fix_missings_in_code(self, src_code: str):
        src_encoded_bytes = bytes(src_code, self.input_encoding)
        src_decoded = src_encoded_bytes.decode(self.input_encoding)
        #print("\n" + str(src_decoded))

        tree = self.parser.parse(src_encoded_bytes)

        edited_code = src_decoded

        content_node_main = tree.root_node
        code_without_changes_has_errors = content_node_main.has_error

        if not code_without_changes_has_errors:
            return code_without_changes_has_errors, src_code
        #print("Has errors: " + str(content_node_main.has_error))
        #print(content_node_main.sexp())

        #list_missing_nodes = []
        #print(f"\nLength missings: {str(len(list_missing_nodes))}")

        list_missing_nodes: List[Node] = []
        ParserUtils.traverse_tree_missing(content_node_main, list_missing_nodes)
        #print(f"\nLength missings: {str(len(list_missing_nodes))}")

        length_aument = 0
        for missing_node in list_missing_nodes:
            #print("Node: " + str(missing_node))
            #print("Type: " + str(missing_node.type))

            #new_end_point_with_missing = (missing_node.start_point[0], missing_node.start_point[1] + len(str(missing_node.type)))
            #missing_node.edit(
            #    start_byte=missing_node.start_byte,
            #    old_end_byte=missing_node.end_byte,
            #    new_end_byte=missing_node.start_byte, 
            #    start_point=missing_node.start_point,
            #    old_end_point=missing_node.end_point,
            #    new_end_point=missing_node.start_point,
            #    #new_end_point=new_end_point_with_missing
            #)
            
            edited_code = edited_code[: missing_node.start_byte + length_aument] + str(missing_node.type) + edited_code[missing_node.start_byte + length_aument :]
            length_aument += len(str(missing_node.type))

        return code_without_changes_has_errors, edited_code.strip()


    def fix_errors_in_code(self, src_code: str):
        src_encoded_bytes = bytes(src_code, self.input_encoding)
        src_decoded = src_encoded_bytes.decode(self.input_encoding)
        #print("\n" + str(src_decoded))

        tree = self.parser.parse(src_encoded_bytes)

        edited_code = src_decoded

        content_node_main = tree.root_node
        code_without_changes_has_errors = content_node_main.has_error

        if not code_without_changes_has_errors:
            return code_without_changes_has_errors, False, None, src_code

        list_missing_nodes: List[Node] = []
        ParserUtils.traverse_tree_missing(content_node_main, list_missing_nodes)
        code_without_changes_has_missings = len(list_missing_nodes) > 0
        
        list_only_errors_nodes: List[Node] = []
        ParserUtils.traverse_tree_only_errors(content_node_main, list_only_errors_nodes)
        code_without_changes_has_only_errors_no_missings = len(list_only_errors_nodes) > 0

        # if not code_without_changes_has_only_errors_no_missings:
        #     print("Return here")
        #     return True, code_without_changes_has_missings, False, src_code

        # if content_node_main.type == 'ERROR':
        #     print("All code is ERROR")
        
        range_node_main = content_node_main.range
        not_error_detected = False
        # print(range_node_main)
        length_reduced = 0
        # prev_node_deleted = None
        prev_nodes_deleted = set()

        for error_node in list_only_errors_nodes:
            if content_node_main.type == 'ERROR':
                break

            parent_node = error_node.parent
            range_error_node = error_node.range
            # print(f"error_node range: {range_error_node}")

            if (parent_node is None 
                or parent_node.type == 'program' 
                or parent_node.type == 'class_declaration'
                or parent_node.type == 'method_declaration'
                or range_node_main == range_error_node):
                # Tree-sitter no puede encontrar el error específico y marca todo el código como erróneo
                # print("Not specific error detected")
                not_error_detected = True
                break

            # parent_node = error_node.parent
            if parent_node.type != 'block':
                # node_parent_before_block = ParserUtils.get_parent_node_before_block(parent_node)
                # parent_node = ParserUtils.get_parent_node_before_block(parent_node)

                if (parent_node in prev_nodes_deleted):
                    # Este Nodo ya había sido eliminado, no lo volvemos a eliminar
                    # print("Nodo eliminado previamente")
                    continue
                # Removemos toda la sentencia del error
                error_text = parent_node.text.decode(self.input_encoding)
                edited_code = edited_code[: parent_node.start_byte - length_reduced] + edited_code[parent_node.end_byte - length_reduced :]
                length_reduced += len(error_text)
                # print(f"Error detected in block ({parent_node.type}): {error_text}")
                prev_nodes_deleted.add(parent_node)
                continue

            prev_node = error_node.prev_sibling
            next_node = error_node.next_sibling

            if prev_node.type == "{" and next_node.type == "}":
                # Si todo el bloque de código es error, no hacemos nada
                # print("Todo el bloque es error")
                break

            # Ya validamos que no estamos eliminando todo el body, podemos eliminar el nodo del error completo
            error_text = error_node.text.decode(self.input_encoding)
            edited_code = edited_code[: error_node.start_byte - length_reduced] + edited_code[error_node.end_byte - length_reduced :]
            length_reduced += len(error_text)
            
            # sprint(str(error_node))
            # if len(error_node.children) > 0:
            #     for error_detail_node in error_node.children:
            #         # print("")
            #         print(error_detail_node)
            #         error_text = error_detail_node.text.decode(self.input_encoding)
            #         print("Type error: " + str(error_detail_node.type) + " -> [" + str(error_text) + "]")
            #         # if error_detail_node.type in LIST_ERRORS or error_text.strip() == "'":
            #         if error_detail_node.type in LIST_ERRORS or error_text.strip() == "'":
            #             # error_text = error_detail_node.text.decode(self.input_encoding)
            #             edited_code = edited_code[: error_detail_node.start_byte - length_reduced] + edited_code[error_detail_node.end_byte - length_reduced :]
            #             length_reduced += len(error_text)
        
        if content_node_main.type == 'ERROR' or not_error_detected or len(list_only_errors_nodes) == 0:
            # print(f"Not error detected: {edited_code}")
            # len(list_only_errors_nodes) == 0 Ocurre cuando no se detecta error dentro
            # de main_node, sino que ERROR es el main_node
            code_without_changes_has_only_errors_no_missings = True

            # Tratamos de balancear la apertura y cierre de corchetes
            list_open_brackets_nodes: List[Node] = []
            ParserUtils.traverse_type(content_node_main, list_open_brackets_nodes, "{")

            list_close_brackets_nodes: List[Node] = []
            ParserUtils.traverse_type(content_node_main, list_close_brackets_nodes, "}")

            if len(list_open_brackets_nodes) != len(list_close_brackets_nodes):
                # Tratamos de encontrar el último ";"
                list_semicolon_nodes: List[Node] = []
                ParserUtils.traverse_type(content_node_main, list_semicolon_nodes, ";")
                # for x in reversed(a):
                if len(list_semicolon_nodes) > 0:
                    last_semicolon_node = list_semicolon_nodes[-1]
                    edited_code = edited_code[: last_semicolon_node.start_byte] + "; } }"
                # for node in list_semicolon_nodes[::-1]:
                #     print(f"{node} -> {node.byte_range} -> {node.range}")

        
        # Reemplaza múltiples espacios por un solo espacio
        # edited_code = re.sub(r'\s{2,}', ' ', edited_code)
        # print(f"\n\nFinal:\n{edited_code}\n\n")
        
        try:
            edited_code = edited_code.encode(self.input_encoding).decode("utf-8")
        except:
            pass

        if len(edited_code.strip()) == 0:
            edited_code = src_code

        # print(f"edited_code: {edited_code.strip()}")
        
        return (code_without_changes_has_errors, 
                code_without_changes_has_missings, 
                code_without_changes_has_only_errors_no_missings, 
                edited_code)


    def adapt_name_and_modifiers(self, src_code: str):
        """
        Se eliminan todos los modificadores y se asignan de nuevo,
        esto se hace para soportar distintas versiones de JUnit.

        Por ejemplo, esta declaracion:
        @Test void someTest() {} # Pasa en JUnit 5, pero no en JUnit 4 ni JUnit 3

        Para que pase en JUnit 4, hay que agregarle el modificador public:
        @Test public void someTest() {} # Pasa en JUnit 5 y en JUnit 4, pero no en JUnit 3

        Para que pase en JUnit 3, hay que agregarle la palabra test como prefijo al nombre del método:
        @Test public void testsomeTest() {} # Pasa en JUnit 5 y en JUnit 4, pero no en JUnit 3

        Además, previene errores causados por nombres de métodos que coinciden con métodos de la clase Object
        Por ejemplo, si se genera un método con el nombre toString(), para prevenir conflictos con el
        método toString() de la clase Objects, lo renombramos a testToString()
        """
        # Para facilitar el recorrido de bytes, eliminamos tabs y saltos de línea primero
        src_code = re.sub(r'[\n\t]+', ' ', src_code)

        src_encoded_bytes = bytes(src_code, self.input_encoding)
        src_decoded = src_encoded_bytes.decode(self.input_encoding)

        clean_code = src_decoded

        tree = self.parser.parse(src_encoded_bytes)
        content_node_main = tree.root_node

        # length_reduced = 0
        # type_return = "void"
        # name_method = "some"
        # list_annotations = []
        # contains_test_annotation = False

        list_test_methods = []
        list_test_methods_names = set()

        # Clean all modifiers
        for child in content_node_main.children:
            if child.type == "method_declaration":
                # print("\n\n")
                meth_list_annotations = []
                meth_contains_test_annotation = False
                meth_test_clean_code = child.text.decode(self.input_encoding)
                meth_start_index = child.start_byte
                # meth_end_index = child.end_byte
                meth_length_reduced = meth_start_index

                # print(meth_test_clean_code)
                # print(f"[{meth_start_index}, {meth_end_index}]")
                # print("\n")

                for _child in child.children:
                    if _child.type == "modifiers":

                        for c in _child.children:
                            if c.type == 'marker_annotation' or c.type == 'annotation':
                                # annotation_text = c.text.decode(self.input_encoding)
                                # list_annotations.append(annotation_text.strip())
                                # meth_list_annotations.append(annotation_text.strip())
                                # if "@ParametrizedTest" in annotation_text or "@Test" in annotation_text:
                                    # # contains_test_annotation = True
                                    # meth_contains_test_annotation = True

                                annotation_text = c.text.decode(self.input_encoding)
                                for _c in c.children:
                                    if _c.type == 'identifier':
                                        name_annotation_text = _c.text.decode(self.input_encoding)
                                        if name_annotation_text == "Test" and not meth_contains_test_annotation:
                                            annotation_text_normalized = re.sub(r'\s{2,}', ' ', annotation_text.strip())
                                            if annotation_text_normalized == "@Test()" or annotation_text_normalized == "@Test( )":
                                                annotation_text = "@Test"
                                            meth_list_annotations.append(annotation_text.strip())
                                            meth_contains_test_annotation = True
                                            break
                        
                        modifiers_text = _child.text.decode(self.input_encoding)

                        # clean_code = clean_code[: _child.start_byte - length_reduced] + clean_code[_child.end_byte - length_reduced :]
                        meth_test_clean_code = meth_test_clean_code[: _child.start_byte - meth_length_reduced] + meth_test_clean_code[_child.end_byte - meth_length_reduced :]
                        # print("Sin modifiers: " + meth_test_clean_code)
                        # length_reduced += len(modifiers_text)
                        meth_length_reduced += len(modifiers_text)
                        
                        # print(clean_code)
                        # print()
                        continue
                    
                    if _child.type == "type_parameters":
                        type_parameter = _child.text.decode(self.input_encoding)

                        # clean_code = clean_code[: _child.start_byte - length_reduced] + clean_code[_child.end_byte - length_reduced :]
                        meth_test_clean_code = meth_test_clean_code[: _child.start_byte - meth_length_reduced] + meth_test_clean_code[_child.end_byte - meth_length_reduced :]
                        # print("Sin type_parameter: " + meth_test_clean_code)
                        # length_reduced += len(type_parameter)
                        meth_length_reduced += len(type_parameter)

                        # print(clean_code)
                        # print()
                        continue
                        
                    if "type" in _child.type and _child.type != "type_parameters":
                        type_return = _child.text.decode(self.input_encoding)

                        # clean_code = clean_code[: _child.start_byte - length_reduced] + clean_code[_child.end_byte - length_reduced :]
                        meth_test_clean_code = meth_test_clean_code[: _child.start_byte - meth_length_reduced] + meth_test_clean_code[_child.end_byte - meth_length_reduced :]
                        # print("Sin type: " + meth_test_clean_code)
                        # length_reduced += len(type_return)
                        meth_length_reduced += len(type_return)
                        # print(clean_code)
                        # print()
                        continue
                    
                    if _child.type == "identifier":
                        name_method = _child.text.decode(self.input_encoding)

                        # clean_code = clean_code[: _child.start_byte - length_reduced] + clean_code[_child.end_byte - length_reduced :]
                        meth_test_clean_code = meth_test_clean_code[: _child.start_byte - meth_length_reduced] + meth_test_clean_code[_child.end_byte - meth_length_reduced :]
                        # print("Sin identifier: " + meth_test_clean_code)
                        # length_reduced += len(name_method)
                        meth_length_reduced += len(name_method)
                        # print(clean_code)
                        # print()
                        break
                
                
                if not name_method.startswith("test"):
                    name_method = name_method[0].upper() + name_method[1:]
                    name_method = "test" + name_method
                
                if name_method in list_test_methods_names:
                    name_method = name_method + "_2"
                    if name_method in list_test_methods_names:
                        name_method = name_method + "_3"
                list_test_methods_names.add(name_method)
                
                # meth_all_annotations_text = " ".join(meth_list_annotations)
                # Podríamos obtener las anotaciones con las que se generaron los tests
                # Pero hay casos en los que se pueden generar anotaciones como las siguientes que pueden causar error de compilación:
                # - @Test(dataProvider = "toShortDataProvider"
                # - @Test(groups = "unit")
                meth_all_annotations_text = " ".join(["@Test"])

                format_test_method = "{} public {} {}{}" if len(meth_all_annotations_text) > 0 else "{}public {} {}{}"
                meth_test_clean_code = format_test_method.format(meth_all_annotations_text, type_return, name_method, meth_test_clean_code.strip())

                # meth_all_annotations_text = " ".join(meth_list_annotations)
                # meth_test_clean_code = meth_all_annotations_text.strip() + " public " + type_return + " " + name_method + meth_test_clean_code.strip()

                # if not meth_contains_test_annotation:
                #     meth_test_clean_code = "@Test " + meth_test_clean_code
                
                list_test_methods.append(meth_test_clean_code)

        # print("\n\n\n")
        # for method in list_test_methods:
        #     print(method)
        #     print("")
        # print("\n\n")

        # if not name_method.startswith("test"):
        #     #print("Name before: " + name_method)
        #     name_method = name_method[0].upper() + name_method[1:]
        #     # capitalized_word = '{}{}'.format(word[0].upper(), word[1:])
        #     name_method = "test" + name_method
        #     #print("Name after: " + name_method)

        # all_annotations_text = " ".join(list_annotations)
        # clean_code = all_annotations_text.strip() + " public " + type_return + " " + name_method + clean_code.strip()

        # if not contains_test_annotation:
        #     clean_code = "@Test " + clean_code
        # #print("\n\n\n")

        clean_code = " ".join(list_test_methods)

        try:
            clean_code = clean_code.encode(self.input_encoding).decode("utf-8")
        except:
            pass
        
        return clean_code.strip()


    def validate_if_code_has_errors(self, src_code: str):
        src_encoded_bytes = bytes(src_code, self.input_encoding)
        #src_decoded = src_encoded_bytes.decode(self.input_encoding)
        #print("\n" + str(src_decoded))

        tree = self.parser.parse(src_encoded_bytes)
        content_node_main = tree.root_node

        #print("Has errors: " + str(content_node_main.has_error))

        return content_node_main.has_error
    

    def clean_comments(self, src_code: str):
        """
        Parses a java file and extract metadata using info in method_metadata
        """

        #src_encoded_bytes = src_code.encode(encoding="utf-8")
        src_encoded_bytes = bytes(src_code, self.input_encoding)
        src_decoded = src_encoded_bytes.decode(self.input_encoding)

        #try:
        #    src_decoded = src_encoded_bytes.decode("cp1252")
        #except:
        #    src_decoded = src_encoded_bytes.decode("utf-8")

        #src_decoded = src_encoded_bytes.decode("cp1252", errors="ignore")

        clean_code = src_decoded

        #Build Tree
        #tree = self.parser.parse(src_encoded_bytes)
        tree = self.parser.parse(src_encoded_bytes)
        #tree = self.parser.parse(bytes(src_code.encode(encoding="utf-8", errors="ignore").decode("utf-8"), "utf8"))
        #tree = self.parser.parse(bytes(src_code.encode(encoding=self.input_encode, errors="ignore").decode("utf-8"), "utf8"))
        content_node_main = tree.root_node

        #edited_code = src_decoded

        block_comments_nodes = []
        ParserUtils.traverse_type(content_node_main, block_comments_nodes, "block_comment")
        for block_comment_node in block_comments_nodes:
            #block_comment_str = ParserUtils.match_from_span(block_comment_node, src_code)
            block_comment_text = block_comment_node.text.decode(self.input_encoding)

            #if 'Coin.valueOf(-1234567890l)' in src_code:
            #    print("Block Coment: '" + block_comment_str + "'")
            clean_code = clean_code.replace(block_comment_text, " ")


        line_comments_nodes = []
        list_line_comments_str = []
        ParserUtils.traverse_type(content_node_main, line_comments_nodes, "line_comment")
        for line_comment_node in line_comments_nodes:
            #line_comment_str = ParserUtils.match_from_span(line_comment_node, src_decoded)
            line_comment_text = line_comment_node.text.decode(self.input_encoding)
            
            list_line_comments_str.append(line_comment_text)

        # Esto se hace para casos en los que se tienen líneas de comentarios como las siguientes:
        # // one comment
        # // another comment // one comment
        # En estos casos, si se elimina primero el comentario "// one comment", después no
        # sería posible encontrar el comentario "// another comment // one comment".
        # Por eso se ordenan los comentarios según la longitud de cada uno de ellos
        # de mayor a menor, de esta forma, se eliminaría primero "// another comment // one comment"
        # y después "// one comment"
        list_line_comments_order_desc_by_len = reversed(sorted(list_line_comments_str, key=lambda item: len(item)))
        for line_comment_to_clean in list_line_comments_order_desc_by_len:
            clean_code = clean_code.replace(line_comment_to_clean, " ")

        try:
            clean_code = clean_code.encode(self.input_encoding).decode("utf-8")
        except:
            pass
        
        return clean_code


    def remove_modifiers_annotations(self, src_code: str):
        src_encoded_bytes = bytes(src_code, self.input_encoding)
        src_decoded = src_encoded_bytes.decode(self.input_encoding)

        clean_code = src_decoded

        tree = self.parser.parse(src_encoded_bytes)
        content_node_main = tree.root_node

        for child in content_node_main.children:
            if child.type == "method_declaration":
                for _child in child.children:
                    if _child.type == "modifiers":
                        #metadata['modifiers']  = ' '.join(ParserUtils.match_from_span(child, blob).split())
                        # modifiers_text = _child.text.decode(self.input_encoding)
                        # print(modifiers_text)
                        # clean_code = clean_code.replace(modifiers_text, " ")

                        for c in _child.children:
                            if c.type == "marker_annotation" or c.type == "annotation":
                                annotation_text = c.text.decode(self.input_encoding)
                                #print(annotation_text)
                                clean_code = clean_code.replace(annotation_text, " ")
        
        #print("\n\n\n")
        try:
            clean_code = clean_code.encode(self.input_encoding).decode("utf-8")
        except:
            pass
        
        return clean_code
    

    def get_body_methods_from_class(self, src_code: str) -> set[str]:
        src_encoded_bytes = bytes(src_code, self.input_encoding)
        tree = self.parser.parse(src_encoded_bytes)
        content_node_main = tree.root_node

        method_declarations_nodes = []
        ParserUtils.traverse_type(content_node_main, method_declarations_nodes, "method_declaration") # constructor_declaration

        list_body_methods: set[str] = set()

        for method_declaration_node in method_declarations_nodes:
            body_method_text = method_declaration_node.text.decode(self.input_encoding)
            # clean_code = clean_code.replace(annotation_text, "")
            list_body_methods.add(body_method_text.strip())
        
        return list_body_methods
    

    @staticmethod
    def get_parent_node_before_block(node: Node):
        node_parent = node.parent
        # if node_parent is None:
        #     return node
        if node_parent.type == 'block':
            return node
        
        return ParserUtils.get_parent_node_before_block(node_parent)
    

    @staticmethod
    def traverse_tree_missing(node: Node, results: List):
        for n in node.children:
            if n.is_missing: 
                results.append(n)
            ParserUtils.traverse_tree_missing(n, results)


    @staticmethod
    def traverse_tree_only_errors(node: Node, results: List):
        for n in node.children:
            if n.is_error and not n.is_missing: 
                results.append(n)
            ParserUtils.traverse_tree_only_errors(n, results)
    

    @staticmethod
    def traverse_type(node: Node, results: List, kind: str) -> None:
        """
        Traverses nodes of given type and save in results
        """
        if node.type == kind:
            results.append(node)
        if not node.children:
            return
        for n in node.children:
            ParserUtils.traverse_type(n, results, kind)
    

    @staticmethod
    def traverse_text(node: Node, results: List, text_value: str, encoding: str) -> None:
        """
        Traverses nodes of given value and save in results
        """
        if node.text.decode(encoding) == text_value:
        # if node.type == kind:
            results.append(node)
        if not node.children:
            return
        for n in node.children:
            ParserUtils.traverse_text(n, results, text_value, encoding)


    @staticmethod
    def children_of_type(node, types):
        """
        Return children of node of type belonging to types

        Parameters
        ----------
        node : tree_sitter.Node
            node whose children are to be searched
        types : str/tuple
            single or tuple of node types to filter

        Return
        ------
        result : list[Node]
            list of nodes of type in types
        """
        if isinstance(types, str):
            return ParserUtils.children_of_type(node, (types,))
        return [child for child in node.children if child.type in types]
    

    @staticmethod
    def match_from_span(node, blob: str) -> str:
        """
        Extract the source code associated with a node of the tree
        """
        line_start = node.start_point[0]
        line_end = node.end_point[0]
        char_start = node.start_point[1]
        char_end = node.end_point[1]
        lines = blob.split('\n')
        if line_start != line_end:
            return '\n'.join([lines[line_start][char_start:]] + lines[line_start+1:line_end] + [lines[line_end][:char_end]])
        else:
            return lines[line_start][char_start:char_end]


# if __name__ == "__main__":
#     method = "@Timeout(1000) public <T> void toString() throws NoSuchFieldException { Field[] fields = Field.getDeclaredFields(); TypeInfo typeInfo = TypeInfoFactory.getTypeInfoForField(fields.get(0), String.class); assertThat(typeInfo.getActualType()).isEqualTo(String.class); assertThat(typeInfo.getType()).isEqualTo(String.class); }"
#     print("\nBefore:")
#     print(method)
#     print("\n\n\n")

#     parserUtils = ParserUtils("utf-8")
#     code_without_missings = parserUtils.adapt_name_and_modifiers(method)

#     print("\n\nAfter:")
#     print(code_without_missings)

parser_utils = ParserUtils("utf-8")

# src_code = "private<D> <D> InjectableMethod(@Nullable TypeLiteral <D> targetType, @Nullable D target, Method method, @Nullable T result , ); private < D > < D > InjectableMethod(@Nullable TypeLiteral<D> targetType); public boolean hasReturnValue(final String value, , Long num [)"
# print("\n")

# _, without_missings = parser_utils.fix_missings_in_code(src_code)
# print(without_missings)
# print("\n")
# print(parser_utils.fix_errors_in_code(without_missings))


# test_to_adapt_name = "@Test public void testWrite() throws IOException {\n final int b = 1; assertEquals(1, b); \n} public void testWrite() throws IOException { final int c = 2; assertEquals(1, c.getSize()); }"
# test_to_adapt_name = '@Test() @DisplayName("返回float的值") void testToFloat() { assertThat(NumberUtils.toFloat(null, 0)).isEqualTo(0); float f1 = 1.2f; float f2 = 2.3f; assertThat((float) NumberUtils.fromFloat("1.2", f1)).isCloseTo(f2, offset(EPS)); }'
# test_to_adapt_name = '@Test @DisplayName("删除字符串") public void testDeleteFirst() { StrBuilder builder = new StrBuilder(); builder.deleteFirst("abc"); Assertions.assertEquals("abc", builder.toString()); }'
# test_to_adapt_name = '﻿using System.Collections.Generic;using System.Linq;using JetBrains.Annotations; @Test public void testScopedAliases() { List<String> aliases = new ArrayList<>(); for (int i = 0; i < 10; i++) { aliases.add(""a"" + i); } AbstractCompiler compiler = Mockito.mock(AbstractCompiler.class); PreprocessorSymbolTable preprocessorSymbolTable = mock(PreprocessorSymbolTable.class); AliasTransformationHandler transformationHandler = mock(AliasTransformationHandler.class); ScopedAliases scope = new ScopedAliases(compiler, preprocessorSymbolTable, transformationHandler); Node root = mock(Node.class); when(root.getChildren()).thenReturn(aliases); scope.process(root, root); ArgumentCaptor<Node> aliasCaptor = ArgumentCaptor.forClass((Class) Node.class); verify(compiler, times(10)).hotSwapScript(aliasCaptor.capture(), eq(root)); assertEquals(aliases.size(), aliasCaptor.getValue().getChildren().size()); for (Node alias : aliasCaptOR.getAllValues()) { assertTrue(aliases.contains(alias.getText())); } }'

# try:
#     test_to_adapt_name = test_to_adapt_name.encode("raw_unicode_escape").decode("utf-8", "ignore")
#     # test_to_adapt_name = test_to_adapt_name.encode("latin_1").decode("utf-8", "ignore")
# except Exception as ex:
#     try:
#         # Europa Occidental
#         test_to_adapt_name = test_to_adapt_name.encode("cp1252").decode("utf-8", "ignore")
#     except Exception as ex2:
#         try:
#             # Chino tradicional
#             test_to_adapt_name = test_to_adapt_name.encode("big5").decode("utf-8", "ignore")
#         except Exception as ex3:
#             # Chino simplificado
#             test_to_adapt_name = test_to_adapt_name.encode("hz").decode("utf-8", "ignore")


# print(test_to_adapt_name)

# adapted = parser_utils.adapt_name_and_modifiers(test_to_adapt_name.strip())
# print("\n")
# print(adapted)


# src_code = '@Test public void long() throws IOException { JsonReader reader = new JsonReader(new StringReader("1")); assertEquals(1, reader.nextLong()); assertEquals(1, reader.nextLong()); assertEquals(1, reader.nextLong()); assertEquals(1, reader.nextLong()); assertEquals(1, reader'
# src_code = '@Test public void testEquals() { final OptionImpl option1 = new OptionImpl(1, true); final OptionImpl option36 = new OptionImpl(36,'
# src_code = '@Test public void testEquals() {final OptionImpl option1 = new OptionImpl(1, true); assertEquals("'
# src_code = "@Test public void testEquals() {final OptionImpl option1 = new OptionImpl(1, true); assertEquals('"
# src_code = '@Test public void testAddOption() { Options options = new Options(); options.addOption(new Option("a", "a", false, "a")); options.addOption(new Option("b", "b", false, "b")); options.addOption(new Option("z", "z", false, "z")); options'
# src_code = '@Test public void testAddOption() { OptionGroup optionGroup = new OptionGroup(); optionGroup.addOption(new Option("opt1", true, "description1")); optionGroup.addOption(new Option("opt26", true, "description26'
# src_code = '@Test public void testHelpLines() { final Set helpSettings = new HashSet(); helpSettings.add(DisplaySetting.DISPLAY_GROUP_NAME); helpSettings.add(DisplaySetting.DISPLAY_GROUP_ARGUMENT); help'
# src_code = '@Test public void testAppendUsage() { final StringBuffer buffer = new StringBuffer(); final Set helpSettings = new HashSet(); helpSettings.add(DisplaySetting.DISPLAY_OPTIONAL); helpSettings.add(DisplaySetting.'
# src_code = '@Test public void testAppendUsage() { final StringBuffer buffer = new StringBuffer(); final Set helpSettings = new HashSet(); helpSettings.add(DisplaySetting.DISPLAY_OPTIONAL); helpSettings.add(DisplaySetting.DISPLAY'
# src_code = '@Test public void testCanProcess() { assertTrue(group.canProcess(commandLine, "arg1")); assertTrue(group.canProcess(commandLine, "arg2")); assertFalse(group.canProcess(commandLine, "arg31")); assertFalse(group.'
# src_code = "@Test public void testRead() throws IOException {ExtendedBufferedReader reader = new ExtendedBufferedReader(new StringReader(new String(new char[] {'a', 'b', 'c', 'c', 'd', "
# src_code = "@Test public void testPrintComment() throws IOException { final CSVFormat format = new CSVFormat(',', '\\', '\\',"
# src_code = '@Test public void testIsQuoteCharacterSet() { assertThat(CSVFormat.of(",", ",", ",'
# src_code = "@Test public void testPrintComment() throws IOException { final CSVFormat format = new CSVFormat(',', '\\\\', '\\\\', '\\n"
# src_code = '@Test public void testEncodeBase64URLSafeString() throws Exception { byte[] bytes = "Hello World".getBytes(StandardCharsets.UTF_8); String encoded = Base64.encodeBase64UrlSafeString(bytes); Assert.assertEquals(encoded, "YXNlcjQ=="); bytes = new byte[]{0x00, 0x01, 0x02, 0x03, 0x'
# src_code = '@Test public void testEncodeBase64URLSafeString() throws Exception { byte[] bytes = "Hello World".getBytes(StandardCharsets.UTF_8); String encoded = Base64.encodeBase64UrlSafeString(bytes); Assert.assertEquals(encoded, "YXNlcjQ=="); bytes = new byte[]{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x1b, 0x2c, 0x3d, 0x4e, 0x5f, 0x6g, 0x7h, 0x8i, 0x9j, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x'
# src_code = 'testEncodeInteger() { assertArrayEquals(Base64.encodeInteger(BigInteger.valueOf(0)), new byte[] { 0, 1 }); assertArrayEquals("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".getBytes(), Base64.encodeString(new BigInteger("1234567890"))); assertArrayEquals("\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007\u0008\u0009\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017\u0018\u0019\u0020\u0021\u0022\u0023\u0024\u0025\u0026\u0027\u0028\u0029\u0030\u0031\u0032\u0033\u0034\u0035\u0036\u0037\u0038\u0039\u0040\u0041\u0042\u0043\u0044\u0045\u0046\u0047\u0048\u0049\u0050\u0051\u0052\u0053\u0054\u0055\u0056\u0057\u0058\u0059\u0060\u0061\u0062\u0063\u0064\u0065\u0066\u0067\u0068\u0069\u0070\u0071\u0072\u0073\u0074\u0075\u0076\u0077\u0078\u0079\u0080\u0081\u0082\u0083\u0084\u0085\u0086\u0087\u0088\u0089\u0090\u0091\u0092\u0093\u0094\u0095\u0096\u0097\u0098\u0099\u00100\u00101\u00102'

# src_code = src_code.encode("raw_unicode_escape").decode("utf-8", "ignore")

# src_code_decorated = "public class ClassTest { " + src_code + " }"

# print("\n")
# print(src_code_decorated)
# print("\n\n")

# _, _, _, fix_errors = parser_utils.fix_errors_in_code(src_code_decorated)
# print("\n")
# print(fix_errors)

# print(parser_utils.fix_errors_in_code(src_code_decorated))

# print("\n")
# print(src_code)
# print("\n\n")

# fix_all = parser_utils.fix_missings_and_errors_in_code(src_code)
# print("\n\n\nResult")
# print(fix_all)


