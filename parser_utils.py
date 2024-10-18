import copy
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node
from typing import List, Dict, Any, Set, Optional
from packaging import version
# https://setuptools.pypa.io/en/latest/pkg_resources.html
import pkg_resources


class ParserUtils():

    def __init__(self, input_encoding):
        #print(f"\n\n Version tree-sitter: {str(tree_sitter.__version__)} \n\n")
        tree_sitter_version = pkg_resources.get_distribution("tree_sitter").version
        #print(f"\n\n Version tree-sitter: '{str(tree_sitter_version)}' \n\n")

        if version.parse(str(tree_sitter_version)) < version.parse("0.22.0"):
            JAVA_LANGUAGE = Language(tsjava.language(), "java")
        else:
            JAVA_LANGUAGE = Language(tsjava.language())
        
        self.parser = Parser()
        self.parser.set_language(JAVA_LANGUAGE)
        self.input_encoding = input_encoding
    

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

        return code_without_changes_has_errors, edited_code


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
        src_encoded_bytes = bytes(src_code, self.input_encoding)
        src_decoded = src_encoded_bytes.decode(self.input_encoding)

        clean_code = src_decoded

        tree = self.parser.parse(src_encoded_bytes)
        content_node_main = tree.root_node

        type_return = "void"
        name_method = "some"
        # Clean all modifiers
        for child in content_node_main.children:
            if child.type == "method_declaration":
                for _child in child.children:
                    if _child.type == "modifiers":
                        #metadata['modifiers']  = ' '.join(ParserUtils.match_from_span(child, blob).split())
                        modifiers_text = _child.text.decode(self.input_encoding)
                        # print(modifiers_text)
                        clean_code = clean_code.replace(modifiers_text, " ")
                        clean_code = clean_code.strip()
                        #print(clean_code)
                    
                    if _child.type == "type_parameters":
                        type_parameter = _child.text.decode(self.input_encoding)
                        clean_code = clean_code.replace(type_parameter, " ")
                        clean_code = clean_code.strip()
                        #print(clean_code)
                    
                    if "type" in _child.type and _child.type != "type_parameters":
                        type_return = _child.text.decode(self.input_encoding)
                        length_type_return = len(type_return)
                        #print(f"Type {str(length_type_return)}: " + type_return)
                        clean_code = clean_code[length_type_return:]
                        clean_code = clean_code.strip()
                        #print(clean_code)
                    
                    if _child.type == "identifier":
                        name_method = _child.text.decode(self.input_encoding)
                        length_identifier = len(name_method)
                        #print(f"Identifier {str(length_identifier)}: " + name_method)
                        clean_code = clean_code[length_identifier:]
                        clean_code = clean_code.strip()
                        #print(clean_code)
        
        if not name_method.startswith("test"):
            #print("Name before: " + name_method)
            name_method = name_method[0].upper() + name_method[1:]
            # capitalized_word = '{}{}'.format(word[0].upper(), word[1:])
            name_method = "test" + name_method
            #print("Name after: " + name_method)

        clean_code = "@Test public " + type_return + " " + name_method + clean_code
        #print("\n\n\n")
        try:
            clean_code = clean_code.encode(self.input_encoding).decode("utf-8")
        except:
            pass
        
        return clean_code


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
    

    @staticmethod
    def traverse_tree_missing(node: Node, results: List):
        for n in node.children:
            if n.is_missing: 
                results.append(n)
            ParserUtils.traverse_tree_missing(n, results)


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
#     method = "@Test public void toString() throws NoSuchFieldException { Field[] fields = Field.getDeclaredFields(); TypeInfo typeInfo = TypeInfoFactory.getTypeInfoForField(fields.get(0), String.class); assertThat(typeInfo.getActualType()).isEqualTo(String.class); assertThat(typeInfo.getType()).isEqualTo(String.class); }"
#     print("\nBefore:")
#     print(method)

#     parserUtils = ParserUtils("utf-8")
#     code_without_missings = parserUtils.adapt_name_and_modifiers(method)

#     print("\n\nAfter:")
#     print(code_without_missings)
