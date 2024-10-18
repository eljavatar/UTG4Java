import tarfile
import os
import argparse
import time

def compress_file_to_tar_bz2(file_path, output_path):
    # Asegúrate de que el archivo existe
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe")

    # Abre el archivo tar.bz2 en modo de escritura
    with tarfile.open(output_path, "w:bz2") as tar:
        # Agrega el archivo input al archivo tar.bz2
        tar.add(file_path, arcname=os.path.basename(file_path))
        print(f"Comprimido {file_path} en {output_path}")


def compress_folder_to_tar_bz2(folder_path, output_path):
    # Asegúrate de que la carpeta existe
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"La carpeta {folder_path} no existe")

    # Abre el archivo tar.bz2 en modo de escritura
    with tarfile.open(output_path, "w:bz2") as tar:
        # Agrega la carpeta al archivo tar.bz2
        tar.add(folder_path, arcname=os.path.basename(folder_path))
        print(f"Comprimido {folder_path} en {output_path}")

# # Ruta de la carpeta que deseas comprimir
# folder_path = '/ruta/a/tu/carpeta'

# # Ruta del archivo tar.bz2 de salida
# output_path = '/ruta/a/tu/archivo_comprimido.tar.bz2'


def parse_args():
    """
    Parse the args passed from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path", 
        type=str,
        help="Folder input to compress",
    )
    parser.add_argument(
        "--output_path", 
        type=str,
        help="Output path of file tar.bz2",
    )
    parser.add_argument(
        "--type_input", 
        default="folder",
        type=str,
        help="file or older",
    )

    return vars(parser.parse_args())


def main():
    args = parse_args()
    input_folder_path = args['input_folder_path']
    output_path = args['output_path']
    type_input = args['type_input']

    start_time = time.time()

    if type_input == 'folder':
        compress_folder_to_tar_bz2(input_folder_path, output_path)
    else:
        compress_file_to_tar_bz2(input_folder_path, output_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_timeformatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("\n\n")
    print(f"Total time seconds: {elapsed_time}")
    print(f"Total time formatted: {elapsed_timeformatted}")


if __name__ == '__main__':
    main()