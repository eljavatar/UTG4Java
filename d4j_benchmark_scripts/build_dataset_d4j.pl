#!/usr/bin/env perl

use warnings;
use strict;
use FindBin;
use File::Basename;
use Cwd qw(abs_path);
use Getopt::Std;
use Pod::Usage;
use JSON;
use Text::CSV;
# Install module Text::CSV
# cpan Text::CSV

use lib abs_path("$FindBin::Bin/../core");
use Constants;
use Project;


my %cmd_opts;
getopts('p:v:t:s:d:o:', \%cmd_opts) or pod2usage(1);

pod2usage(1) unless defined $cmd_opts{p};
my $PID        = $cmd_opts{p};
my $VID        = $cmd_opts{v};
my $path_scaff = $cmd_opts{s};
my $path_data  = $cmd_opts{d};
my $out_dir    = $cmd_opts{o};

my $TMP_DIR = Utils::get_tmp_dir($cmd_opts{t});
system("mkdir -p $TMP_DIR");

my $project = Project::create_project($PID);
$project->{prog_root} = $TMP_DIR;

my $project_dir = "$PROJECTS_DIR/$PID";
my $path_scaffoldings = "$path_scaff/${PID}";
my $path_input_corpus = "$path_data/${PID}";

# Create output directory
#system("mkdir -p $TMP_DIR");
Utils::exec_cmd("mkdir -p $out_dir", "Creating output directory")
        or die("Failed to create output directory!");


# Get ids of fixed versions
my @ids;
if (defined $VID) {
    $VID =~ /^(\d+)$/ or die "Wrong version_id format: $VID! Expected: \\d+";
    #my $FID = Utils::check_vid("${VID}f")->{bid};
    $project->contains_version_id("${VID}f") or die "Version id (${VID}f) does not exist in project: $PID";
    @ids = ($VID);
} else {
    @ids = $project->get_bug_ids();
}


my @lista_de_hashes = ();


foreach my $id (@ids) {
    my $vid = "${id}f";

    my $file_scaffolding = "${path_scaffoldings}/${PID}-${vid}-scaffolding.tar.bz2";
    # si el scaffolding de la versión no existe, saltamos a la siguiente versión
    if (!-e $file_scaffolding) {
        next;
    }

    # Hash all modified classes
    my %modified_classes = ();
    open(IN, "<${project_dir}/modified_classes/${id}.src") or die "Cannot read modified classes";
    while(<IN>) {
        chomp;
        $modified_classes{$_} = 1;
    }
    close(IN);

    my $path_methods = "${path_input_corpus}/${id}/methods";
    #print(STDERR "Path methods: $path_methods\n");

    # si el folder de los métodos en el corpus de entrada y para la versión indicada no existe,
    # saltamos a la siguiente versión
    if (!-e $path_methods) {
        next;
    }
    
    # Hash all methods
    my %methods_in_corpus = ();
    opendir(my $dir, $path_methods) or die "No se pudo abrir el directorio '$path_methods' $!";
    # Itera sobre cada carpeta en el directorio, donde cada carpera corresponde a un método
    while (my $number_method = readdir $dir) {
        # Ignora las entradas '.' y '..'
        next if ($number_method =~ /^\.+$/);

        # my $file_method = "$path_methods/$number_method/corpus.txt";

        # # Verifica si el archivo corpus.txt existe
        # if (-e $file_method) {
        #     # Abre el archivo para lectura
        #     open(my $fh, '<', $file_method) or die "No se pudo abrir el archivo '$file_method' $!";
        #     # Lee el contenido del archivo
        #     my $contenido = do { local $/; <$fh> };
        #     # Cierra el archivo
        #     close($fh);

        #     $methods_in_corpus{$number_method} = $contenido;
        # }

        my $path_json_file_method = "$path_methods/$number_method/corpus.json";
        # Verifica si el archivo corpus.json existe
        if (-e $path_json_file_method) {
            # Abre el archivo para lectura
            open(my $fh, '<', $path_json_file_method) or die "No se pudo abrir el archivo '$path_json_file_method' $!";
            # Lee el contenido del archivo
            my $json_text = do { local $/; <$fh> };
            # Decodifica el JSON en una estructura de datos Perl
            my $json_data = decode_json($json_text);
            # Cierra el archivo
            close($fh);

            # Imprimir la estructura de datos Perl (para ver el contenido)
            #use Data::Dumper;
            #print Dumper($json_data);

            $methods_in_corpus{$number_method} = $json_data;
        }
    }

    my $autonumber = 1;

    foreach my $class (keys %modified_classes) {
        my $file_class = $class;
        # Obtengo la ruta de la clase: example: from com.example.Foo => com/example/Foo
        $file_class =~ s/\./\//g;
        
        my $focal_class = $class;
        # usamos una expresión regular para obtener el substring después del último punto
        $focal_class =~ /.*\.([^\.]*)$/;
        my $focal_class_name = $1;

        my $test_class_scaffolding_name = "${focal_class_name}_ESTest";
        my $path_class_scaffolding = "${file_class}_ESTest.java";

        #foreach my $key (keys %methods_in_corpus) {
        #foreach my $value (values %methods_in_corpus) {
        #while (my ($key, $value) = each %methods_in_corpus) {
        while (my ($method_number, $method_corpus) = each %methods_in_corpus) {
            # my $class_in_method = $focal_class_name;
            # if ($class_in_method =~ /\$/) {
            #     ($class_in_method) = $class_in_method =~ /\$([^\$]*)$/;
            # }

            my $path_focal_class_in_corpus = $method_corpus->{path_focal_class};
            #print(STDERR "Class in method: $path_focal_class_in_corpus\n");

            # Extraemos el nombre del archivo .java con extensión
            my ($focal_class_in_corpus_with_ext) = $path_focal_class_in_corpus =~ /\/([^\/]+)$/;
            # Extraemos el nombre del archivo .java sin extensión
            my ($focal_class_in_corpus_without_ext) = $focal_class_in_corpus_with_ext =~ /^(.*?)\./;

            #print(STDERR "Class in method: $class_in_method\n");
            #print(STDERR "Focal class: $focal_class_name\n");
            #print(STDERR "Test class: $test_class_scaffolding_name\n");
            #print(STDERR "Path scaffolding: $path_class_scaffolding\n");
            #print(STDERR "Method to Test: $method_corpus\n");
            #print(STDERR "\n");

            # if ($method_corpus =~ /^$class_in_method/) {
            #if ($focal_class_in_corpus_without_ext ne $focal_class_name) {
            if ($focal_class_in_corpus_without_ext eq $focal_class_name) {
                #my $autonumber_id = "${PID}_${id}_${autonumber}_${focal_class_name}";
                my $method_id = "${PID}_${id}_${method_number}_${focal_class_name}";

                #print "Valor del contador: $autonumber_id\n";
                #print(STDERR "Id autonumber: $autonumber_id\n");

                #$hash_row->{id} = $autonumber_id;
                #$hash_row->{method_to_test} = $method_to_test;

                my $hash_row = {
                    #id => $autonumber_id,
                    id => $method_id,
                    project => $PID,
                    version => $id, 
                    number_method_by_version => $method_number,
                    modified_class => $class,
                    focal_class => $focal_class_name,
                    path_focal_class_in_project => $method_corpus->{path_focal_class},
                    test_class => $test_class_scaffolding_name,
                    path_test_class_scaffolding => $path_class_scaffolding,
                    package_focal_class => $method_corpus->{package_focal_class},
                    imports_focal_class => $method_corpus->{imports_focal_class},
                    #method_to_test => $method_corpus
                    src_fm_fc_ms_ff => $method_corpus->{src_fm_fc_ms_ff},
                    src_fm_fc_dctx => $method_corpus->{src_fm_fc_dctx},
                    src_fm_fc_dctx_priv => $method_corpus->{src_fm_fc_dctx_priv}
                };

                push(@lista_de_hashes, $hash_row);

                $autonumber++;
            }
        }
    }
}

# Ordenamos la lista de hashes
@lista_de_hashes = sort {
    # Comparar por 'version' primero
    $a->{version} <=> $b->{version}
    ||
    # Comparar por 'number_method_by_version' si 'version' es igual
    $a->{number_method_by_version} <=> $b->{number_method_by_version}
} @lista_de_hashes;


# Creamos una instancia de Text::CSV
my $csv = Text::CSV->new({ binary => 1, eol => "\n" });

# Abrimos un archivo para escribir
open my $file, ">", "${out_dir}/${PID}_methods.csv" or die "Cannot write methods hashes in CSV";
# Especificamos el orden de las columnas
my @sort_columns = (
    'id', 
    'project', 
    'version', 
    'number_method_by_version',
    'modified_class', 
    'focal_class', 
    'path_focal_class_in_project', 
    'test_class', 
    'path_test_class_scaffolding', 
    'package_focal_class', 
    'imports_focal_class', 
    #'method_to_test'
    'src_fm_fc_ms_ff', 
    'src_fm_fc_dctx', 
    'src_fm_fc_dctx_priv'
);
# Escribimos los nombres de las columnas en la primera línea
$csv->print($file, \@sort_columns);
foreach my $hash (@lista_de_hashes) {
    # Escribimos cada valor en el archivo
    $csv->print($file, [@{$hash}{@sort_columns}]);
}
# Cerramos el archivo
close $file;

# Clean up
system("rm -rf $TMP_DIR");
