#!/usr/bin/env perl

use warnings;
use strict;
use Cwd;
use FindBin;
use File::Basename;
use Cwd qw(abs_path);
use Getopt::Std;
use Pod::Usage;
use Text::CSV;
use File::Copy;
use Archive::Tar;
use File::Find;
use File::Path qw(remove_tree);
use Data::Dumper;
# Install module Text::CSV
# cpan Text::CSV
use IPC::Open3; # Para ejecutar un proceso y capturar su salida
use Symbol 'gensym';

use lib abs_path("$FindBin::Bin/../core");
use Constants;
use Project;

use Time::HiRes qw(gettimeofday tv_interval);


my %cmd_opts;
getopts('p:v:f:s:o:t:F', \%cmd_opts) or pod2usage(1);

pod2usage(1) unless defined $cmd_opts{p} and 
                    defined $cmd_opts{f};

my $PID        = $cmd_opts{p};
my $VID        = $cmd_opts{v};
my $FILE_TESTS = $cmd_opts{f};
my $SCAFF      = $cmd_opts{s};
my $out_dir    = $cmd_opts{o};
my $TMP_DIR    = $cmd_opts{t};
my $FIX_SRC_CODE = defined $cmd_opts{F} ? 1 : 0;

if (!defined $SCAFF) {
    $SCAFF = "/scaffoldings/$PID";
}

if (!defined $out_dir) {
    $out_dir = "/generated-tests/$PID";
}
if (!-e $out_dir) {
    # Create output directory
    #system("mkdir -p $out_dir");
    Utils::exec_cmd("mkdir -p $out_dir", "Creating output directory $out_dir")
            or die("Failed to create output directory!");
}

if (!defined $TMP_DIR) {
    $TMP_DIR = "/tmp/generated-tests/$PID";
}
if (!-e $TMP_DIR) {
    Utils::exec_cmd("mkdir -p $TMP_DIR", "Creating temporary directory $TMP_DIR")
            or die("Failed to create temporary directory $TMP_DIR!");
}

# Guarda el directorio de trabajo actual
my $current_dir = getcwd;

# Guarda el tiempo de inicio
my $start_time = [gettimeofday];


# Crea un nuevo objeto CSV para leer el archivo
my $csv = Text::CSV->new({ binary => 1, auto_diag => 1 });
# Abre el archivo CSV
open my $file, "<:encoding(utf8)", "$FILE_TESTS" or die "No es posible abrir el archivo $FILE_TESTS: $!";
# Lee la línea de encabezado para obtener los nombres de las columnas
#my @columns = @{$csv->getline($file)};
my @columns = $csv->header($file);
# Crea una lista vacía para almacenar los objetos
my @list_generated_tests;
# Lee cada línea del archivo CSV
while (my $row = $csv->getline_hr($file)) {
    # Crea un nuevo objeto con los datos de la línea
    my $object = { map { $_ => $row->{$_} } @columns };

    # Agrega el objeto a la lista de objetos
    push @list_generated_tests, $object;
}
# Cierra el archivo CSV
close $file;

# Agrupamos por project, version, test_class
my %groups;

# Itera sobre cada objeto en la lista de objetos
for my $object (@list_generated_tests) {
    # Crea una clave de grupo con los valores de 'project', 'version' y 'test_class'
    my $group_key = join '|', @{$object}{qw(project version test_class)};

    # Agrega el objeto al grupo correspondiente
    push @{$groups{$group_key}}, $object;
}


# Tengo una lista de objetos en perl que obtengo a partir de un archivo CSV con la siguiente estructura:

# for my $object (@$list_methods_by_group) {
#     my $id = $object->{id};
#     my $project = $object->{project};
#     my $num_results = $object->{num_results};
#     my $result_1 = $object->{result_1};
#     my $result_2 = $object->{result_2};
#     my $result_n = $object->{result_n};
# }

# Donde la cantidad de columnas "result_i" depende del valor en num_results. Es decir, si num_results=4, tendré las columnas result_1, result_2, result_3 y result_4

# Puedes darme el código en Perl para obtener dinámicamente los campos result_i  a partir del valor de num_results?



# Itera sobre cada grupo en el hash de grupos
while (my ($group_key, $list_methods_by_group) = each %groups) {
    # Divide la clave del grupo en 'project', 'version' y 'test_class'
    #my ($project, $version, $modified_class) = split /\|/, $group_key;
    #my ($project, $version, $modified_class) = split("|", $group_key);
    my ($g_project, $g_version, $g_test_class) = split(/\|/, $group_key);

    next if ($PID ne $g_project);
    next if defined($VID) and ($VID ne $g_version);

    # Get file
    my $name_scaffolding = "${g_project}-${g_version}f-scaffolding.tar.bz2";
    my $scaffolding_file = "$SCAFF/$name_scaffolding";

    #copy($scaffolding_file, $TMP_DIR) or die "La copia del archivo falló: $!";

    #my @sort_columns = (
        #'id', 
        #'project', 
        #'version', 
        #'number_method_by_version',
        #'modified_class', 
        #'focal_class', 
        #'path_focal_class_in_project', 
        #'test_class', 
        #'path_test_class_scaffolding', 
        #'package_focal_class', 
        #'imports_focal_class', 
        #'method_to_test'
        #'src_fm_fc_ms_ff', 
        #'src_fm_fc_dctx', 
        #'src_fm_fc_dctx_priv'
        # num_results
    #);

    for my $object (@$list_methods_by_group) {
        my $id = $object->{id};
        my $project = $object->{project};
        my $version = $object->{version};
        my $number_method_by_version = $object->{number_method_by_version};
        my $modified_class = $object->{modified_class};
        my $focal_class = $object->{focal_class};
        my $path_focal_class_in_project = $object->{path_focal_class_in_project};
        my $test_class = $object->{test_class};
        my $path_test_class_scaffolding = $object->{path_test_class_scaffolding};
        my $package_focal_class = $object->{package_focal_class};
        my $imports_focal_class = $object->{imports_focal_class};
        my $num_results = $object->{num_results};

        # for my $i (1..$num_results) {
        for (my $i = 1; $i <= $num_results; $i++) {
            my $result_key = "generated_test_$i";
            my $generated_test_i = $object->{$result_key};
            if ($generated_test_i) {
                if ($FIX_SRC_CODE) {
                    my $code_before_changes = $generated_test_i;
                    # Valida si tiene error de sintaxis por missings y trata de fixearlos
                    my $has_syntax_error = _get_result_parser_utils($code_before_changes, "HAS_ERROR");
                    #print "Resultado has_syntax_error First (${version}--${number_method_by_version}): $has_syntax_error\n";

                    #my $code_fixed;
                    if ($has_syntax_error eq "Error") {
                        my $code_fixed = _get_result_parser_utils($code_before_changes, "FIX_MISSINGS");
                        $generated_test_i = $code_fixed;
                        #print "Resultado Fix_Missing (${version}--${number_method_by_version}): $generated_test_i\n";
                    }

                    # Valida de nuevo si no tiene error de sintaxis para adpatar name y modifiers
                    # $has_syntax_error = _get_result_parser_utils($generated_test_i, "HAS_ERROR");
                    # #print "Resultado has_syntax_error Second (${version}--${number_method_by_version}): $has_syntax_error\n";
                    # if ($has_syntax_error ne "Error") {
                    #     $generated_test_i = _get_result_parser_utils($generated_test_i, "ADAPT_MODIFIERS");
                    #     #print "Resultado Adapt_Modifiers (${version}--${number_method_by_version}): $generated_test_i\n";
                    # }
                    #print "\n\n";
                }

                _generate_scaffolding_by_suite($id, $project, $version, $i, $name_scaffolding, $scaffolding_file, $path_test_class_scaffolding, $package_focal_class, $imports_focal_class, $test_class, $generated_test_i, $out_dir, $current_dir);
            }
        }
    }
}

# Clean up temp_path
system("rm -rf $TMP_DIR");
system("rm -rf /tmp/generated-tests");

# Calcula el tiempo transcurrido
my $elapsed = tv_interval($start_time);
# Convierte el tiempo a horas, minutos y segundos
my $hours   = int($elapsed / 3600);
my $minutes = int(($elapsed % 3600) / 60);
my $seconds = int($elapsed % 60);
printf("\nTiempo de ejecucion project ${PID}: %02d:%02d:%02d\n\n", $hours, $minutes, $seconds);



sub _generate_scaffolding_by_suite() {
    my ($id, $project, $version, $result_suite, $name_scaffolding, $scaffolding_file, $path_test_class_scaffolding, $package_focal_class, $imports_focal_class, $test_class, $generated_test, $out_dir, $current_dir) = @_;

    my ($p, $v, $method_number, $class) = split(/_/, $id);

    # Create directory by method
    my $method_path = "$TMP_DIR/$version/$method_number/$result_suite";
    Utils::exec_cmd("mkdir -p $method_path", "Creating temporary directory (${result_suite}) for $id")
            or die("Failed to create temporary directory $id!");

    # Create copy of scaffolding in TMP_PATH
    copy($scaffolding_file, $method_path) or die "La copia del archivo $method_path falló: $!";

    my $temp_scaffolding = "$method_path/$name_scaffolding";

    # Cambia al directorio de destino
    chdir $method_path or die "No se pudo cambiar al directorio de destino del método $id: $!";

    # Extract file
    my $tar = Archive::Tar->new;
    $tar->read($temp_scaffolding, 1);
    $tar->extract;

    # Borramos el archivo de scaffolding que se había copiado previamente
    unlink $temp_scaffolding or die "No se pudo eliminar el archivo $name_scaffolding temporal del método $id: $!";
    
    # Obtenemos el path del scaffolding de la clase de test
    my $file_java_test = "$method_path/$path_test_class_scaffolding";
    #print(STDERR "Java file test: $file_java_test\n");
    #print(STDERR "Java class: $test_class\n");

    # Leemos las líneas de la clase java
    open(IN, "<$file_java_test") or die $!; my @lines_class = <IN>; close IN;

    if (! -e "$file_java_test.bak") {
        copy("$file_java_test", "$file_java_test.bak") or die "Cannot backup file ($file_java_test): $!";
    }

    # Agregamos imports custom
    my @imports_custom;
    push @imports_custom, "import org.junit.Rule;";
    push @imports_custom, "import org.junit.rules.Timeout;";
    _update_imports($id, $file_java_test, $package_focal_class, \@imports_custom, \@lines_class);
    # Releemos las líneas del archivo ya que han sido modificadas
    open(IN, "<$file_java_test") or die $!; @lines_class = <IN>; close IN;
    
    my @imports_list = split(/\|/, $imports_focal_class);
    # Elimina elementos vacíos (si el string original está vacío o contiene elementos vacíos)
    @imports_list = grep { $_ ne '' } @imports_list;
    # use Data::Dumper;
    # print Dumper(\@imports_list);
    my $has_changes = 0;
    if (@imports_list) {
        # Validamos si debemos reemplazar imports
        #_update_imports($id, $file_java_test, @imports_list, @lines_class);
        # Pasamos referencias en vez de las listas directamente para evitar que se mezclen
        $has_changes = _update_imports($id, $file_java_test, undef, \@imports_list, \@lines_class);
        if ($has_changes) {
            # Leemos de nuevo las líneas de la clase java ya que han sido modificadas
            open(IN, "<$file_java_test") or die $!; @lines_class = <IN>; close IN;
        }
    }

    # Reemplazamos el método generado en la clase scaffolding
    _replace_method($id, $result_suite, $file_java_test, $generated_test, @lines_class);


    # Eliminamos las otras clases y carpetas diferentes a la $test_class
    # Esto se hace porque cuando se genera el scaffolding, se genera por version,
    # y para una misma version pueden haber varias modified_class a las cuales se
    # les genera su respectivo scaffolding para cada una dentro del mismo zip

    # Encuentra todos los archivos y directorios en el directorio especificado
    find(sub {
        # Obtiene el nombre base del archivo actual
        my $nombre_base = basename($File::Find::name, '.java');
        #print(STDERR "File a eliminar: $nombre_base\n");

        # Escapa los caracteres no alfanuméricos en $test_class (ejemplo: $Gson$Types_ESTest)
        my $test_class_escaped = quotemeta($test_class);

        # Comprueba si el nombre base del archivo no comienza con $test_class
        if ($nombre_base !~ /^$test_class_escaped/ && -f) {
            # Si es un archivo, lo elimina
            #print(STDERR "Eliminamos el archivo: $nombre_base\n");
            unlink $File::Find::name or die "No se pudo eliminar el archivo $File::Find::name: $!\n";
        }
    }, $method_path);

    # Encuentra todos los directorios vacíos en el directorio especificado
    finddepth(sub {
        # Si es un directorio, no es el directorio actual y está vacío, lo elimina
        #if (-d && !glob($_.'/*')) {
        if (-d && $_ ne '.' && !glob($_.'/*')) {
            #print(STDERR "Eliminamos el directorio: $_\n");
            rmdir $_ or warn "No se pudo eliminar el directorio $_: $!\n";
        }
    }, $method_path);


    # Declaramos el nombre el comprimido
    #my $name_comprimido = "${project}-${version}f-generated_test-${method_number}.tar.bz2";
    my $name_comprimido = "${project}-${version}f-generated_test.${method_number}.$result_suite.tar.bz2";

    # Creamos el comprimido y lo copiamos en la carpeta de outputs
    system("tar -cjf $out_dir/$name_comprimido .");

    #print(STDERR "id: $id\n");
    #print(STDERR "focal_class: $focal_class\n");
    #print(STDERR "test_class: $test_class\n");
    #print(STDERR "generated_test: $generated_test\n\n");

    # Vuelve al directorio de trabajo actual
    chdir $current_dir or die "No se pudo volver al directorio de trabajo actual $current_dir: $!";

    # Clean up method_path
    system("rm -rf $method_path");
}



sub _replace_method {
    my ($id, $suite, $file, $method, @lines_class) = @_;

    # Backup file if necessary
    # if (! -e "$file.bak") {
    #     #printf ("\n\n Creating AGAIN BACKUP...\n\n");
    #     copy("$file", "$file.bak") or die "Cannot backup file ($file): $!";
    # }

    #my $rule_timeout = '@Rule\npublic Timeout globalTimeout = Timeout.seconds(10);'

    # Reemplaza la línea con el método generado
    for (@lines_class) {
        # Without timeout
        #s{// Test Method generated}{// Starting generated-test $id\n    $method\n    // Ending generated-test $id} if /\/\/ Test Method generated/;
        # Since JUnit 4.12
        #s{// Test Method generated}{\@Rule\n    public Timeout globalTimeout = Timeout.seconds(10);\n\n    // Starting generated-test $id\n    $method\n    // Ending generated-test $id} if /\/\/ Test Method generated/;
        # Before JUnit 4.12: https://github.com/junit-team/junit4/blob/main/src/main/java/org/junit/rules/Timeout.java
        s{// Test Method generated}{\@Rule\n    public Timeout globalTimeout = new Timeout(10000);\n\n    // Starting generated-test $id (Suite $suite)\n    $method\n    // Ending generated-test $id (Suite $suite)} if /\/\/ Test Method generated/;
    }

    # Escribe en el archivo
    open my $fh, '>', $file or die "No se pudo abrir el archivo $file para insertar el método $id: $!";
    print $fh @lines_class;
    close $fh;
}



sub _update_imports {
    my ($id, $file, $package_focal_class, $imports_ref, $lines_ref) = @_;
    my @imports_list = @{$imports_ref};  # Desreferenciar para obtener la lista
    my @lines_class = @{$lines_ref};     # Desreferenciar para obtener la lista

    #printf ("\n\n\n\n\nUpdating imports\n\n");

    my $package_str = "";
    my $line_package = 0;
    my $has_package = 0;
    my %existing_imports;
    my $line_first_import = 0;
    my $import_already = 0;
    my $has_changes = 0;

    for (my $i=0; $i<=$#lines_class; ++$i) {
        if ($lines_class[$i] =~ /^\s*package\s+/) {
            $line_package = $i;
            $has_package = 1;
            $package_str = $lines_class[$i];
        }
        if ($lines_class[$i] =~ /^\s*import\s+/) {
            $existing_imports{$lines_class[$i]} = 1;
            if (!$import_already) {
                $line_first_import = $i;
                $import_already = 1;
            }
        }
    }

    my @imports_to_add;
    foreach my $import (@imports_list) {
        unless (exists $existing_imports{"$import\n"}) {
            push @imports_to_add, "$import\n";
        }
    }

    my @new_lines;
    if (@imports_to_add) {
        $has_changes = 1;
        if ($has_package) {
            for (my $i=0; $i<=$line_package; ++$i) {
                push @new_lines, $lines_class[$i];
            }
            push @new_lines, "\n";

            foreach my $import (@imports_to_add) {
                push @new_lines, $import;
            }

            for (my $i=$line_package+1; $i<=$#lines_class; ++$i) {
                push @new_lines, $lines_class[$i];
            }
        } else {
            if ($package_focal_class) {
                push @new_lines, "$package_focal_class\n\n";
            }

            foreach my $import (@imports_to_add) {
                push @new_lines, $import;
            }
            push @new_lines, "\n";

            if ($import_already) {
                for (my $i=$line_first_import+1; $i<=$#lines_class; ++$i) {
                    push @new_lines, $lines_class[$i];
                }
            } else {
                for (my $i=0; $i<=$#lines_class; ++$i) {
                    push @new_lines, $lines_class[$i];
                }
            }
        }
    } else {
        if (!$has_package && $package_focal_class) {
            $has_changes = 1;
            push @new_lines, "$package_focal_class\n\n";
            for (my $i=0; $i<=$#lines_class; ++$i) {
                push @new_lines, $lines_class[$i];
            }
        } else {
            @new_lines = @{$lines_ref}
        }
    }

    # printf ("\n\n line_first_import\n\n");
    # printf ($line_first_import);

    # printf ("\n\n line_package\n\n");
    # printf ($line_package);

    # printf ("\n\n existing_imports\n\n");
    # print Dumper(\%existing_imports);

    # printf ("\n\n imports_to_add\n\n");
    # print Dumper(\@imports_to_add);

    # printf ("\n\n new_lines\n\n");
    # print Dumper(\@new_lines);

    if ($has_changes) {
        # Backup file if necessary
        #printf ("\n\n Creating BACKUP...\n\n");
        # if (! -e "$file.bak") {
        #     copy("$file", "$file.bak") or die "Cannot backup file ($file): $!";
        # }

        #printf ("\n\n Overriding FILE...\n\n");
        # Escribe en el archivo
        open my $fh, '>', $file or die "No se pudo abrir el archivo $file para actualizar imports para el método $id: $!";
        print $fh @new_lines;
        close $fh;
    }

    return $has_changes;
}


sub _get_result_parser_utils {
    my ($src_code, $function) = @_;

    my $python_script = 'parser_utils.py';
    #my $src_code = 'example source code';
    # Construimos la línea de comando
    my @command = (
        'python3', $python_script,
        #'--java_file', $test_java_file,
        '--src_code', $src_code,
        '--function', $function
    );

    # Ejecutamos el script Python y capturamos la salida
    my $stdout = gensym;
    my $stderr = gensym;
    my $pid = open3(undef, $stdout, $stderr, @command);
    #my $pid = open3(undef, $stdout, $stderr, 'python3', $python_script, $src_code);

    # Leemos la salida estándar
    my $res_script_python = do { local $/; <$stdout> };
    chomp $res_script_python;

    # Leemos la salida de error (si hay)
    my $error_output = do { local $/; <$stderr> };
    chomp $error_output;

    # Esperamos a que el proceso termine
    waitpid($pid, 0);

    # Imprimimos el resultado
    #print "Resultado del script Python: $res_script_python\n";
    if ($error_output) {
        print "Errores del script Python: $error_output\n";
    }

    return $res_script_python;
}


=pod
foreach (@list_generated_tests) {
    # Obtenemos la info de cada archivo de tests
    my $id = $_->{id};
    my $project = $_->{project};
    my $version = $_->{version};
    my $modified_class = $_->{modified_class};
    my $focal_class = $_->{focal_class};
    my $test_class = $_->{test_class};
    my $path_test_class_scaffolding = $_->{path_test_class_scaffolding};
    my $generated_test = $_->{generated_test};

    next if ($PID ne $project);
    next if defined($VID) and ($VID ne $version);

    # Obtenemos el archivo scaffolding para la versión
}
=cut
