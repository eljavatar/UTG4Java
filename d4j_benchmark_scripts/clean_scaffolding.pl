#!/usr/bin/env perl

use warnings;
use strict;
use FindBin;
use File::Basename;
use Cwd qw(abs_path);
use Getopt::Std;
use Pod::Usage;
#use Text::CSV;
# Install module Text::CSV
# cpan Text::CSV

use lib abs_path("$FindBin::Bin/../core");
use Constants;
use Project;

#
# Process arguments and issue usage message if necessary.
#
my %cmd_opts;
getopts('c:s:', \%cmd_opts) or pod2usage(1);

pod2usage(1) unless defined $cmd_opts{c} and
                    defined $cmd_opts{s};

my $target_classes      = $cmd_opts{c};
my $path_scaffoldings   = $cmd_opts{s};

#print(STDERR "Getting Target classes: $target_classes\n");
#print(STDERR "Getting Path scaffoldings: $path_scaffoldings\n");

# Obtengo las clases que debo obtener en los scaffoldings
my %classes_to_extract = ();
open(IN, "<$target_classes") or die "Cannot read target classes";
while(<IN>) {
    chomp;
    $classes_to_extract{$_} = 1;
}
close(IN);

foreach my $class (keys %classes_to_extract) {
    my $file = $class;
    # Obtengo la ruta de la clase: example: from com.example.Foo => com/example/Foo
    $file =~ s/\./\//g;

    # Las clases en el archivo target_clases vienen con la forma com.example.Foo
    # En la ruta de path_scaffoldings vienen carpetas con dos archivos de la siguiente forma:
    # - com/example/Foo_ESTest.java
    # - com/example/Foo_ESTest_scaffolding.java
    #
    # Así pues, la idea es abrir los archivos com/example/Foo_ESTest.java a partir
    # de las clases que vienen en el archivo target_clases

    #my $file_class_src = "$project_root/$src_dir/$file_class.java";

    my $java_file = "${path_scaffoldings}/${file}_ESTest.java";

    my $class_name = $class;
    #$class_name =~ /.*\.(\w+)$/;
    # usamos una expresión regular para obtener el substring después del último punto
    $class_name =~ /.*\.([^\.]*)$/;
    $class_name = $1;
    $class_name = "${class_name}_ESTest";

    #print(STDERR "Getting Java File: $java_file\n");
    #print(STDERR "Getting Class: $class\n");
    #print(STDERR "Getting Class name: $class_name\n");

    #import org.mockito.Mock;
    #import org.mockito.Mockito;
    #import static org.mockito.BDDMockito.*;
    #import static org.assertj.core.api.Assertions.*;

    # Obtenemos las líenas del archivo .java
    open(IN, "<$java_file") or die "No se puede abrir el archivo java '$java_file' $!";
    my @lines_class = <IN>;
    close IN;

    _clean_class($java_file, $class_name, @lines_class);
}

sub _clean_class {
    my ($file, $class_name, @lines_class) = @_;

    # Abre el archivo para escritura
    open(OUT, ">$file") or die "No se pudo abrir el archivo '$file' $!";

    foreach my $linea (@lines_class) {
        if ($linea =~ /public class \Q$class_name\E/) {
            print OUT $linea;
            print OUT "\n    // Test Method generated\n\n}\n";
            last;
        } else {
            # Si no, escribe la línea original
            print OUT $linea;
        }
    }

    # Cierra el archivo
    close(OUT);
}
