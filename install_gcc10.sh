#!/bin/bash
set -e

# Carpeta destino
INSTALL_DIR=$HOME/gcc-10
mkdir -p $INSTALL_DIR

# Preparar
cd $HOME
mkdir -p build_gcc && cd build_gcc

# Descargar GCC 10.3.0
wget https://ftp.gnu.org/gnu/gcc/gcc-10.3.0/gcc-10.3.0.tar.gz
tar -xzf gcc-10.3.0.tar.gz
cd gcc-10.3.0

# Descargar dependencias
./contrib/download_prerequisites

# Crear build dir
mkdir build && cd build

# Configurar instalación local
../configure --prefix=$INSTALL_DIR --disable-multilib --enable-languages=c,c++

# Compilar
make -k -j$(nproc)
make install