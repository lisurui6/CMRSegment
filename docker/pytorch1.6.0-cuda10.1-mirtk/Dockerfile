FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get install -y build-essential git libjpeg-dev && \
    apt-get install -y vim tmux

RUN pip install --upgrade pip setuptools && \
    pip install nibabel vedo scikit-image tqdm pyhocon SimpleITK pandas tensorboard


ARG THREADS
ARG VTK_VERSION=8.2.0
ARG EIGEN_VERSION=3.3.7
ARG CXX_STANDARD=c++14

RUN CXX_STANDARD=c++14 EIGEN_VERSION=3.3.7 VTK_VERSION=8.2.0 /bin/sh -c NUM_CPUS=${THREADS:-`cat /proc/cpuinfo | grep processor | wc -l`} \
    && echo "Maximum number of build threads = $NUM_CPUS" \
    && apt-get update && apt-get install -y --no-install-recommends software-properties-common  \
    && apt-get update && apt-get install -y --no-install-recommends wget gcc g++ make cmake python3 freeglut3-dev \
    libarpack2-dev libboost-math-dev libboost-random-dev libflann-dev libfftw3-dev libgtest-dev libinsighttoolkit4-dev \
    libnifti-dev libpng-dev libsuitesparse-dev libtbb-dev uuid-dev zlib1g-dev \
    && mkdir /usr/src/gtest/build && cd /usr/src/gtest/build && cmake .. && make -j $NUM_CPUS && \
    mv -f libgtest.a libgtest_main.a /usr/lib && cd /usr/src && rm -rf /usr/src/gtest/build

RUN if [ -z ${VTK_VERSION} ]; then apt-get install -y libeigen3-dev; else EIGEN_SOURCE_DIR=/usr/src/eigen-${EIGEN_VERSION} && \
    mkdir ${EIGEN_SOURCE_DIR} /usr/include/eigen3 && cd ${EIGEN_SOURCE_DIR} && \
    wget -O archive.tar.bz2 https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2 && \
    tar vxjf archive.tar.bz2 --strip 1 && mv signature_of_eigen3_matrix_library Eigen /usr/include/eigen3/ && \
    mv cmake/FindEigen3.cmake /usr/share/cmake-3.10/Modules/ && cd /usr/src && rm -rf ${EIGEN_SOURCE_DIR}; fi

RUN if [ -z ${VTK_VERSION} ]; then apt-get install -y libvtk7-dev; else VTK_RELEASE=`echo ${VTK_VERSION} | sed s/\.[0-9]*$//` && \
    cd /usr/src && wget http://www.vtk.org/files/release/${VTK_RELEASE}/VTK-${VTK_VERSION}.tar.gz && \
    tar -xvzf VTK-${VTK_VERSION}.tar.gz && rm -f VTK-${VTK_VERSION}.tar.gz && mkdir VTK-${VTK_VERSION}/Build && \
    cd VTK-${VTK_VERSION}/Build && cmake -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=Release  \
    -D CMAKE_CXX_STANDARD=${CXX_STANDARD} -D VTK_USE_SYSTEM_PNG=ON -D VTK_USE_SYSTEM_ZLIB=ON -D BUILD_SHARED_LIBS=ON \
    -D BUILD_EXAMPLES=OFF -D BUILD_TESTING=OFF -D BUILD_DOCUMENTATION=OFF ..  && \
    make install && cd /usr/src && rm -rf /usr/src/VTK-${VTK_VERSION} && ldconfig; fi && rm -rf /var/lib/apt/lists/*

RUN cd /workspace && git clone https://github.com/BioMedIA/MIRTK.git && cd ./MIRTK && git submodule update --init && cd .. && mv MIRTK /usr/src/mirtk

RUN BUILD_TESTING=OFF /bin/sh -c ls /usr/src/mirtk && NUM_CPUS=${THREADS:-`cat /proc/cpuinfo | grep processor | wc -l`} && \
    echo "Maximum number of build threads = $NUM_CPUS" && mkdir /usr/src/mirtk/Build && cd /usr/src/mirtk/Build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=ON \
    -D BUILD_APPLICATIONS=ON -D BUILD_TESTING=${BUILD_TESTING} -D BUILD_DOCUMENTATION=OFF -D BUILD_CHANGELOG=OFF \
    -D MODULE_Common=ON -D MODULE_Numerics=ON -D MODULE_Image=ON -D MODULE_IO=ON -D MODULE_PointSet=ON \
    -D MODULE_Transformation=ON -D MODULE_Registration=ON -D MODULE_Deformable=ON -D MODULE_DrawEM=ON \
    -D MODULE_Mapping=ON -D MODULE_Scripting=ON -D MODULE_Viewer=OFF -D WITH_ARPACK=ON -D WITH_FLANN=ON \
    -D WITH_MATLAB=OFF -D WITH_NiftiCLib=ON -D WITH_PNG=ON -D WITH_PROFILING=ON -D WITH_TBB=ON -D WITH_UMFPACK=ON \
    -D WITH_ITK=ON -D WITH_VTK=ON -D WITH_ZLIB=ON ..

RUN cd /usr/src/mirtk/Build && \
    make install && cd /usr/src && rm -rf /usr/src/mirtk && echo "/usr/local/lib/mirtk" > /etc/ld.so.conf.d/mirtk.conf && ldconfig
