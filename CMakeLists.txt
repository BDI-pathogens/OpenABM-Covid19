cmake_minimum_required(VERSION 3.15)
project(covid19ibm C)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_STANDARD 11)

find_package(GSL)
include_directories(src)
include_directories(${GSL_INCLUDE_DIRS})

add_executable(covid19ibm
        src/constant.c
        src/constant.h
        src/demographics.c
        src/demographics.h
        src/disease.c
        src/disease.h
        src/doctor.c
        src/doctor.h
        src/hospital.c
        src/hospital.h
        src/individual.c
        src/individual.h
        src/input.c
        src/input.h
        src/interventions.c
        src/interventions.h
        src/main.c
        src/model.c
        src/model.h
        src/network.c
        src/network.h
        src/nurse.c
        src/nurse.h
        src/params.c
        src/params.h
        src/structure.h
        src/utilities.c
        src/utilities.h
        src/ward.c
        src/ward.h)

target_link_libraries(covid19ibm ${GSL_LIBRARIES})
